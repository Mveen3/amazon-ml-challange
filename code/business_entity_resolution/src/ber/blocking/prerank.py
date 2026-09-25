"""Stage 1b: cheap-feature pre-ranker (OOF) + score floor tuned on the ceiling.

Keep rule for a pair (e, r):
  p_pre >= floor  OR  (r ranks e in its top ``keep_rev_top`` S1s and p_pre >= floor_low)
then at most ``max_per_s1`` pairs per S1 by p_pre. The floor is the largest grid
value whose ceiling stays within ``ceiling_tol`` of the unfiltered union.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ..eval.metric import ceiling
from ..models.gbdt import load_folds, predict_by_fold, to_matrix, train_oof
from ..utils import chunks, ensure_dir, inference_only, load_json, log, n_workers, save_json, work_dir
from .keys import KEY_TYPES

RANK_COLS = ["a1f_rank", "a1r_rank", "n1f_rank", "n1r_rank", "d1f_rank", "d1r_rank"]
PRE_FEATURES = (RANK_COLS + ["a1_sim", "n1_sim", "d1_sim", "a1_cos", "n1_cos", "key_block", "key_n"]
                + [f"key_{k}" for k in KEY_TYPES]
                + ["name_ratio", "name_tsr", "addr_tsr", "num_hit", "pin_eq", "pin_diff", "legal_eq", "src",
                   "rec_addr_missing", "rec_script", "rec_is_domain", "s1_name_freq", "s1_addr_share",
                   "s1_ncand", "rec_ncand", "hop2"])

_NORM = ["uid", "name_core", "addr_latin", "legal", "num_primary", "nums", "pin", "script", "is_domain",
         "addr_missing", "name_core_sorted", "addr_key"]


def s1_stats(cfg, split: str) -> pl.DataFrame:
    """Ambiguity of each S1: how many S1s share its core name / exact address (per country)."""
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src", "country"])
    norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=["uid", "name_core_sorted", "addr_key",
                                                                         "addr_missing"])
    s = rec.filter(pl.col("src") == 1).join(norm, on="uid")
    s = s.with_columns([
        pl.len().over(["country", "name_core_sorted"]).cast(pl.Int32).alias("s1_name_freq"),
        pl.when(pl.col("addr_missing")).then(0).otherwise(pl.len().over(["country", "addr_key"]))
        .cast(pl.Int32).alias("s1_addr_share"),
    ])
    return s.select([pl.col("uid").alias("s1_uid"), "s1_name_freq", "s1_addr_share"])


def attach_norm(pairs: pl.DataFrame, norm: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    n = norm.select(["uid"] + cols)
    pairs = pairs.join(n.rename({c: f"s_{c}" for c in cols}).rename({"uid": "s1_uid"}), on="s1_uid", how="left")
    return pairs.join(n.rename({c: f"r_{c}" for c in cols}).rename({"uid": "rec_uid"}), on="rec_uid", how="left")


def _fuzzy(a: list[str], b: list[str], scorer, workers: int) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def cheap_features(pairs: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    """Add PRE_FEATURES columns to ``pairs`` (s1_uid, rec_uid, channel columns[, hop2])."""
    norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=_NORM)
    src = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src"])
    stats = s1_stats(cfg, split)
    # candidate-list sizes need the whole table; the string-heavy part is done in row chunks
    pairs = pairs.with_columns([pl.len().over("s1_uid").cast(pl.Int32).alias("s1_ncand"),
                                pl.len().over("rec_uid").cast(pl.Int32).alias("rec_ncand")])
    workers = n_workers(cfg)
    parts = []
    for s, e in chunks(pairs.height, int(cfg.prerank.get("chunk_rows", 20_000_000))):
        df = attach_norm(pairs.slice(s, e - s), norm, [c for c in _NORM if c != "uid"])
        df = df.join(src.rename({"uid": "rec_uid"}), on="rec_uid", how="left").join(stats, on="s1_uid", how="left")
        parts.append(_cheap_chunk(df, pairs.columns, workers))
    return pl.concat(parts).sort(["s1_uid", "rec_uid"])


def _cheap_chunk(df: pl.DataFrame, pair_cols: list[str], workers: int) -> pl.DataFrame:
    a, b = df["s_name_core"].to_list(), df["r_name_core"].to_list()
    df = df.with_columns([
        pl.Series("name_ratio", _fuzzy(a, b, fuzz.ratio, workers)),
        pl.Series("name_tsr", _fuzzy(a, b, fuzz.token_set_ratio, workers)),
        pl.Series("addr_tsr", _fuzzy(df["s_addr_latin"].to_list(), df["r_addr_latin"].to_list(),
                                     fuzz.token_set_ratio, workers)),
    ])
    if "hop2" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias("hop2"))
    for c in ["d1f_rank", "d1r_rank", "d1_sim", "key_mask", "key_block", "key_n"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias(c))
    df = df.with_columns([
        *[((pl.col("key_mask").fill_null(0).cast(pl.Int64) // (1 << i)) % 2).cast(pl.Int8).alias(f"key_{k}")
          for i, k in enumerate(KEY_TYPES)],
        (pl.col("r_nums").list.contains(pl.col("s_num_primary")) & (pl.col("s_num_primary") != ""))
        .fill_null(False).cast(pl.Int8).alias("num_hit"),
        ((pl.col("s_pin") == pl.col("r_pin")) & (pl.col("s_pin") != "")).cast(pl.Int8).alias("pin_eq"),
        ((pl.col("s_pin") != pl.col("r_pin")) & (pl.col("s_pin") != "") & (pl.col("r_pin") != ""))
        .cast(pl.Int8).alias("pin_diff"),
        ((pl.col("s_legal") == pl.col("r_legal")) & (pl.col("s_legal") != "")).cast(pl.Int8).alias("legal_eq"),
        pl.col("r_addr_missing").cast(pl.Int8).alias("rec_addr_missing"),
        pl.col("r_script").cast(pl.Int8).alias("rec_script"),
        pl.col("r_is_domain").cast(pl.Int8).alias("rec_is_domain"),
    ])
    keep = list(pair_cols) + [c for c in PRE_FEATURES if c not in pair_cols]
    return df.select(list(dict.fromkeys(keep)))


def _labels(pairs: pl.DataFrame, cfg) -> pl.DataFrame:
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet")).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "fold"])
    return (pairs.join(truth, on=["s1_uid", "rec_uid"], how="left").with_columns(pl.col("label").fill_null(0))
            .join(s1, on="s1_uid", how="left").sort(["s1_uid", "rec_uid"]))


def _select(df: pl.DataFrame, floor: float, pc) -> pl.DataFrame:
    df = df.with_columns(pl.col("p_pre").rank("ordinal", descending=True).over("rec_uid").alias("_rrank"))
    kept = df.filter((pl.col("p_pre") >= floor) |
                     ((pl.col("_rrank") <= int(pc.keep_rev_top)) & (pl.col("p_pre") >= float(pc.floor_low))))
    kept = kept.with_columns(pl.col("p_pre").rank("ordinal", descending=True).over("s1_uid").alias("_srank"))
    return kept.filter(pl.col("_srank") <= int(pc.max_per_s1)).drop(["_rrank", "_srank"])


def run_prerank(cfg) -> None:
    pc = cfg.prerank
    mdir = ensure_dir(work_dir(cfg, None, "models", "prerank"))
    if not inference_only(cfg):
        _train_prerank(cfg, mdir)
    chosen = float(load_json(mdir / "floor.json")["floor"])
    # ---------------- test: fold-average
    models = load_folds(mdir)
    union = pl.read_parquet(work_dir(cfg, "test", "cands", "union.parquet"))
    te = cheap_features(union, cfg, "test").with_columns(pl.lit(-1, dtype=pl.Int8).alias("fold"))
    te = te.with_columns(pl.Series("p_pre", predict_by_fold(models, to_matrix(te, PRE_FEATURES), te["fold"].to_numpy())))
    out = work_dir(cfg, "test", "cands")
    te.write_parquet(out / "pre_all.parquet")
    _select(te, chosen, pc).write_parquet(out / "pre.parquet")


def _train_prerank(cfg, mdir) -> None:
    pc = cfg.prerank
    union = pl.read_parquet(work_dir(cfg, "train", "cands", "union.parquet"))
    tr = _labels(cheap_features(union, cfg, "train"), cfg)
    X = to_matrix(tr, PRE_FEATURES)
    oof, _ = train_oof(X, tr["label"].to_numpy(), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy(), pc.gbdt,
                       PRE_FEATURES, mdir, seed=int(cfg.run.seed), threads=n_workers(cfg))
    tr = tr.with_columns(pl.Series("p_pre", oof))
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
    base = ceiling(tr, truth, s1)
    chosen, grid = float(pc.floor_grid[0]), []
    for fl in sorted(float(x) for x in pc.floor_grid):
        sel = _select(tr, fl, pc)
        c = ceiling(sel, truth, s1)
        grid.append({"floor": fl, "ceiling": c, "pairs_per_s1": sel.height / s1.height})
        log().info("   floor %.4f: ceiling %.5f, %.2f pairs/S1", fl, c, sel.height / s1.height)
        if c >= base - float(pc.ceiling_tol):
            chosen = fl
    save_json({"union_ceiling": base, "floor": chosen, "grid": grid}, mdir / "floor.json")
    log().info("  union ceiling %.5f -> chosen floor %.4f", base, chosen)
    out = work_dir(cfg, "train", "cands")
    tr.drop("label").write_parquet(out / "pre_all.parquet")
    _select(tr, chosen, pc).drop("label").write_parquet(out / "pre.parquet")
