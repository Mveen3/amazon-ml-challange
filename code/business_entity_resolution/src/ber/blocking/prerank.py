"""Stage 1b: cheap-feature pre-ranker (OOF) + score floor tuned on the ceiling.

Memory-lean: the pre-ranker trains on a sample of S1 entities (``sample_entities``) and then scores
the union files chunk by chunk, keeping only pairs whose p_pre can pass any floor in the grid.

Keep rule for a pair (e, r):
  p_pre >= floor  OR  (r ranks e in its top ``keep_rev_top`` S1s and p_pre >= floor_low)
then at most ``max_per_s1`` pairs per S1 by p_pre. The floor is the largest grid
value whose ceiling stays within ``ceiling_tol`` of the unfiltered union.
"""
from __future__ import annotations

import shutil

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from ..eval.metric import ceiling, macro_f05
from ..models.gbdt import fit_folds, load_folds, predict_by_fold, to_matrix
from ..utils import chunks, ensure_dir, inference_only, load_json, log, n_workers, save_json, work_dir
from .candidates import scan_union, union_files
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


def cheap_context(cfg, split: str) -> dict:
    """Lean tables the cheap features join against (loaded once per split)."""
    return {"norm": pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=_NORM),
            "src": pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src"]),
            "stats": s1_stats(cfg, split)}


def candidate_counts(pairs: pl.LazyFrame | pl.DataFrame) -> dict:
    """Candidate-list sizes per S1 and per record (need the whole country table, but only two columns)."""
    lf = pairs.lazy().select(["s1_uid", "rec_uid"])
    return {"s1": lf.group_by("s1_uid").agg(pl.len().cast(pl.Int32).alias("s1_ncand")).collect(),
            "rec": lf.group_by("rec_uid").agg(pl.len().cast(pl.Int32).alias("rec_ncand")).collect()}


def cheap_features(pairs: pl.DataFrame, cfg, split: str, ctx: dict | None = None,
                   counts: dict | None = None) -> pl.DataFrame:
    """Add PRE_FEATURES columns to ``pairs`` (s1_uid, rec_uid, channel columns[, hop2]).

    ``counts`` (from ``candidate_counts`` over the full country table) makes chunked scoring exact;
    without it the counts are taken over ``pairs`` itself.
    """
    ctx = ctx or cheap_context(cfg, split)
    counts = counts or candidate_counts(pairs)
    pairs = (pairs.drop([c for c in ("s1_ncand", "rec_ncand") if c in pairs.columns])
             .join(counts["s1"], on="s1_uid", how="left").join(counts["rec"], on="rec_uid", how="left"))
    workers = n_workers(cfg)
    parts = []
    for s, e in chunks(pairs.height, int(cfg.prerank.get("chunk_rows", 5_000_000))):
        df = attach_norm(pairs.slice(s, e - s), ctx["norm"], [c for c in _NORM if c != "uid"])
        df = df.join(ctx["src"].rename({"uid": "rec_uid"}), on="rec_uid", how="left").join(
            ctx["stats"], on="s1_uid", how="left")
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
    """Train on sampled entities, then stream-score every union file (train OOF, test fold-mean)."""
    pc = cfg.prerank
    mdir = ensure_dir(work_dir(cfg, None, "models", "prerank"))
    if not inference_only(cfg):
        _train_prerank(cfg, mdir)
    models = load_folds(mdir)
    if not inference_only(cfg):
        _score_split(cfg, "train", models)
        _tune_floor(cfg, mdir)
    _score_split(cfg, "test", models)
    floor = float(load_json(mdir / "floor.json")["floor"])
    for split in (("test",) if inference_only(cfg) else ("train", "test")):
        _write_selection(cfg, split, floor)


def _keep_min(pc) -> float:
    """Pairs below this can never be selected by any floor in the grid (see ``_select``)."""
    return min(float(pc.floor_low), min(float(x) for x in pc.floor_grid))


def _train_prerank(cfg, mdir) -> None:
    pc = cfg.prerank
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "fold"])
    sample = s1["s1_uid"] if s1.height <= int(pc.sample_entities) else \
        s1["s1_uid"].sample(int(pc.sample_entities), seed=int(cfg.run.seed)).sort()
    keep = sample.implode()
    ctx = cheap_context(cfg, "train")
    rows = []
    for f in union_files(cfg, "train"):
        lf = pl.scan_parquet(f)
        sub = lf.filter(pl.col("s1_uid").is_in(keep)).collect()
        if sub.height:
            rows.append(cheap_features(sub, cfg, "train", ctx=ctx, counts=candidate_counts(lf)))
    tr = _labels(pl.concat(rows, how="diagonal_relaxed"), cfg)
    del rows
    log().info("  pre-ranker training sample: %d pairs of %d entities", tr.height, tr["s1_uid"].n_unique())
    X, y, fold, ents = to_matrix(tr, PRE_FEATURES), tr["label"].to_numpy(), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy()
    del tr  # the polars table is several GB; folds may train concurrently, each with its own copy of X
    fit_folds(X, y, fold, ents, pc.gbdt, PRE_FEATURES, mdir, seed=int(cfg.run.seed), threads=n_workers(cfg))


def _score_split(cfg, split: str, models) -> None:
    """Cheap features + p_pre for every union pair, chunk by chunk; keep p_pre >= keep_min."""
    pc = cfg.prerank
    ctx = cheap_context(cfg, split)
    folds = pl.read_parquet(work_dir(cfg, split, "s1.parquet"), columns=["s1_uid", "fold"])
    pdir = work_dir(cfg, split, "cands", "pre_parts")
    if pdir.exists():
        shutil.rmtree(pdir)
    ensure_dir(pdir)
    kmin, n_all, n_kept, part = _keep_min(pc), 0, 0, 0
    for f in union_files(cfg, split):
        lf = pl.scan_parquet(f)
        counts = candidate_counts(lf)
        n = lf.select(pl.len()).collect().item()
        for s, e in chunks(n, int(pc.get("chunk_rows", 5_000_000))):
            feats = cheap_features(lf.slice(s, e - s).collect(), cfg, split, ctx=ctx, counts=counts)
            feats = feats.join(folds, on="s1_uid", how="left", maintain_order="left")
            p = predict_by_fold(models, to_matrix(feats, PRE_FEATURES), feats["fold"].to_numpy())
            kept = feats.with_columns(pl.Series("p_pre", p)).filter(pl.col("p_pre") >= kmin)
            kept.write_parquet(pdir / f"part_{part:05d}.parquet")
            n_all, n_kept, part = n_all + feats.height, n_kept + kept.height, part + 1
            log().info("   %s %s rows %d-%d scored (%d kept so far)", split, f.stem, s, e, n_kept)
    out = work_dir(cfg, split, "cands", "pre_all.parquet")
    pl.scan_parquet(str(pdir) + "/*.parquet").sink_parquet(out)
    shutil.rmtree(pdir)
    log().info("  %s: %d union pairs scored, %d with p_pre >= %.4g kept", split, n_all, n_kept, kmin)


def _tune_floor(cfg, mdir) -> None:
    import gc

    pc = cfg.prerank
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
    
    all_hits = []
    for uf in union_files(cfg, "train"):
        chunk = pl.scan_parquet(str(uf)).select(["s1_uid", "rec_uid"]).collect()
        hits = chunk.join(truth, on=["s1_uid", "rec_uid"])
        if hits.height > 0:
            all_hits.append(hits)
        del chunk, hits
        gc.collect()
    hit_df = pl.concat(all_hits) if all_hits else pl.DataFrame({"s1_uid": [], "rec_uid": []})
    del all_hits
    base = macro_f05(hit_df, truth, s1)
    del hit_df
    gc.collect()
    lean = pl.read_parquet(work_dir(cfg, "train", "cands", "pre_all.parquet"), columns=["s1_uid", "rec_uid", "p_pre"])
    chosen, grid = float(pc.floor_grid[0]), []
    for fl in sorted(float(x) for x in pc.floor_grid):
        sel = _select(lean, fl, pc)
        c = ceiling(sel, truth, s1)
        grid.append({"floor": fl, "ceiling": c, "pairs_per_s1": sel.height / s1.height})
        log().info("   floor %.4f: ceiling %.5f, %.2f pairs/S1", fl, c, sel.height / s1.height)
        if c >= base - float(pc.ceiling_tol):
            chosen = fl
    save_json({"union_ceiling": base, "floor": chosen, "grid": grid}, mdir / "floor.json")
    log().info("  union ceiling %.5f -> chosen floor %.4f", base, chosen)


def _write_selection(cfg, split: str, floor: float) -> None:
    """Apply the keep rule on a lean (s1, rec, p_pre) frame, then stream the kept full rows to pre.parquet."""
    cdir = work_dir(cfg, split, "cands")
    lean = pl.read_parquet(cdir / "pre_all.parquet", columns=["s1_uid", "rec_uid", "p_pre"])
    sel = _select(lean, floor, cfg.prerank).select(["s1_uid", "rec_uid"])
    (pl.scan_parquet(cdir / "pre_all.parquet").join(sel.lazy(), on=["s1_uid", "rec_uid"], how="semi")
     .sort(["s1_uid", "rec_uid"]).sink_parquet(cdir / "pre.parquet"))
    log().info("  %s: %d candidates after floor %.4f (%.2f per S1)", split, sel.height, floor,
               sel.height / max(1, pl.scan_parquet(work_dir(cfg, split, "s1.parquet")).select(pl.len()).collect().item()))
