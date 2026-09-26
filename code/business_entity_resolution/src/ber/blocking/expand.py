"""Stage 1c: 2-hop expansion through record<->record neighbours, then candidate_pairs.

A record that shares little with its S1 (trade name, empty address) usually
shares a lot with that S1's other, confidently-matched copies. For each S1 we add
the record-kNN neighbours of its confident members (p_pre >= conf) that are not
yet candidates. This happens *before* the final candidate file is written, so
final matches remain a subset of candidates.

It is applied only if it raises the train ceiling (best achievable macro F0.5 of the candidate set)
by at least ``min_gain``, so it can add recoverable matches but never costs accuracy when it does
not help. The decision is made on a sample of ``probe_entities`` train S1s (cost ~ the sample, not
the whole set); only if it passes is the expansion run on all of train and on test.
"""
from __future__ import annotations

import shutil

import numpy as np
import polars as pl

from ..eval.metric import ceiling
from ..models.gbdt import load_folds, predict_by_fold, to_matrix
from ..utils import inference_only, load_json, log, save_json, work_dir
from .candidates import ensure_vectors
from .knn import topk
from .prerank import PRE_FEATURES, candidate_counts, cheap_features

CHANNEL_COLS = ["s1_uid", "rec_uid", "a1f_rank", "a1r_rank", "n1f_rank", "n1r_rank", "d1f_rank", "d1r_rank",
                "a1_sim", "n1_sim", "d1_sim", "key_mask", "key_block", "key_n", "a1_cos", "n1_cos", "country"]


def _two_hop(pre: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    ec = cfg.expand
    src_arr = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["src"])["src"].to_numpy()
    out = []
    for country in pre["country"].unique().to_list():
        vdir = ensure_vectors(cfg, split, country)  # rebuilt (deterministically) if a restore dropped them
        uids = np.load(vdir / "uids.npy")
        C = np.load(vdir / "comb_rp.npy", mmap_mode="r")  # only the rows indexed below are read into RAM
        sub = pre.filter(pl.col("country") == country)
        conf = sub.filter(pl.col("p_pre") >= float(ec.conf)).select(["s1_uid", pl.col("rec_uid").alias("m")])
        members = conf["m"].unique().sort().to_numpy()
        if len(members) == 0:
            continue
        rec_pos = np.where(src_arr[uids] != 1)[0]  # records.parquet row == uid
        q_pos = np.searchsorted(uids, members)
        idx, sim = topk(C[rec_pos], C[q_pos], int(ec.k) + 1, cfg.blocking.knn.device,
                         float(cfg.blocking.knn.mem_gb), max_gpus=int(cfg.blocking.knn.get("max_gpus", 0)))
        k = idx.shape[1]
        nb = pl.DataFrame({"m": np.repeat(members, k), "n": uids[rec_pos[np.clip(idx.ravel(), 0, None)]],
                           "sim": sim.ravel(), "ok": idx.ravel() >= 0})
        nb = nb.filter(pl.col("ok") & (pl.col("m") != pl.col("n")) & (pl.col("sim") >= float(ec.min_cos))).drop("ok")
        add = (conf.join(nb, on="m").group_by(["s1_uid", "n"]).agg(pl.col("sim").max().alias("hop_sim"))
               .rename({"n": "rec_uid"}).join(sub.select(["s1_uid", "rec_uid"]), on=["s1_uid", "rec_uid"], how="anti"))
        add = (add.with_columns(pl.col("hop_sim").rank("ordinal", descending=True).over("s1_uid").alias("_r"))
               .filter(pl.col("_r") <= int(ec.max_add)).drop("_r").with_columns(pl.lit(country).alias("country")))
        out.append(add)
    return pl.concat(out) if out else pl.DataFrame(schema={"s1_uid": pl.Int64, "rec_uid": pl.Int64,
                                                          "hop_sim": pl.Float32, "country": pl.Utf8})


def _score(pre: pl.DataFrame, new: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    """p_pre for hop-2 pairs (context counted over old + new); existing rows keep their p_pre.

    Features are computed for the new pairs only -- the existing candidates are not recomputed.
    """
    keys = ["s1_uid", "rec_uid"]
    counts = candidate_counts(pl.concat([pre.select(keys), new.select(keys)]))
    channels = [c for c in CHANNEL_COLS if c in pre.columns and c not in keys + ["country"]]
    rows = pl.concat([pre.head(0).select(channels),  # typed empty frame: new rows get null channel columns
                      new.select(keys + ["country"]).with_columns(pl.lit(1, dtype=pl.Int8).alias("hop2"))],
                     how="diagonal_relaxed")
    feats = cheap_features(rows, cfg, split, counts=counts)
    if split == "train":
        s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "fold"])
        feats = feats.join(s1, on="s1_uid", how="left")
    else:
        feats = feats.with_columns(pl.lit(-1, dtype=pl.Int8).alias("fold"))
    models = load_folds(work_dir(cfg, None, "models", "prerank"))
    feats = feats.with_columns(pl.Series("p_pre", predict_by_fold(models, to_matrix(feats, PRE_FEATURES),
                                                                  feats["fold"].to_numpy())))
    old = pre.with_columns(pl.lit(0, dtype=pl.Int8).alias("hop2"))
    return pl.concat([old, feats.select([c for c in old.columns if c in feats.columns])], how="diagonal_relaxed")


def _gain(pre: pl.DataFrame, new: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> tuple[float, float]:
    keys = ["s1_uid", "rec_uid"]
    return ceiling(pre, truth, s1), ceiling(pl.concat([pre.select(keys), new.select(keys)]), truth, s1)


def _decide(cfg, pre: pl.DataFrame, mdir) -> tuple[bool, pl.DataFrame | None]:
    """Train: is the expansion worth it? Returns (use, new pairs if they were computed for all of train)."""
    ec = cfg.expand
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
    probe, new = int(ec.get("probe_entities", 0)), None
    if 0 < probe < s1.height:
        sample = s1["s1_uid"].sample(probe, seed=int(cfg.run.seed) + 5).sort().implode()
        pre_p, s1_p = pre.filter(pl.col("s1_uid").is_in(sample)), s1.filter(pl.col("s1_uid").is_in(sample))
        new_p = _two_hop(pre_p, cfg, "train")
        before, after = _gain(pre_p, new_p, truth, s1_p)
        added, basis = new_p.height, f"probe of {probe} entities"
    else:
        new = _two_hop(pre, cfg, "train")
        before, after = _gain(pre, new, truth, s1)
        added, basis = new.height, "all train entities"
    use = (after - before) >= float(ec.min_gain)
    save_json({"ceiling_before": before, "ceiling_after": after, "gain": after - before, "min_gain": float(ec.min_gain),
              "added": added, "basis": basis, "use": use}, mdir / "expand.json")
    log().info("  2-hop (%s): +%d pairs, ceiling %.5f -> %.5f (gain %+.5f, needs >= %.5f) -> use=%s",
               basis, added, before, after, after - before, float(ec.min_gain), use)
    return use, new


def run_expand(cfg) -> None:
    ec = cfg.expand
    mdir = work_dir(cfg, None, "models", "prerank")
    for split in (("test",) if inference_only(cfg) else ("train", "test")):
        cdir = work_dir(cfg, split, "cands")
        if not ec.enabled:  # pre.parquet already carries hop2 = 0 and is sorted
            shutil.copyfile(cdir / "pre.parquet", cdir / "final.parquet")
            log().info("  %s: 2-hop expansion disabled; final candidates = pre-ranker selection", split)
            continue
        pre = pl.read_parquet(cdir / "pre.parquet")
        if split == "train":
            use, new = _decide(cfg, pre, mdir)
        else:  # test follows the train decision and skips the kNN entirely when it was "no"
            use, new = bool(load_json(mdir / "expand.json")["use"]), None
        if use and new is None:
            new = _two_hop(pre, cfg, split)
        if use and new is not None and new.height:
            final = _score(pre, new, cfg, split)
            log().info("  %s: +%d 2-hop candidates", split, new.height)
        else:
            final = pre.with_columns(pl.lit(0, dtype=pl.Int8).alias("hop2"))
        final.sort(["s1_uid", "rec_uid"]).write_parquet(cdir / "final.parquet")
        log().info("  %s final candidates: %d pairs", split, final.height)
