"""Stage 1c: 2-hop expansion through record<->record neighbours, then candidate_pairs.

A record that shares little with its S1 (trade name, empty address) usually
shares a lot with that S1's other, confidently-matched copies. For each S1 we add
the record-kNN neighbours of its confident members (p_pre >= conf) that are not
yet candidates. This happens *before* the final candidate file is written, so
final matches remain a subset of candidates. Enabled on test only if it raised
the train ceiling by at least ``min_gain``.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..eval.metric import ceiling
from ..models.gbdt import load_folds, predict_by_fold, to_matrix
from ..utils import inference_only, load_json, log, save_json, work_dir
from .candidates import safe
from .knn import topk
from .prerank import PRE_FEATURES, cheap_features

CHANNEL_COLS = ["s1_uid", "rec_uid", "a1f_rank", "a1r_rank", "n1f_rank", "n1r_rank", "d1f_rank", "d1r_rank",
                "a1_sim", "n1_sim", "d1_sim", "key_mask", "key_block", "key_n", "a1_cos", "n1_cos", "country"]


def _two_hop(pre: pl.DataFrame, cfg, split: str) -> pl.DataFrame:
    ec = cfg.expand
    src_arr = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["src"])["src"].to_numpy()
    out = []
    for country in pre["country"].unique().to_list():
        vdir = work_dir(cfg, split, "vec", safe(country))
        uids = np.load(vdir / "uids.npy")
        C = np.load(vdir / "comb_rp.npy")
        sub = pre.filter(pl.col("country") == country)
        conf = sub.filter(pl.col("p_pre") >= float(ec.conf)).select(["s1_uid", pl.col("rec_uid").alias("m")])
        members = conf["m"].unique().sort().to_numpy()
        if len(members) == 0:
            continue
        rec_pos = np.where(src_arr[uids] != 1)[0]  # records.parquet row == uid
        q_pos = np.searchsorted(uids, members)
        idx, sim = topk(C[rec_pos], C[q_pos], int(ec.k) + 1, cfg.blocking.knn.device, float(cfg.blocking.knn.mem_gb))
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
    """p_pre for hop-2 pairs with context counted on the combined set; old rows keep their p_pre."""
    comb = pl.concat([pre.select([c for c in CHANNEL_COLS if c in pre.columns]).with_columns(pl.lit(0, dtype=pl.Int8).alias("hop2")),
                      new.select(["s1_uid", "rec_uid", "country"]).with_columns(pl.lit(1, dtype=pl.Int8).alias("hop2"))],
                     how="diagonal_relaxed")
    feats = cheap_features(comb, cfg, split).filter(pl.col("hop2") == 1)
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


def run_expand(cfg) -> None:
    ec = cfg.expand
    mdir = work_dir(cfg, None, "models", "prerank")
    for split in (("test",) if inference_only(cfg) else ("train", "test")):
        cdir = work_dir(cfg, split, "cands")
        pre = pl.read_parquet(cdir / "pre.parquet")
        use = False
        new = None
        if ec.enabled:
            new = _two_hop(pre, cfg, split)
            if split == "train":
                truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
                s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
                before = ceiling(pre, truth, s1)
                after = ceiling(pl.concat([pre.select(["s1_uid", "rec_uid"]), new.select(["s1_uid", "rec_uid"])]),
                                truth, s1)
                use = (after - before) >= float(ec.min_gain)
                save_json({"ceiling_before": before, "ceiling_after": after, "added": new.height, "use": use},
                          mdir / "expand.json")
                log().info("  2-hop: +%d pairs, ceiling %.5f -> %.5f (use=%s)", new.height, before, after, use)
            else:
                use = load_json(mdir / "expand.json")["use"]
        final = _score(pre, new, cfg, split) if (use and new is not None and new.height) else \
            pre.with_columns(pl.lit(0, dtype=pl.Int8).alias("hop2"))
        final.sort(["s1_uid", "rec_uid"]).write_parquet(cdir / "final.parquet")
        log().info("  %s final candidates: %d pairs", split, final.height)
