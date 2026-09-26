"""Entity gate: P(S1 has >= 1 true match), learned from its candidate-list profile.

Replaces the independence-based product of (1 - p_i), which badly underestimates
P(singleton) for look-alike-heavy candidate lists.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..blocking.prerank import s1_stats
from ..features.context import density_normalize, score_context
from ..utils import ensure_dir, inference_only, log, n_workers, save_json, work_dir
from .calibrate import Calibrator
from .gbdt import load_folds, predict_by_fold, predict_mean, to_matrix, train_oof

GATE_FEATURES = ["g_top1", "g_top2", "g_top3", "g_gap12", "g_sum", "g_n03", "g_n05", "g_n08", "g_n_owned",
                 "g_ncand", "g_p2max", "g_top1_comp", "g_best_name", "g_best_addr", "g_best_numhit",
                 "s1_name_freq", "s1_addr_share"]


def ownership(df: pl.DataFrame) -> pl.DataFrame:
    """Each record keeps only its best S1 (``q_own``); verified: records have <= 1 owner."""
    owner = pl.col("q").rank("ordinal", descending=True).over("rec_uid") == 1
    return df.with_columns([owner.alias("owner"), pl.when(owner).then(pl.col("q")).otherwise(0.0).alias("q_own")])


def entity_frame(cfg, split: str, r2: pl.DataFrame) -> pl.DataFrame:
    df = score_context(ownership(r2.sort(["s1_uid", "rec_uid"])), "q", "g")
    agg = (df.sort(["s1_uid", "q_own", "rec_uid"], descending=[False, True, False])
           .group_by("s1_uid", maintain_order=True).agg([
        pl.col("q_own").head(3).alias("_tops"), pl.col("q_own").sum().alias("g_sum"),
        (pl.col("q_own") >= 0.3).sum().alias("g_n03"), (pl.col("q_own") >= 0.5).sum().alias("g_n05"),
        (pl.col("q_own") >= 0.8).sum().alias("g_n08"), (pl.col("q_own") > 0).sum().alias("g_n_owned"),
        pl.len().alias("g_ncand"), pl.col("p2").max().alias("g_p2max"),
        pl.col("g_rec_comp").first().alias("g_top1_comp"), pl.col("nm_tset").max().alias("g_best_name"),
        pl.col("ad_tset").max().alias("g_best_addr"), pl.col("num_hit").max().alias("g_best_numhit"),
    ]))
    agg = agg.with_columns([pl.col("_tops").list.get(i, null_on_oob=True).fill_null(0.0).alias(f"g_top{i + 1}")
                            for i in range(3)]).drop("_tops")
    agg = agg.with_columns((pl.col("g_top1") - pl.col("g_top2")).alias("g_gap12"))
    s1 = pl.read_parquet(work_dir(cfg, split, "s1.parquet")).join(s1_stats(cfg, split), on="s1_uid", how="left")
    ent = s1.join(agg, on="s1_uid", how="left")
    fill0 = ["g_top1", "g_top2", "g_top3", "g_gap12", "g_sum", "g_n03", "g_n05", "g_n08", "g_n_owned", "g_ncand"]
    ent = ent.with_columns([pl.col(c).fill_null(0) for c in fill0])
    if bool(cfg.gate.get("density_norm", True)):  # list sizes relative to the country's typical list
        ent = density_normalize(ent, ["g_ncand", "g_n_owned"], by="prof")
    return ent.sort("s1_uid")


def run_gate(cfg) -> None:
    mdir = ensure_dir(work_dir(cfg, None, "models", "gate"))
    if not inference_only(cfg):
        _train_gate(cfg, mdir)
    cal = Calibrator.load(mdir / "calibrator.pkl")
    te = entity_frame(cfg, "test", pl.read_parquet(work_dir(cfg, "test", "preds", "r2.parquet")))
    p = predict_mean(load_folds(mdir), to_matrix(te, GATE_FEATURES))
    te.select(["s1_uid", "prof"]).with_columns(pl.Series("g", cal.transform(p, te["prof"].to_numpy()))).write_parquet(
        work_dir(cfg, "test", "preds", "gate.parquet"))


def _train_gate(cfg, mdir) -> None:
    tr = entity_frame(cfg, "train", pl.read_parquet(work_dir(cfg, "train", "preds", "r2.parquet")))
    y = (tr["n_true"].to_numpy() > 0).astype(np.float32)
    oof, _ = train_oof(to_matrix(tr, GATE_FEATURES), y, tr["fold"].to_numpy(), tr["s1_uid"].to_numpy(),
                       cfg.gate.gbdt, GATE_FEATURES, mdir, seed=int(cfg.run.seed) + 200, threads=n_workers(cfg))
    prof = tr["prof"].to_numpy()
    cal = Calibrator(fallback=cfg.decision.fallback_profile, min_rows=100).fit(oof, y, prof)
    cal.save(mdir / "calibrator.pkl")
    g = cal.transform(oof, prof)
    save_json({"brier": float(np.mean((g - y) ** 2)), "base_rate": float(y.mean())}, mdir / "oof_metrics.json")
    log().info("  gate OOF brier %.5f (has-match rate %.4f)", float(np.mean((g - y) ** 2)), float(y.mean()))
    tr.select(["s1_uid", "prof", "fold", "n_true"]).with_columns(pl.Series("g", g)).write_parquet(
        work_dir(cfg, "train", "preds", "gate.parquet"))


def rescore_gate(cfg, split: str, r2: pl.DataFrame) -> pl.DataFrame:
    """OOF-consistent gate for a modified (train) r2 frame (stress test)."""
    mdir = work_dir(cfg, None, "models", "gate")
    ent = entity_frame(cfg, split, r2)
    ent = ent.filter(pl.col("s1_uid").is_in(r2["s1_uid"].unique().implode()) | (pl.col("g_ncand") == 0))
    p = predict_by_fold(load_folds(mdir), to_matrix(ent, GATE_FEATURES), ent["fold"].to_numpy())
    cal = Calibrator.load(mdir / "calibrator.pkl")
    return ent.select(["s1_uid", "prof", "fold", "n_true"]).with_columns(
        pl.Series("g", cal.transform(p, ent["prof"].to_numpy())))
