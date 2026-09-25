"""Unmatched-record density stress test (docs §10.4).

Test has ~5.75 S2+S3 records per S1 vs ~4.68 in train. If that comes from extra
*unmatched* records, deleting ~19% of train S1s (keeping their records, which
become orphans) reproduces it.
  check()    compares the per-record best pre-ranker score distribution of test
             vs train and vs train-with-drops (KS distance); accept if drops fit better.
  simulate() re-runs competition features -> round 2 -> calibration -> gate on
             the reduced train set (OOF models), for robust threshold tuning.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..features.consensus import R2_EXTRA
from ..features.context import context_names, score_context
from ..models.calibrate import Calibrator
from ..models.gate import ownership, rescore_gate
from ..models.rounds import r2_frame, rescore_r2
from ..decision.select import build_arrays
from ..utils import log, save_json, work_dir


def _dropped(cfg) -> pl.Series:
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid"])
    rng = np.random.default_rng(int(cfg.run.seed) + 999)
    return s1.filter(pl.Series(rng.random(s1.height) < float(cfg.stress.drop_frac)))["s1_uid"]


def _best_per_record(cfg, split: str, drop: pl.Series | None = None) -> np.ndarray:
    pre = pl.read_parquet(work_dir(cfg, split, "cands", "pre_all.parquet"), columns=["s1_uid", "rec_uid", "p_pre"])
    if drop is not None:
        pre = pre.filter(~pl.col("s1_uid").is_in(drop.implode()))
    n_rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["src"]).filter(pl.col("src") != 1).height
    best = pre.group_by("rec_uid").agg(pl.col("p_pre").max())["p_pre"].to_numpy()
    return np.concatenate([best, np.zeros(max(0, n_rec - len(best)), np.float32)])


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.linspace(0, 1, 201)
    ca = np.searchsorted(np.sort(a), grid, side="right") / max(1, len(a))
    cb = np.searchsorted(np.sort(b), grid, side="right") / max(1, len(b))
    return float(np.abs(ca - cb).max())


def check(cfg) -> dict:
    test = _best_per_record(cfg, "test")
    train = _best_per_record(cfg, "train")
    dropped = _best_per_record(cfg, "train", _dropped(cfg))
    res = {"ks_test_train": _ks(test, train), "ks_test_dropped": _ks(test, dropped),
           "low_tail_test": float((test < 0.1).mean()), "low_tail_train": float((train < 0.1).mean()),
           "low_tail_dropped": float((dropped < 0.1).mean())}
    res["accepted"] = res["ks_test_dropped"] < res["ks_test_train"]
    save_json(res, work_dir(cfg, None, "models", "stress_check.json"))
    log().info("  stress check: %s", res)
    return res


def simulate(cfg) -> dict:
    drop = _dropped(cfg)
    extras = pl.read_parquet(work_dir(cfg, "train", "feats", "r2_extras.parquet"))
    extras = extras.filter(~pl.col("s1_uid").is_in(drop.implode()))
    c1 = context_names("c1")
    extras = score_context(extras.drop(c1), "p1", "c1")  # competition recomputed without the dropped S1s
    df, feats = r2_frame(cfg, "train", extras.select(["s1_uid", "rec_uid"] + R2_EXTRA))
    p2 = rescore_r2(cfg, df, feats)
    cal = Calibrator.load(work_dir(cfg, None, "models", "r2", "calibrator.pkl"))
    prof = df["prof"].to_numpy()
    r2 = df.select(["s1_uid", "rec_uid", "fold", "prof", "label", "src", "nm_tset", "ad_tset", "num_hit"]).with_columns(
        [pl.Series("p2", p2), pl.Series("q", cal.transform(p2, prof))])
    gate = rescore_gate(cfg, "train", r2).filter(~pl.col("s1_uid").is_in(drop.implode()))
    log().info("  stress simulate: %d S1 kept, %d pairs", gate.height, r2.height)
    return build_arrays(ownership(r2.sort(["s1_uid", "rec_uid"])), gate.sort("s1_uid"), int(cfg.decision.max_members))
