"""Round-1 and round-2 pair models (OOF on train, fold-average on test)."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from ..features.consensus import R2_EXTRA, build_r2_extras
from ..features.pair import R1_FEATURES, load_r1
from ..utils import ensure_dir, inference_only, log, n_workers, save_json, work_dir
from .calibrate import Calibrator
from .cross_encoder import load_ce
from .gbdt import load_folds, predict_by_fold, predict_mean, to_matrix, train_oof

KEYS = ["s1_uid", "rec_uid"]


def _auc(y, p, tag: str) -> dict:
    ok = ~np.isnan(p)
    res = {"auc": float(roc_auc_score(y[ok], p[ok])), "ap": float(average_precision_score(y[ok], p[ok]))}
    log().info("  %s OOF: AUC %.5f  AP %.5f", tag, res["auc"], res["ap"])
    return res


def run_r1(cfg) -> None:
    mdir = ensure_dir(work_dir(cfg, None, "models", "r1"))
    if not inference_only(cfg):
        tr = load_r1(cfg, "train")
        feats = [f for f in R1_FEATURES if f in tr.columns]
        y = tr["label"].to_numpy().astype(np.float32)
        oof, _ = train_oof(to_matrix(tr, feats), y, tr["fold"].to_numpy(), tr["s1_uid"].to_numpy(), cfg.r1.gbdt,
                           feats, mdir, seed=int(cfg.run.seed), threads=n_workers(cfg),
                           monotone=cfg.r1.get("monotone", []))
        save_json(_auc(y, oof, "r1"), mdir / "oof_metrics.json")
        pdir = ensure_dir(work_dir(cfg, "train", "preds"))
        tr.select(KEYS + ["fold"]).with_columns(pl.Series("p1", oof)).write_parquet(pdir / "r1.parquet")
        del tr
    models = load_folds(mdir)
    te = load_r1(cfg, "test")
    p = predict_mean(models, to_matrix(te, models[0].features))
    pdir = ensure_dir(work_dir(cfg, "test", "preds"))
    te.select(KEYS + ["fold"]).with_columns(pl.Series("p1", p)).write_parquet(pdir / "r1.parquet")


def r2_extras(cfg, split: str) -> pl.DataFrame:
    p1 = pl.read_parquet(work_dir(cfg, split, "preds", "r1.parquet"), columns=KEYS + ["p1"])
    p1 = p1.join(load_r1(cfg, split, columns=KEYS + ["num_hit"]), on=KEYS, how="left")
    ce = load_ce(cfg, split) if cfg.ce.enabled else None
    return build_r2_extras(cfg, split, p1, ce)


def r2_frame(cfg, split: str, extras: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    df = load_r1(cfg, split).join(extras, on=KEYS, how="inner").sort(KEYS)
    feats = [f for f in R1_FEATURES + R2_EXTRA if f in df.columns]
    return df, feats


def run_r2(cfg) -> None:
    mdir = ensure_dir(work_dir(cfg, None, "models", "r2"))
    for split in (("test",) if inference_only(cfg) else ("train", "test")):
        r2_extras(cfg, split).write_parquet(ensure_dir(work_dir(cfg, split, "feats")) / "r2_extras.parquet")
    if not inference_only(cfg):
        _train_r2(cfg, mdir)
    models = load_folds(mdir)
    cal = Calibrator.load(mdir / "calibrator.pkl")
    extras = pl.read_parquet(work_dir(cfg, "test", "feats", "r2_extras.parquet"))
    te, _ = r2_frame(cfg, "test", extras)
    p = predict_mean(models, to_matrix(te, models[0].features))
    prof = te["prof"].to_numpy()
    te.select(KEYS + ["fold", "prof", "src", "nm_tset", "ad_tset", "num_hit"]).with_columns(
        [pl.Series("p2", p), pl.Series("q", cal.transform(p, prof))]).write_parquet(
        work_dir(cfg, "test", "preds", "r2.parquet"))


def _train_r2(cfg, mdir) -> None:
    extras = pl.read_parquet(work_dir(cfg, "train", "feats", "r2_extras.parquet"))
    tr, feats = r2_frame(cfg, "train", extras)
    y = tr["label"].to_numpy().astype(np.float32)
    oof, _ = train_oof(to_matrix(tr, feats), y, tr["fold"].to_numpy(), tr["s1_uid"].to_numpy(), cfg.r2.gbdt, feats,
                       mdir, seed=int(cfg.run.seed) + 100, threads=n_workers(cfg),
                       monotone=cfg.r2.get("monotone", []))
    save_json(_auc(y, oof, "r2"), mdir / "oof_metrics.json")
    prof = tr["prof"].to_numpy()
    cal = Calibrator(fallback=cfg.decision.fallback_profile).fit(oof, y, prof)
    cal.save(mdir / "calibrator.pkl")
    out = tr.select(KEYS + ["fold", "prof", "label", "src", "nm_tset", "ad_tset", "num_hit"]).with_columns(
        [pl.Series("p2", oof), pl.Series("q", cal.transform(oof, prof))])
    out.write_parquet(work_dir(cfg, "train", "preds", "r2.parquet"))


def rescore_r2(cfg, df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """OOF-consistent round-2 probabilities for an arbitrary (train) frame (used by the stress test)."""
    models = load_folds(work_dir(cfg, None, "models", "r2"))
    return predict_by_fold(models, to_matrix(df, feats), df["fold"].to_numpy())
