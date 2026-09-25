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
from .gbdt import fit_folds, load_folds, to_matrix
from .stream import columns_of, entities, load_rows, predict_shards, r1_paths, sample_entities

KEYS = ["s1_uid", "rec_uid"]


def _auc(y, p, tag: str) -> dict:
    ok = ~np.isnan(p)
    res = {"auc": float(roc_auc_score(y[ok], p[ok])), "ap": float(average_precision_score(y[ok], p[ok]))}
    log().info("  %s OOF: AUC %.5f  AP %.5f", tag, res["auc"], res["ap"])
    return res


def run_r1(cfg) -> None:
    mdir = ensure_dir(work_dir(cfg, None, "models", "r1"))
    if not inference_only(cfg):
        paths = r1_paths(cfg, "train")
        feats = [f for f in R1_FEATURES if f in columns_of(paths)]
        sample = sample_entities(entities(paths), int(cfg.r1.sample_entities), int(cfg.run.seed))
        tr = load_rows(paths, KEYS + ["fold", "label"] + feats, sample)
        X, y, fold, ents = to_matrix(tr, feats), tr["label"].to_numpy().astype(np.float32), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy()
        del tr  # free the polars table before (possibly concurrent) fold training
        models = fit_folds(X, y, fold, ents, cfg.r1.gbdt, feats, mdir, seed=int(cfg.run.seed),
                           threads=n_workers(cfg), monotone=cfg.r1.get("monotone", []))
        del X
        pr = predict_shards(paths, models, feats, KEYS + ["fold", "label"])
        save_json(_auc(pr["label"].to_numpy(), pr["pred"].to_numpy(), "r1"), mdir / "oof_metrics.json")
        pr.select(KEYS + ["fold", pl.col("pred").alias("p1")]).write_parquet(
            ensure_dir(work_dir(cfg, "train", "preds")) / "r1.parquet")
        del pr
    models = load_folds(mdir)
    pt = predict_shards(r1_paths(cfg, "test"), models, models[0].features, KEYS + ["fold"])
    pt.select(KEYS + ["fold", pl.col("pred").alias("p1")]).write_parquet(
        ensure_dir(work_dir(cfg, "test", "preds")) / "r1.parquet")


def r2_extras(cfg, split: str) -> pl.DataFrame:
    p1 = pl.read_parquet(work_dir(cfg, split, "preds", "r1.parquet"), columns=KEYS + ["p1"])
    p1 = p1.join(load_r1(cfg, split, columns=KEYS + ["num_hit"]), on=KEYS, how="left")
    ce = load_ce(cfg, split) if cfg.ce.enabled else None
    return build_r2_extras(cfg, split, p1, ce)


R2_OUT = KEYS + ["fold", "prof", "label", "src", "nm_tset", "ad_tset", "num_hit"]


def _join_extras(extras: pl.DataFrame):
    """Shard transform: attach round-2 extras (shards cover contiguous S1 ranges, so slice by range first)."""
    def f(df: pl.DataFrame) -> pl.DataFrame:
        lo, hi = df["s1_uid"].min(), df["s1_uid"].max()
        return df.join(extras.filter(pl.col("s1_uid").is_between(lo, hi)), on=KEYS, how="inner")
    return f


def score_r2(cfg, split: str, extras: pl.DataFrame, models=None) -> pl.DataFrame:
    """Round-2 scores for every pair of ``split`` (train: OOF by fold) -> R2_OUT columns + ``pred``."""
    models = models or load_folds(work_dir(cfg, None, "models", "r2"))
    return predict_shards(r1_paths(cfg, split), models, models[0].features, R2_OUT, transform=_join_extras(extras))


def run_r2(cfg) -> None:
    mdir = ensure_dir(work_dir(cfg, None, "models", "r2"))
    for split in (("test",) if inference_only(cfg) else ("train", "test")):
        r2_extras(cfg, split).sort(KEYS).write_parquet(ensure_dir(work_dir(cfg, split, "feats")) / "r2_extras.parquet")
    if not inference_only(cfg):
        _train_r2(cfg, mdir)
    cal = Calibrator.load(mdir / "calibrator.pkl")
    extras = pl.read_parquet(work_dir(cfg, "test", "feats", "r2_extras.parquet"))
    te = score_r2(cfg, "test", extras)
    p = te["pred"].to_numpy()
    te.select([c for c in R2_OUT if c != "label"]).with_columns(
        [pl.Series("p2", p), pl.Series("q", cal.transform(p, te["prof"].to_numpy()))]).write_parquet(
        work_dir(cfg, "test", "preds", "r2.parquet"))


def _train_r2(cfg, mdir) -> None:
    extras = pl.read_parquet(work_dir(cfg, "train", "feats", "r2_extras.parquet"))
    paths = r1_paths(cfg, "train")
    avail = columns_of(paths)
    feats = [f for f in R1_FEATURES + R2_EXTRA if f in avail or f in extras.columns]
    sample = sample_entities(entities(paths), int(cfg.r2.sample_entities), int(cfg.run.seed) + 100)
    tr = load_rows(paths, KEYS + ["fold", "label"] + feats, sample)
    tr = tr.join(extras.filter(pl.col("s1_uid").is_in(sample.implode())), on=KEYS, how="inner").sort(KEYS)
    X, y, fold, ents = to_matrix(tr, feats), tr["label"].to_numpy().astype(np.float32), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy()
    del tr  # free the polars table before (possibly concurrent) fold training
    models = fit_folds(X, y, fold, ents, cfg.r2.gbdt, feats, mdir, seed=int(cfg.run.seed) + 100,
                       threads=n_workers(cfg), monotone=cfg.r2.get("monotone", []))
    del X
    pr = score_r2(cfg, "train", extras, models)
    y, oof, prof = pr["label"].to_numpy().astype(np.float32), pr["pred"].to_numpy(), pr["prof"].to_numpy()
    save_json(_auc(y, oof, "r2"), mdir / "oof_metrics.json")
    cal = Calibrator(fallback=cfg.decision.fallback_profile).fit(oof, y, prof)
    cal.save(mdir / "calibrator.pkl")
    pr.select(R2_OUT).with_columns([pl.Series("p2", oof), pl.Series("q", cal.transform(oof, prof))]).write_parquet(
        work_dir(cfg, "train", "preds", "r2.parquet"))
