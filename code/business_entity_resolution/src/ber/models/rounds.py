"""Round-1 and round-2 pair models (OOF on train, fold-average on test)."""
from __future__ import annotations

import gc

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from ..features.consensus import R2_EXTRA, build_r2_extras
from ..features.pair import R1_FEATURES, load_r1
from ..utils import ensure_dir, inference_only, log, n_workers, save_json, work_dir
from .calibrate import Calibrator
from .cross_encoder import load_ce
from .gbdt import fit_folds, load_folds, to_matrix
from .stream import columns_of, entities, load_rows, predict_shards, r1_paths, training_sample

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
        sample = training_sample(paths, entities(paths), int(cfg.r1.sample_entities),
                                 int(cfg.r1.get("max_train_rows", 8_000_000)), int(cfg.run.seed), "r1")
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


def _fit_r2(cfg, mdir, seed: int, tag: str = "r2") -> pl.DataFrame:
    """One round-2 fold set in ``mdir``, trained on the entity sample drawn with ``seed``; returns its train OOF
    scores (R2_OUT + ``pred``)."""
    extras = pl.read_parquet(work_dir(cfg, "train", "feats", "r2_extras.parquet"))
    paths = r1_paths(cfg, "train")
    avail = columns_of(paths)
    feats = [f for f in R1_FEATURES + R2_EXTRA if f in avail or f in extras.columns]
    sample = training_sample(paths, entities(paths), int(cfg.r2.sample_entities),
                             int(cfg.r2.get("max_train_rows", 8_000_000)), seed, tag)
    tr = load_rows(paths, KEYS + ["fold", "label"] + feats, sample)
    tr = tr.join(extras.filter(pl.col("s1_uid").is_in(sample.implode())), on=KEYS, how="inner").sort(KEYS)
    X, y, fold, ents = to_matrix(tr, feats), tr["label"].to_numpy().astype(np.float32), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy()
    del tr  # free the polars table before (possibly concurrent) fold training
    models = fit_folds(X, y, fold, ents, cfg.r2.gbdt, feats, mdir, seed=seed,
                       threads=n_workers(cfg), monotone=cfg.r2.get("monotone", []))
    del X
    pr = score_r2(cfg, "train", extras, models)
    save_json(_auc(pr["label"].to_numpy().astype(np.float32), pr["pred"].to_numpy(), tag), mdir / "oof_metrics.json")
    return pr


def _train_r2(cfg, mdir) -> None:
    pr = _fit_r2(cfg, mdir, int(cfg.run.seed) + 100)
    y, oof, prof = pr["label"].to_numpy().astype(np.float32), pr["pred"].to_numpy(), pr["prof"].to_numpy()
    cal = Calibrator(fallback=cfg.decision.fallback_profile).fit(oof, y, prof)
    cal.save(mdir / "calibrator.pkl")
    pr.select(R2_OUT).with_columns([pl.Series("p2", oof), pl.Series("q", cal.transform(oof, prof))]).write_parquet(
        work_dir(cfg, "train", "preds", "r2.parquet"))


def run_r2bag(cfg) -> None:
    """Round-2 bagging: ``r2.bags - 1`` more round-2 fold sets, each trained on another entity sample (seed), are
    averaged with the round-2 model's scores, and one calibrator is refitted on the averaged train OOF. Together they
    learn from far more of the train entities than one model can hold in RAM (``r2.sample_entities`` each).

    ``preds/r2.parquet`` is rewritten with the averaged ``p2`` / ``q``; the round-2 model's own score is kept as
    ``p2_b0``, so a re-run averages the same inputs again."""
    n = int(cfg.r2.get("bags", 1))
    if n <= 1:
        log().info("  r2.bags = %d: no extra round-2 models", n)
        return
    from ..checkpoint import sync_now

    splits = ("test",) if inference_only(cfg) else ("train", "test")
    for k in range(1, n):
        mdir = ensure_dir(work_dir(cfg, None, "models", f"r2_bag{k}"))
        out = work_dir(cfg, "train", "preds", f"r2_bag{k}.parquet")
        if not inference_only(cfg) and not (out.exists() and (mdir / "trained.json").exists()):
            seed = int(cfg.run.seed) + 100 + 1000 * k
            pr = _fit_r2(cfg, mdir, seed, tag=f"r2 bag {k}")
            pr.select(KEYS + [pl.col("pred").alias("p2")]).write_parquet(out)
            del pr
            gc.collect()
            save_json({"seed": seed}, mdir / "trained.json")
            sync_now(f"round-2 bag {k} trained")  # a new session resumes with the next bag
        te = work_dir(cfg, "test", "preds", f"r2_bag{k}.parquet")
        if not te.exists():
            extras = pl.read_parquet(work_dir(cfg, "test", "feats", "r2_extras.parquet"))
            score_r2(cfg, "test", extras, load_folds(mdir)).select(KEYS + [pl.col("pred").alias("p2")]).write_parquet(te)
    cal_path = work_dir(cfg, None, "models", "r2_bag_calibrator.pkl")
    cols = [f"p2_b{k}" for k in range(n)]
    for split in splits:
        path = work_dir(cfg, split, "preds", "r2.parquet")
        r2 = pl.read_parquet(path)
        if "p2_b0" not in r2.columns:
            r2 = r2.with_columns(pl.col("p2").alias("p2_b0"))
        for k in range(1, n):
            b = pl.read_parquet(work_dir(cfg, split, "preds", f"r2_bag{k}.parquet")).rename({"p2": f"p2_b{k}"})
            r2 = r2.join(b, on=KEYS, how="left")
        r2 = (r2.with_columns(pl.mean_horizontal([pl.col(c).fill_nan(None) for c in cols]).alias("p2"))
              .drop(cols[1:]).sort(KEYS))
        p, prof = r2["p2"].to_numpy(), r2["prof"].to_numpy()
        if split == "train":
            y = r2["label"].to_numpy().astype(np.float32)
            Calibrator(fallback=cfg.decision.fallback_profile).fit(p, y, prof).save(cal_path)
            save_json({"bags": n, "single": _auc(y, r2["p2_b0"].to_numpy(), "r2 (bag 0)"),
                       "bagged": _auc(y, p, f"r2 ({n} bags)")},
                      work_dir(cfg, None, "models", "r2_bag_metrics.json"))
        r2.with_columns(pl.Series("q", Calibrator.load(cal_path).transform(p, prof))).write_parquet(path)
        log().info("  %s: round-2 scores = mean of %d models", split, n)
