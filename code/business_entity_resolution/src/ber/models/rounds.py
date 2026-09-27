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
R3_EXTRA = ["p1_r1"]  # round 3: the round-2 extras rebuilt from round-2 scores, plus the round-1 score


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


def _fit_r2(cfg, mdir, seed: int, tag: str = "r2", stage: str = "r2") -> pl.DataFrame:
    """One round-2 (or round-3) fold set in ``mdir``, trained on the entity sample drawn with ``seed``; returns its
    train OOF scores (R2_OUT + ``pred``)."""
    sc = cfg[stage]
    extras = pl.read_parquet(work_dir(cfg, "train", "feats", f"{stage}_extras.parquet"))
    paths = r1_paths(cfg, "train")
    avail = columns_of(paths)
    feats = [f for f in R1_FEATURES + R2_EXTRA + (R3_EXTRA if stage == "r3" else [])
             if f in avail or f in extras.columns]
    sample = training_sample(paths, entities(paths), int(sc.sample_entities),
                             int(sc.get("max_train_rows", 8_000_000)), seed, tag)
    tr = load_rows(paths, KEYS + ["fold", "label"] + feats, sample)
    tr = tr.join(extras.filter(pl.col("s1_uid").is_in(sample.implode())), on=KEYS, how="inner").sort(KEYS)
    X, y, fold, ents = to_matrix(tr, feats), tr["label"].to_numpy().astype(np.float32), tr["fold"].to_numpy(), tr["s1_uid"].to_numpy()
    del tr  # free the polars table before (possibly concurrent) fold training
    models = fit_folds(X, y, fold, ents, sc.gbdt, feats, mdir, seed=seed,
                       threads=n_workers(cfg), monotone=sc.get("monotone", []))
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
    _run_bags(cfg, "r2")


def run_r3bag(cfg) -> None:
    """Round-3 bagging, as ``run_r2bag``: ``r3.bags - 1`` more round-3 fold sets on other entity samples, averaged
    with round 3's scores (kept as ``p2_r3b0``) and recalibrated."""
    _run_bags(cfg, "r3")


def _run_bags(cfg, stage: str) -> None:
    n = int(cfg[stage].get("bags", 1))
    if n <= 1 or (stage == "r3" and not (cfg.get("r3") or {}).get("enabled", False)):
        log().info("  %s.bags = %d: no extra %s models", stage, n, stage)
        return
    from ..checkpoint import sync_now

    base, off = ("p2_b", 100) if stage == "r2" else ("p2_r3b", 300)
    splits = ("test",) if inference_only(cfg) else ("train", "test")
    for k in range(1, n):
        mdir = ensure_dir(work_dir(cfg, None, "models", f"{stage}_bag{k}"))
        out = work_dir(cfg, "train", "preds", f"{stage}_bag{k}.parquet")
        if not inference_only(cfg) and not (out.exists() and (mdir / "trained.json").exists()):
            seed = int(cfg.run.seed) + off + 1000 * k
            pr = _fit_r2(cfg, mdir, seed, tag=f"{stage} bag {k}", stage=stage)
            pr.select(KEYS + [pl.col("pred").alias("p2")]).write_parquet(out)
            del pr
            gc.collect()
            save_json({"seed": seed}, mdir / "trained.json")
            sync_now(f"{stage} bag {k} trained")  # a new session resumes with the next bag
        te = work_dir(cfg, "test", "preds", f"{stage}_bag{k}.parquet")
        if not te.exists():
            extras = pl.read_parquet(work_dir(cfg, "test", "feats", f"{stage}_extras.parquet"))
            score_r2(cfg, "test", extras, load_folds(mdir)).select(KEYS + [pl.col("pred").alias("p2")]).write_parquet(te)
    cal_path = work_dir(cfg, None, "models", f"{stage}_bag_calibrator.pkl")
    cols = [f"{base}{k}" for k in range(n)]
    for split in splits:
        path = work_dir(cfg, split, "preds", "r2.parquet")
        r2 = pl.read_parquet(path)
        if cols[0] not in r2.columns:
            r2 = r2.with_columns(pl.col("p2").alias(cols[0]))
        for k in range(1, n):
            b = pl.read_parquet(work_dir(cfg, split, "preds", f"{stage}_bag{k}.parquet")).rename({"p2": cols[k]})
            r2 = r2.join(b, on=KEYS, how="left")
        r2 = (r2.with_columns(pl.mean_horizontal([pl.col(c).fill_nan(None) for c in cols]).alias("p2"))
              .drop(cols[1:]).sort(KEYS))
        p, prof = r2["p2"].to_numpy(), r2["prof"].to_numpy()
        if split == "train":
            y = r2["label"].to_numpy().astype(np.float32)
            Calibrator(fallback=cfg.decision.fallback_profile).fit(p, y, prof).save(cal_path)
            save_json({"bags": n, "single": _auc(y, r2[cols[0]].to_numpy(), f"{stage} (bag 0)"),
                       "bagged": _auc(y, p, f"{stage} ({n} bags)")},
                      work_dir(cfg, None, "models", f"{stage}_bag_metrics.json"))
        r2.with_columns(pl.Series("q", Calibrator.load(cal_path).transform(p, prof))).write_parquet(path)
        log().info("  %s: %s scores = mean of %d models", split, stage, n)


def r3_extras(cfg, split: str) -> pl.DataFrame:
    """Round-3 features: the round-2 extras (score context, consensus with confident members, competing clusters)
    rebuilt from the round-2 scores instead of the round-1 scores, plus the round-1 score (``p1_r1``)."""
    r2 = pl.read_parquet(work_dir(cfg, split, "preds", "r2.parquet"))
    col = "p2_r2" if "p2_r2" in r2.columns else "p2"   # after round 3, r2.parquet keeps round 2's score as p2_r2
    p = r2.select(KEYS + [pl.col(col).alias("p1")]).join(load_r1(cfg, split, columns=KEYS + ["num_hit"]), on=KEYS,
                                                           how="left")
    ce = load_ce(cfg, split) if cfg.ce.enabled else None
    ex = build_r2_extras(cfg, split, p, ce)
    p1 = pl.read_parquet(work_dir(cfg, split, "preds", "r1.parquet"), columns=KEYS + ["p1"]).rename({"p1": "p1_r1"})
    return ex.join(p1, on=KEYS, how="left").sort(KEYS)


def run_r3(cfg) -> None:
    """Round 3 (collective stacking): a pair model on the round-2 features recomputed from the round-2 scores, so
    each record's cluster context (confident members, competing S1s) comes from the stronger model. Its calibrated
    scores replace round 2's in ``preds/r2.parquet`` (round 2's kept as ``p2_r2`` / ``q_r2``), so the gate, the
    thresholds and the outputs run on them unchanged. Off unless ``r3.enabled``."""
    rc = cfg.get("r3") or {}
    if not rc.get("enabled", False):
        log().info("  r3.enabled = false: no round 3")
        return
    from ..checkpoint import sync_now

    mdir = ensure_dir(work_dir(cfg, None, "models", "r3"))
    splits = ("test",) if inference_only(cfg) else ("train", "test")
    for split in splits:
        r3_extras(cfg, split).write_parquet(ensure_dir(work_dir(cfg, split, "feats")) / "r3_extras.parquet")
    if not inference_only(cfg) and not (mdir / "calibrator.pkl").exists():
        pr = _fit_r2(cfg, mdir, int(cfg.run.seed) + 300, tag="r3", stage="r3")
        y, oof, prof = pr["label"].to_numpy().astype(np.float32), pr["pred"].to_numpy(), pr["prof"].to_numpy()
        Calibrator(fallback=cfg.decision.fallback_profile).fit(oof, y, prof).save(mdir / "calibrator.pkl")
        pr.select(KEYS + [pl.col("pred").alias("p3")]).write_parquet(work_dir(cfg, "train", "preds", "r3_train.parquet"))
        sync_now("round 3 trained")
    extras = pl.read_parquet(work_dir(cfg, "test", "feats", "r3_extras.parquet"))
    score_r2(cfg, "test", extras, load_folds(mdir)).select(KEYS + [pl.col("pred").alias("p3")]).write_parquet(
        work_dir(cfg, "test", "preds", "r3_test.parquet"))
    cal = Calibrator.load(mdir / "calibrator.pkl")
    for split in splits:
        path = work_dir(cfg, split, "preds", "r2.parquet")
        r2 = pl.read_parquet(path)
        if "p2_r2" not in r2.columns:
            r2 = r2.with_columns(pl.col("p2").alias("p2_r2"), pl.col("q").alias("q_r2"))
        p3 = pl.read_parquet(work_dir(cfg, split, "preds", f"r3_{split}.parquet"))
        r2 = r2.drop(["p2", "q"]).join(p3, on=KEYS, how="left").rename({"p3": "p2"}).sort(KEYS)
        p = r2["p2"].to_numpy()
        r2 = r2.with_columns(pl.Series("q", cal.transform(p, r2["prof"].to_numpy())))
        if split == "train":
            y = r2["label"].to_numpy().astype(np.float32)
            save_json({"r2": _auc(y, r2["p2_r2"].to_numpy(), "r2"), "r3": _auc(y, p, "r3")},
                      work_dir(cfg, None, "models", "r3_metrics.json"))
        r2.write_parquet(path)
        log().info("  %s: pair scores = round 3", split)
