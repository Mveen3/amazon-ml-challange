"""Exact competition metric (macro F0.5 over S1 entities) and candidate ceiling."""
from __future__ import annotations

import numpy as np
import polars as pl


def f05(pred: set, true: set) -> float:
    """Reference single-entity implementation (used in tests)."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def f05_vec(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    out = np.zeros_like(tp)
    single = n_true == 0
    out[single] = (n_pred[single] == 0).astype(np.float64)
    ok = (~single) & (tp > 0)
    p = tp[ok] / n_pred[ok]
    r = tp[ok] / n_true[ok]
    out[ok] = 1.25 * p * r / (0.25 * p + r)
    return out


def entity_scores(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 F0.5. ``pred``/``truth``: (s1_uid, rec_uid); ``s1``: s1_uid, n_true (+ any extras)."""
    pred = pred.select(["s1_uid", "rec_uid"]).unique()
    tp = pred.join(truth.select(["s1_uid", "rec_uid"]), on=["s1_uid", "rec_uid"]).group_by("s1_uid").agg(pl.len().alias("tp"))
    npred = pred.group_by("s1_uid").agg(pl.len().alias("n_pred"))
    df = (s1.join(tp, on="s1_uid", how="left").join(npred, on="s1_uid", how="left")
          .with_columns([pl.col("tp").fill_null(0), pl.col("n_pred").fill_null(0)]))
    f = f05_vec(df["tp"].to_numpy(), df["n_pred"].to_numpy(), df["n_true"].to_numpy())
    return df.with_columns(pl.Series("f05", f))


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> float:
    return float(entity_scores(pred, truth, s1)["f05"].mean())


def ceiling(cands: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> float:
    """Best achievable macro F0.5 if the matcher picked exactly the true pairs among ``cands``."""
    hit = cands.select(["s1_uid", "rec_uid"]).unique().join(truth, on=["s1_uid", "rec_uid"])
    return macro_f05(hit, truth, s1)


def report(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame, cands: pl.DataFrame | None = None) -> dict:
    """Macro F0.5 overall and by profile / singleton / true-cluster-size, plus micro P/R."""
    es = entity_scores(pred, truth, s1)
    out = {"macro_f05": float(es["f05"].mean()), "n_s1": es.height}
    tp, npred, ntrue = es["tp"].sum(), es["n_pred"].sum(), es["n_true"].sum()
    out["micro_precision"] = float(tp / npred) if npred else 0.0
    out["micro_recall"] = float(tp / ntrue) if ntrue else 0.0
    out["by_profile"] = {r[0]: round(r[1], 5) for r in es.group_by("prof").agg(pl.col("f05").mean()).iter_rows()}
    single = es.with_columns((pl.col("n_true") == 0).alias("singleton"))
    out["singleton_f05"] = float(single.filter(pl.col("singleton"))["f05"].mean() or 0.0)
    out["non_singleton_f05"] = float(single.filter(~pl.col("singleton"))["f05"].mean() or 0.0)
    out["by_true_size"] = {int(r[0]): round(r[1], 5) for r in
                           es.group_by("n_true").agg(pl.col("f05").mean()).sort("n_true").iter_rows()}
    if cands is not None:
        out["ceiling"] = ceiling(cands, truth, s1)
        out["cands_per_s1"] = cands.height / max(1, s1.height)
    return out
