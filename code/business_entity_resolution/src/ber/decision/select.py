"""Vectorised set selection for macro F0.5 (see docs §3 and §9).

Per S1 (candidates = records it owns, sorted by calibrated q desc):
  first member   iff q1 * kappa > P(singleton) = 1 - g   and   q1 >= t_min
  k-th addition  iff q >= t_add[k] + delta   (stop at the first failure)
  per-source caps (S2 <= 5, S3 <= 6) skip, not stop.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..eval.metric import f05_vec


def build_arrays(pairs: pl.DataFrame, ents: pl.DataFrame, J: int) -> dict:
    """pairs: s1_uid, rec_uid, q_own, src[, label]; ents: s1_uid, g, prof[, n_true] (defines row order)."""
    ents = ents.with_row_index("_row")
    p = (pairs.filter(pl.col("q_own") > 0).sort(["s1_uid", "rec_uid"])
         .with_columns(pl.col("q_own").rank("ordinal", descending=True).over("s1_uid").alias("_r"))
         .filter(pl.col("_r") <= J).join(ents.select(["s1_uid", "_row"]), on="s1_uid"))
    ne = ents.height
    rows, cols = p["_row"].to_numpy(), p["_r"].to_numpy().astype(np.int64) - 1
    Q = np.zeros((ne, J), np.float32)
    SRC = np.zeros((ne, J), np.int8)
    REC = np.full((ne, J), -1, np.int64)
    Q[rows, cols] = p["q_own"].to_numpy()
    SRC[rows, cols] = p["src"].to_numpy()
    REC[rows, cols] = p["rec_uid"].to_numpy()
    out = {"Q": Q, "SRC": SRC, "REC": REC, "G": ents["g"].to_numpy().astype(np.float32),
           "PROF": ents["prof"].to_numpy(), "S1": ents["s1_uid"].to_numpy()}
    if "label" in p.columns and "n_true" in ents.columns:
        Y = np.zeros((ne, J), bool)
        Y[rows, cols] = p["label"].to_numpy() == 1
        out["Y"], out["NTRUE"] = Y, ents["n_true"].to_numpy()
    return out


def select_mask(Q, SRC, G, kappa, t_min, delta, t_add, caps) -> np.ndarray:
    ne, J = Q.shape
    t_add = np.asarray(t_add, dtype=np.float32)
    sel = np.zeros((ne, J), bool)
    first = (Q[:, 0] > 0) & (Q[:, 0] * kappa > 1.0 - G) & (Q[:, 0] >= t_min)
    sel[:, 0] = first
    k = first.astype(np.int32)
    cnt = {s: (first & (SRC[:, 0] == s)).astype(np.int32) for s in caps}
    active = first.copy()
    for j in range(1, J):
        thr = t_add[np.minimum(k, len(t_add) - 1)] + delta
        above = active & (Q[:, j] > 0) & (Q[:, j] >= thr)
        capped = np.zeros(ne, bool)
        for s, c in caps.items():
            capped |= (SRC[:, j] == s) & (cnt[s] >= c)
        take = above & ~capped
        sel[:, j] = take
        k += take
        for s in caps:
            cnt[s] += take & (SRC[:, j] == s)
        active = above
    return sel


def entity_f05(sel: np.ndarray, Y: np.ndarray, NTRUE: np.ndarray) -> np.ndarray:
    return f05_vec((sel & Y).sum(1), sel.sum(1), NTRUE)


def selected_pairs(arr: dict, sel: np.ndarray) -> pl.DataFrame:
    r, c = np.nonzero(sel)
    return pl.DataFrame({"s1_uid": arr["S1"][r], "rec_uid": arr["REC"][r, c], "q": arr["Q"][r, c]})
