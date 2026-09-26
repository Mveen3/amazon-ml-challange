"""Expected-F0.5 set selection (improvement.md A4).

For one S1 with owned candidates in score order (source caps applied: an over-cap record is skipped) and the
gate's P(entity has >= 1 true match) = g, predicting the top k scores

    E[F(0)] = 1 - g                                          (empty is right only for a singleton)
    E[F(k)] = g * E_r[ 1.25 TP_k / (k + 0.25 N) ] / P_r(N >= 1)   (k >= 1)

where, given that the entity is not a singleton, member j is a true match with probability r_j (independent),
TP_k = true members among the top k, N = all true members + U, and U ~ Poisson(u) counts true matches that are
not among the candidates at all. The distributions of TP_k and of the true members after position k are
Poisson-binomial and are built with prefix / suffix dynamic programs, vectorised over all entities (J <= 12).

Probabilities: r_j = s(a * logit(q_j) + b) / g when ``cond`` (conditioned on the gate), else s(a * logit(q_j) + b).
a (sharpness), b (bias) and u are tuned per country profile on train out-of-fold, like the heuristic rule.
"""
from __future__ import annotations

import numpy as np

_UMAX = 3  # Poisson(u) truncated here (u <= 0.3 in the grid -> the tail beyond 3 is negligible)


def _eligible_order(Q: np.ndarray, SRC: np.ndarray, caps: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per entity: positions of the records that can be taken in order under the per-source caps, moved to the
    front (stable). Returns (idx, n_elig): ``idx[:, :n]`` are the eligible positions in score order."""
    ne, J = Q.shape
    cnt = {s: np.zeros(ne, np.int32) for s in caps}
    elig = np.zeros((ne, J), bool)
    for j in range(J):
        ok = Q[:, j] > 0
        for s, c in caps.items():
            ok &= ~((SRC[:, j] == s) & (cnt[s] >= c))
        elig[:, j] = ok
        for s in caps:
            cnt[s] += ok & (SRC[:, j] == s)
    idx = np.argsort(~elig, axis=1, kind="stable")
    return idx, elig.sum(1)


def _conv_bernoulli(dist: np.ndarray, p: np.ndarray) -> np.ndarray:
    """dist (ne, L) of a count; add an independent Bernoulli(p) -> (ne, L + 1)."""
    out = np.zeros((dist.shape[0], dist.shape[1] + 1), dist.dtype)
    out[:, :-1] = dist * (1.0 - p)[:, None]
    out[:, 1:] += dist * p[:, None]
    return out


def expected_f05(P: np.ndarray, n_elig: np.ndarray, G: np.ndarray, u: float, cond: bool) -> np.ndarray:
    """E[F0.5] of predicting the top k eligible members, k = 0..J -> (ne, J + 1). ``P``: eligible-first probs."""
    ne, J = P.shape
    P = np.where(np.arange(J)[None, :] < n_elig[:, None], P, 0.0).astype(np.float64)
    g = np.clip(G.astype(np.float64), 1e-6, 1.0)
    if cond:
        P = np.clip(P / g[:, None], 0.0, 1.0)
    pois = np.exp(-u) * np.array([u ** i / np.prod(range(1, i + 1)) for i in range(_UMAX + 1)])
    pois = pois / pois.sum()
    # prefix: distribution of true members among the first k (k = 0..J)
    pre = [np.ones((ne, 1))]
    for k in range(J):
        pre.append(_conv_bernoulli(pre[-1], P[:, k]))
    # suffix: distribution of true members among positions k..J-1, then + U
    suf = [None] * (J + 1)
    suf[J] = np.ones((ne, 1))
    for k in range(J - 1, -1, -1):
        suf[k] = _conv_bernoulli(suf[k + 1], P[:, k])
    E = np.zeros((ne, J + 1))
    E[:, 0] = 1.0 - g
    p_none = suf[0][:, 0] * pois[0]  # P(N = 0) under the member model
    denom = np.maximum(1.0 - p_none, 1e-12) if cond else np.ones(ne)
    for k in range(1, J + 1):
        if u > 0:  # + U: convolve with the (short) Poisson vector, vectorised over entities
            R = np.zeros((ne, suf[k].shape[1] + _UMAX))
            for m, w in enumerate(pois):
                R[:, m:m + suf[k].shape[1]] += suf[k] * w
        else:
            R = suf[k]
        t = np.arange(k + 1)[:, None]
        r = np.arange(R.shape[1])[None, :]
        # F0.5 = 1.25 TP / (k + 0.25 N) with TP = t true among the k predicted and N = t + r true in total
        M = np.where(t > 0, 1.25 * t / (k + 0.25 * (t + r)), 0.0)
        e = ((pre[k] @ M) * R).sum(1)
        E[:, k] = (g * e / denom) if cond else e
        E[k > n_elig, k] = -1.0  # cannot predict more members than exist
    return E


def select_expected(Q: np.ndarray, SRC: np.ndarray, G: np.ndarray, caps: dict, a: float = 1.0, b: float = 0.0,
                    u: float = 0.0, cond: bool = True) -> np.ndarray:
    """Boolean selection (ne, J) in the original column order of ``Q``."""
    ne, J = Q.shape
    idx, n_elig = _eligible_order(Q, SRC, caps)
    q = np.clip(np.take_along_axis(Q, idx, axis=1).astype(np.float64), 1e-6, 1 - 1e-6)
    P = 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))
    E = expected_f05(P, n_elig, G, u, cond)
    k = E.argmax(1)  # first maximum -> the smaller set wins ties
    take = np.arange(J)[None, :] < k[:, None]
    sel = np.zeros((ne, J), bool)
    np.put_along_axis(sel, idx, take, axis=1)
    return sel & (Q > 0)
