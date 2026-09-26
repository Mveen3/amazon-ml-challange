"""Threshold tuning per country profile on full-train OOF, and final selection.

Objective: exact macro F0.5 over *all* train S1 of the profile. If the stress
test (docs §10.4) is enabled and its hypothesis is accepted, the objective is the
worse of (normal, stress) so thresholds stay robust to the denser test set.
Profiles with no training data (France, generic) inherit ``fallback_profile``
thresholds plus optional config ``overrides`` (leaderboard probes).
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl

from ..eval.metric import report
from ..models.gate import ownership
from ..utils import inference_only, load_json, log, save_json, work_dir
from .expected import select_expected
from .select import build_arrays, entity_f05, select_mask, selected_pairs


def _caps(cfg) -> dict:
    return {int(k): int(v) for k, v in cfg.decision.caps.items()}


def load_arrays(cfg, split: str, r2: pl.DataFrame | None = None, gate: pl.DataFrame | None = None) -> dict:
    r2 = r2 if r2 is not None else pl.read_parquet(work_dir(cfg, split, "preds", "r2.parquet"))
    gate = gate if gate is not None else pl.read_parquet(work_dir(cfg, split, "preds", "gate.parquet"))
    return build_arrays(ownership(r2.sort(["s1_uid", "rec_uid"])), gate.sort("s1_uid"), int(cfg.decision.max_members))


def _select(arr: dict, m: np.ndarray, th: dict, cfg) -> np.ndarray:
    """Selection for the rows ``m`` under one threshold set (heuristic rule or expected-F0.5 rule)."""
    if th.get("rule") == "expected":
        return select_expected(arr["Q"][m], arr["SRC"][m], arr["G"][m], _caps(cfg), th["a"], th["b"], th["u"],
                               bool(th["cond"]))
    return select_mask(arr["Q"][m], arr["SRC"][m], arr["G"][m], th["kappa"], th["t_min"], th["delta"],
                       cfg.decision.t_add_base, _caps(cfg))


def _score(arr: dict, m: np.ndarray, th: dict, cfg) -> float:
    return float(entity_f05(_select(arr, m, th, cfg), arr["Y"][m], arr["NTRUE"][m]).mean())


def _search_expected(cfg, arr: dict, prof: str, arr_stress: dict | None) -> dict | None:
    """Grid over the expected-F0.5 rule's knobs (improvement.md A4); None when switched off."""
    ec = cfg.decision.get("expected") or {}
    if not ec.get("enabled", False):
        return None
    m = arr["PROF"] == prof
    ms = arr_stress["PROF"] == prof if arr_stress is not None else None
    best = None
    for a, b, u, cond in itertools.product(ec.a, ec.b, ec.u, ec.cond):
        th = {"rule": "expected", "a": float(a), "b": float(b), "u": float(u), "cond": bool(cond)}
        f = _score(arr, m, th, cfg)
        fs = _score(arr_stress, ms, th, cfg) if arr_stress is not None else f
        if best is None or min(f, fs) > best["objective"]:
            best = {**th, "objective": min(f, fs), "f_normal": f, "f_stress": fs}
    return best


def search(cfg, arr: dict, prof: str, arr_stress: dict | None = None) -> dict:
    gc = cfg.decision.grid
    m = arr["PROF"] == prof
    ms = arr_stress["PROF"] == prof if arr_stress is not None else None
    best = None
    for kappa, t_min, delta in itertools.product(gc.kappa, gc.t_min, gc.delta):
        th = {"kappa": float(kappa), "t_min": float(t_min), "delta": float(delta)}
        f = _score(arr, m, th, cfg)
        fs = _score(arr_stress, ms, th, cfg) if arr_stress is not None else f
        obj = min(f, fs)
        if best is None or obj > best["objective"]:
            best = {**th, "rule": "heuristic", "objective": obj, "f_normal": f, "f_stress": fs}
    log().info("  tuned %s (heuristic rule): %s", prof, {k: round(v, 5) for k, v in best.items() if k != "rule"})
    exp = _search_expected(cfg, arr, prof, arr_stress)
    if exp is not None:
        log().info("  tuned %s (expected-F0.5 rule): %s", prof,
                   {k: (round(v, 5) if isinstance(v, float) else v) for k, v in exp.items() if k != "rule"})
        if exp["objective"] > best["objective"] + float(cfg.decision.expected.get("min_gain", 0.0)):
            log().info("  %s: expected-F0.5 rule wins (+%.5f on train out-of-fold)", prof,
                       exp["objective"] - best["objective"])
            return {**exp, "heuristic": best}
    return best


def run_tune(cfg, arr_stress: dict | None = None) -> dict:
    arr = load_arrays(cfg, "train")
    th = {"_fallback": cfg.decision.fallback_profile}
    for prof in sorted(set(arr["PROF"].tolist())):
        th[prof] = search(cfg, arr, prof, arr_stress)
    save_json(th, work_dir(cfg, None, "models", "thresholds.json"))
    return th


def thresholds_for(cfg, th: dict, prof: str) -> dict:
    base = dict(th.get(prof) or th[th["_fallback"]])
    ov = (cfg.decision.get("overrides") or {}).get(prof, {}) or {}
    if ov.get("rule") == "heuristic" and "heuristic" in base:  # probe: fall back to the heuristic rule
        base = dict(base["heuristic"])
    if base.get("rule") == "expected":
        for k in ("a", "b", "u"):
            if k in ov:
                base[k] = float(ov[k])
        base["b"] = base["b"] + float(ov.get("b_shift", 0.0))  # > 0 predicts more, < 0 fewer
        if ov.get("empty", False):
            base = {"rule": "heuristic", "kappa": 0.0, "t_min": 1.0, "delta": 0.0}
        return base
    for k in ("kappa", "t_min", "delta"):
        if k in ov:
            base[k] = float(ov[k])
    base["delta"] = base["delta"] + float(ov.get("delta_shift", 0.0))
    if ov.get("empty", False):
        base["kappa"] = 0.0  # never predicts anything (France-empty probe)
    return base


def apply(cfg, arr: dict, th: dict) -> pl.DataFrame:
    sel = np.zeros_like(arr["Q"], dtype=bool)
    for prof in sorted(set(arr["PROF"].tolist())):
        m = arr["PROF"] == prof
        sel[m] = _select(arr, m, thresholds_for(cfg, th, prof), cfg)
    return selected_pairs(arr, sel)


def run_predict(cfg, tag: str | None = None) -> dict:
    th = load_json(work_dir(cfg, None, "models", "thresholds.json"))
    rep = {} if inference_only(cfg) else _train_report(cfg, th, tag)
    arr = load_arrays(cfg, "test")
    pred = apply(cfg, arr, th)
    pred.write_parquet(work_dir(cfg, "test", "preds", f"selected{'_' + tag if tag else ''}.parquet"))
    log().info("  test: %d matches for %d S1 (%.2f per S1)", pred.height, len(arr["S1"]),
               pred.height / max(1, len(arr["S1"])))
    return rep


def _train_report(cfg, th: dict, tag: str | None) -> dict:
    # train OOF report with the tuned thresholds (overrides only matter for untrained profiles)
    arr = load_arrays(cfg, "train")
    pred = apply(cfg, arr, th)
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
    cands = pl.read_parquet(work_dir(cfg, "train", "cands", "final.parquet"), columns=["s1_uid", "rec_uid"])
    rep = report(pred, truth, s1, cands)
    save_json(rep, work_dir(cfg, None, "models", f"oof_report{'_' + tag if tag else ''}.json"))
    log().info("  OOF macro F0.5 = %.5f (ceiling %.5f) | %s", rep["macro_f05"], rep["ceiling"], rep["by_profile"])
    return rep
