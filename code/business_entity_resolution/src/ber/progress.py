"""Progress / time-remaining report built from the stage completion markers (stdlib only).

The pipeline writes ``_markers/<stage>.*.done`` (with the stage's duration) when a stage finishes. This module turns
those into two bars:

    PIPELINE  ████████████░░░░░░░░░░░░  48%  stage 8/16: features | worked 2h05m | remaining ~2h10m (estimate)
    SESSION   ██████░░░░░░░░░░░░░░░░░░  27%  of the 12h limit used (3h14m) | 8h46m left | pipeline projected to end at ~5h24m

Estimates: every stage has a rough relative weight (minutes on a 4-core / 2xT4 box). Once stages have finished, the
weights are rescaled by how long they really took (clamped to 0.25x-6x), so the estimate improves as the run goes on.
The stage that is running is assumed to be no more than 95% done, and never to have less than 20% of its estimate left.
The session bar only appears when ``BER_SESSION_LIMIT_H`` is set (the Kaggle runner sets it, together with
``BER_SESSION_START``, the epoch second the notebook session started).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ORDER = ["ingest", "eda", "mine", "normalize", "dense", "block", "prerank", "expand", "features", "r1",
         "ce_train", "ce_infer", "r2", "gate", "tune", "predict", "outputs"]
TRAIN_ONLY = {"eda", "mine", "ce_train", "tune"}

# rough full-scale minutes (4 cores, 2x T4); only the ratios matter
MINUTES = {"ingest": 3, "eda": 1, "mine": 8, "normalize": 10, "dense": 30, "block": 50, "prerank": 60, "expand": 12,
           "features": 40, "r1": 15, "ce_train": 20, "ce_infer": 40, "r2": 20, "gate": 3, "tune": 8, "predict": 2,
           "outputs": 5}


def plan(ce_enabled: bool = True, expand_enabled: bool = True, dense_enabled: bool = False,
         inference_only: bool = False) -> list[tuple[str, float]]:
    """[(stage, weight_minutes)] of the stages this run will actually execute, in order."""
    out = []
    for s in ORDER:
        if (s == "dense" and not dense_enabled) or (s in ("ce_train", "ce_infer") and not ce_enabled):
            continue
        if inference_only and s in TRAIN_ONLY:
            continue
        out.append((s, 1.0 if (s == "expand" and not expand_enabled) else float(MINUTES[s])))
    return out


def plan_from_cfg(cfg) -> list[tuple[str, float]]:
    return plan(bool(cfg.ce.enabled), bool(cfg.expand.enabled), bool(cfg.blocking.get("dense", {}).get("enabled", False)),
                bool(cfg.run.get("inference_only", False)))


def read_markers(work_dir) -> dict[str, dict]:
    """stage -> {"seconds": total duration, "mtime": when it finished} from ``_markers/*.done``."""
    out: dict[str, dict] = {}
    d = Path(work_dir) / "_markers"
    if not d.is_dir():
        return out
    for f in d.glob("*.done"):
        stage = f.name.split(".")[0]
        try:
            secs = float(json.loads(f.read_text()).get("seconds", 0.0))
        except (ValueError, OSError):
            secs = 0.0
        prev = out.get(stage, {"seconds": 0.0, "mtime": 0.0})
        out[stage] = {"seconds": prev["seconds"] + secs, "mtime": max(prev["mtime"], f.stat().st_mtime)}
    return out


def bar(frac: float, width: int = 24) -> str:
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * n + "░" * (width - n)


def fmt(seconds: float) -> str:
    s = int(max(0, seconds))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s" if m < 10 else f"{m}m"


def report(work_dir, stages: list[tuple[str, float]], session_start: float | None = None,
           session_limit_h: float | None = None, run_started: float | None = None, now: float | None = None,
           hint: str = "") -> list[str]:
    """Status lines for the pipeline (and the session, when a limit is given)."""
    now = time.time() if now is None else now
    marks = read_markers(work_dir)
    total_w = sum(w for _, w in stages) or 1.0
    done = [(s, w) for s, w in stages if s in marks]
    todo = [(s, w) for s, w in stages if s not in marks]
    calib = [(w * 60.0, marks[s]["seconds"]) for s, w in done if w >= 5 and marks[s]["seconds"] > 0]
    scale = 1.0
    if calib:
        raw = min(6.0, max(0.25, sum(a for _, a in calib) / sum(e for e, _ in calib)))
        # trust the measured speed in proportion to how much of the run it covers: the first cheap stages say
        # little about the heavy GPU ones, so early estimates stay close to the prior weights
        trust = min(1.0, sum(w for _, w in done if w >= 5) / (0.3 * total_w))
        scale = 1.0 + (raw - 1.0) * trust
    worked = sum(marks[s]["seconds"] for s, _ in done)
    lines: list[str] = []
    remaining = 0.0
    if not todo:
        lines.append(f"PIPELINE  {bar(1.0)} 100%  complete | total {fmt(worked)}")
    else:
        cur, w_cur = todo[0]
        clock = max([m["mtime"] for m in marks.values()] + ([run_started] if run_started else []), default=now)
        el = max(0.0, now - clock)
        est = w_cur * 60.0 * scale
        frac_cur = min(0.95, el / est) if est > 0 else 0.0
        pct = (sum(w for _, w in done) + frac_cur * w_cur) / total_w
        remaining = max(est - el, 0.2 * est) + sum(w * 60.0 * scale for _, w in todo[1:])
        lines.append(f"PIPELINE  {bar(pct)} {pct * 100:3.0f}%  stage {len(done) + 1}/{len(stages)}: {cur} | "
                     f"worked {fmt(worked + el)} | remaining ~{fmt(remaining)} (estimate)")
    if session_start and session_limit_h:
        used, limit = max(0.0, now - session_start), session_limit_h * 3600.0
        line = (f"SESSION   {bar(used / limit)} {min(999, used / limit * 100):3.0f}%  of the {session_limit_h:g}h limit "
                f"used ({fmt(used)}) | {fmt(limit - used)} left")
        if todo:
            line += f" | pipeline projected to end at ~{fmt(used + remaining)}"
            if used + remaining > 0.95 * limit:
                line += "  !! may not finish in this session" + (f" -> {hint}" if hint else "")
        lines.append(line)
    return lines


def session_from_env() -> tuple[float | None, float | None]:
    try:
        start = float(os.environ["BER_SESSION_START"]) if "BER_SESSION_START" in os.environ else None
        limit = float(os.environ["BER_SESSION_LIMIT_H"]) if "BER_SESSION_LIMIT_H" in os.environ else None
    except ValueError:
        return None, None
    return start, limit
