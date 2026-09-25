"""Write ``matching_results.tsv`` and ``candidate_pairs.tsv`` and run the official validator."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from ..io import write_id_lists
from ..utils import log, work_dir


def write_outputs(cfg, probe: str | None = None) -> Path:
    out_dir = Path(cfg.paths.output_dir) / ("probes/" + probe if probe else "")
    rec = pl.read_parquet(work_dir(cfg, "test", "records.parquet"), columns=["uid", "eid", "src"])
    s1 = pl.read_parquet(work_dir(cfg, "test", "s1.parquet"), columns=["s1_uid", "eid"])
    cands = pl.read_parquet(work_dir(cfg, "test", "cands", "final.parquet"), columns=["s1_uid", "rec_uid", "p_pre"])
    if probe == "all_empty":
        pred = pl.DataFrame(schema={"s1_uid": pl.Int64, "rec_uid": pl.Int64, "q": pl.Float32})
    else:
        name = f"selected_{probe}.parquet" if probe else "selected.parquet"
        pred = pl.read_parquet(work_dir(cfg, "test", "preds", name))
    n_before = pred.height
    pred = pred.join(cands.select(["s1_uid", "rec_uid"]), on=["s1_uid", "rec_uid"], how="semi")
    assert pred.height == n_before, "matched pair outside candidate set: pipeline bug"
    write_id_lists(s1, cands, rec, out_dir / "candidate_pairs.tsv",
                   ["source1_entity_id", "candidate_entity_ids"], order_col="p_pre")
    write_id_lists(s1, pred, rec, out_dir / "matching_results.tsv",
                   ["source1_entity_id", "matched_entity_ids"], order_col="q")
    log().info("  wrote %s (%d matches, %d candidates)", out_dir, pred.height, cands.height)
    validate(cfg, out_dir)
    return out_dir


def validate(cfg, out_dir: Path) -> bool:
    validator = Path(cfg.paths.validator)
    if not validator.exists():
        log().warning("  validator not found at %s (skipped)", validator)
        return True
    cmd = [sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(Path(cfg.paths.data_dir) / "test")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    log().info("  validator (exit %d):\n%s", res.returncode, (res.stdout + res.stderr).strip()[-2000:])
    return res.returncode == 0
