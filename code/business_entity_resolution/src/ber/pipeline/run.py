"""Command-line entry point.

    python -m ber.pipeline.run --config configs/default.yaml --stage all
    python -m ber.pipeline.run --config configs/default.yaml --stage features --force
    python -m ber.pipeline.run --config configs/default.yaml --from r2
    python -m ber.pipeline.run --config configs/default.yaml --stage ce_infer --split test --shard 0/4
    python -m ber.pipeline.run --config configs/default.yaml --stage predict,outputs --probe fr_strict \
        --set decision.overrides.france.delta_shift=0.05

Every stage writes its artefacts under ``paths.work_dir`` and a completion marker,
so the pipeline resumes where it stopped; ``--force`` re-runs completed stages.
"""
from __future__ import annotations

import argparse
import time

from ..config import load_config
from ..utils import inference_only, log, mark_done, save_json, set_seed, stage_done, timer, work_dir

PIPELINE = ["ingest", "eda", "mine", "normalize", "dense", "block", "prerank", "expand", "features", "r1",
            "ce_train", "ce_infer", "r2", "gate", "tune", "predict", "outputs"]
EXTRA = ["s1s1", "stress_check"]
SPLITS = ("train", "test")


TRAIN_ONLY = {"eda", "mine", "ce_train", "tune", "s1s1", "stress_check"}
_CFG = None


def _splits(args) -> tuple:
    if _CFG is not None and inference_only(_CFG):
        return ("test",)
    return SPLITS if args.split in (None, "both") else (args.split,)


def run_stage(cfg, stage: str, args) -> dict | None:
    if stage == "ingest":
        from ..io import ingest
        for s in _splits(args):
            ingest(cfg, s)
    elif stage == "eda":
        from ..eda.profile import run_eda
        run_eda(cfg)
    elif stage == "mine":
        from ..mining.tables import mine
        from ..normalize.tables import Tables
        if cfg.mining.enabled:
            mine(cfg)
        else:
            Tables().save(work_dir(cfg, None, "tables.pkl"))
    elif stage == "normalize":
        from ..normalize.run import normalize_split
        for s in _splits(args):
            normalize_split(cfg, s)
    elif stage == "dense":
        if not cfg.blocking.dense.enabled:
            log().info("  dense channel disabled (blocking.dense.enabled=false)")
            return None
        from ..models import biencoder
        if not inference_only(cfg) and (not work_dir(cfg, None, "models", "biencoder", "half1").exists() or args.force):
            biencoder.train(cfg)
        for s in _splits(args):
            biencoder.hits(cfg, s)
    elif stage == "block":
        import polars as pl

        from ..blocking.candidates import block_split
        from ..eval.metric import ceiling
        info = {}
        for s in _splits(args):
            union = block_split(cfg, s)
            if s == "train":
                truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
                s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
                hit = union.join(truth, on=["s1_uid", "rec_uid"]).height
                info = {"union_ceiling": ceiling(union, truth, s1), "pair_recall": hit / truth.height,
                        "pairs_per_s1": union.height / s1.height}
                log().info("  train union: %s", info)
        return info
    elif stage == "prerank":
        from ..blocking.prerank import run_prerank
        run_prerank(cfg)
    elif stage == "expand":
        from ..blocking.expand import run_expand
        run_expand(cfg)
    elif stage == "features":
        from ..features.pair import run_features
        for s in _splits(args):
            run_features(cfg, s)
    elif stage == "r1":
        from ..models.rounds import run_r1
        run_r1(cfg)
    elif stage == "ce_train":
        if not cfg.ce.enabled:
            log().info("  cross-encoder disabled (ce.enabled=false)")
            return None
        from ..models.cross_encoder import train_half
        for h in (0, 1):
            if not work_dir(cfg, None, "models", "ce", f"half{h}").exists() or args.force:
                train_half(cfg, h)
    elif stage == "ce_infer":
        if not cfg.ce.enabled:
            return None
        from ..models.cross_encoder import infer
        i, n = (int(x) for x in (args.shard or "0/1").split("/"))
        for s in _splits(args):
            infer(cfg, s, i, n)
    elif stage == "r2":
        from ..models.rounds import run_r2
        run_r2(cfg)
    elif stage == "gate":
        from ..models.gate import run_gate
        run_gate(cfg)
    elif stage == "stress_check":
        from ..eval.stress import check
        return check(cfg)
    elif stage == "tune":
        from ..decision.tune import run_tune
        arr_stress = None
        if cfg.stress.enabled:
            from ..eval.stress import check, simulate
            res = check(cfg)
            if res["accepted"] or cfg.stress.get("force", False):
                arr_stress = simulate(cfg)
        return run_tune(cfg, arr_stress)
    elif stage == "predict":
        from ..decision.tune import run_predict
        return run_predict(cfg, args.probe)
    elif stage == "outputs":
        from ..pipeline.outputs import write_outputs
        write_outputs(cfg, args.probe)
    elif stage == "s1s1":
        from ..eval.s1s1 import run_s1s1
        return run_s1s1(cfg)
    else:
        raise ValueError(f"unknown stage {stage!r}; choose from {PIPELINE + EXTRA}")
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="stage name, comma list, or 'all'")
    ap.add_argument("--from", dest="start", default=None, help="run the pipeline from this stage onwards")
    ap.add_argument("--split", default=None, choices=["train", "test", "both"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="dotted.key=value (YAML value)")
    ap.add_argument("--probe", default=None, help="name for a leaderboard-probe variant (predict/outputs)")
    ap.add_argument("--shard", default=None, help="i/n sharding for ce_infer across GPUs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    global _CFG
    _CFG = cfg
    set_seed(int(cfg.run.seed))
    if args.start:
        stages = PIPELINE[PIPELINE.index(args.start):]
    elif args.stage in (None, "all"):
        stages = PIPELINE
    else:
        stages = [s.strip() for s in args.stage.split(",")]
    save_json(cfg, work_dir(cfg, None, "_last_config.json"))
    for st in stages:
        if inference_only(cfg) and st in TRAIN_ONLY:
            log().info("⏭  %s skipped (inference_only)", st)
            continue
        per_split = args.split not in (None, "both")
        marker_split = args.split if per_split else None
        if st not in ("predict", "outputs") and not args.force and not args.probe and stage_done(cfg, st, marker_split):
            log().info("⏭  %s already done (use --force to re-run)", st)
            continue
        t0 = time.time()
        with timer(f"stage {st}"):
            info = run_stage(cfg, st, args)
        mark_done(cfg, st, marker_split, {"seconds": round(time.time() - t0, 1), "info": info})


if __name__ == "__main__":
    main()
