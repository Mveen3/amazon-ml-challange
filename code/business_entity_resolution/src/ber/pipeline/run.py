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
import os
import sys
import time
from pathlib import Path

from ..config import load_config
from ..progress import ORDER, plan_from_cfg, report, session_from_env
from ..utils import inference_only, log, mark_done, save_json, set_seed, stage_done, timer, work_dir

EXIT_OOM = 75  # stage ran out of memory (Python MemoryError); the Kaggle runner retries with lower-memory settings
PIPELINE = list(ORDER)  # single source of truth: ber.progress.ORDER
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
        import gc

        import polars as pl

        from ..blocking.candidates import block_split, union_files
        from ..eval.metric import macro_f05
        info = {}
        for s in _splits(args):
            info[s] = block_split(cfg, s)
            if s == "train":
                # Stream per-country file to avoid loading all 179M union pairs at once (OOM on 34 GB).
                # Only accumulate the *hits* (pairs that match truth), which are << total union size.
                truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
                s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"))
                total_pairs = 0
                all_hits = []
                for uf in union_files(cfg, "train"):
                    chunk = pl.scan_parquet(str(uf)).select(["s1_uid", "rec_uid"]).collect()
                    total_pairs += chunk.height
                    hits = chunk.join(truth, on=["s1_uid", "rec_uid"])
                    if hits.height > 0:
                        all_hits.append(hits)
                    del chunk, hits
                    gc.collect()
                # The accumulated hits are ~truth-sized (a few million), not union-sized (179M).
                hit_df = pl.concat(all_hits) if all_hits else pl.DataFrame({"s1_uid": [], "rec_uid": []})
                del all_hits
                ceil_val = macro_f05(hit_df, truth, s1)
                info["train_summary"] = {"union_ceiling": ceil_val,
                                         "pair_recall": hit_df.height / max(1, truth.height),
                                         "pairs_per_s1": total_pairs / s1.height}
                log().info("  train union: %s", info["train_summary"])
                del truth, s1, hit_df
                gc.collect()
        return info
    elif stage == "prerank":
        import shutil

        from ..blocking.candidates import union_dir
        from ..blocking.prerank import run_prerank
        if args.force:  # --force means retrain: drop models / scores kept for resuming an interrupted attempt
            for f in [work_dir(cfg, None, "models", "prerank", "trained.json")] + \
                     [work_dir(cfg, sp, "cands", "pre_all.json") for sp in SPLITS]:
                f.unlink(missing_ok=True)
        run_prerank(cfg)
        if cfg.run.get("cleanup", False):  # the union files are only read by the pre-ranker
            for s in _splits(args):
                shutil.rmtree(union_dir(cfg, s), ignore_errors=True)
            log().info("  cleanup: removed candidate-union files")
    elif stage == "expand":
        from ..blocking.expand import run_expand
        run_expand(cfg)
    elif stage == "features":
        from ..features.pair import run_features
        for s in _splits(args):
            if args.force:  # --force means recompute: drop the plan that lets an interrupted attempt keep its shards
                work_dir(cfg, s, "feats", "r1_plan.json").unlink(missing_ok=True)
            run_features(cfg, s)
    elif stage == "r1":
        from ..models.rounds import run_r1
        run_r1(cfg)
    elif stage == "ce_train":
        if not cfg.ce.enabled:
            log().info("  cross-encoder disabled (ce.enabled=false)")
            return None
        from ..checkpoint import sync_now
        from ..models import ce_parallel
        from ..models.cross_encoder import train_half
        pending = [h for h in (0, 1) if not work_dir(cfg, None, "models", "ce", f"half{h}").exists() or args.force]
        if len(pending) == 2 and ce_parallel.usable(cfg) and ce_parallel.train_halves(cfg, pending):
            sync_now("cross-encoder halves trained (one per GPU)")
        else:  # one GPU, one half left, or a worker failed: the sequential path (DataParallel over all GPUs)
            for h in pending:
                if args.force or not work_dir(cfg, None, "models", "ce", f"half{h}").exists():
                    train_half(cfg, h)
                    sync_now(f"cross-encoder half {h} trained")  # a new session resumes with the next half
    elif stage == "ce_infer":
        if not cfg.ce.enabled:
            return None
        import shutil

        from ..checkpoint import sync_now
        from ..models import ce_parallel
        from ..models.cross_encoder import infer, infer_half, merge_halves
        i, n = (int(x) for x in (args.shard or "0/1").split("/"))
        for s in _splits(args):
            if args.force:  # recompute: drop scores kept for resuming an interrupted attempt
                for h in (0, 1):
                    for f in ("parquet", "json"):
                        work_dir(cfg, s, "preds", f"ce_half{h}.{f}").unlink(missing_ok=True)
                    shutil.rmtree(work_dir(cfg, s, "preds", f"ce_half{h}_parts"), ignore_errors=True)
            elif work_dir(cfg, s, "preds", f"ce_part{i:03d}.parquet").exists():
                log().info("  CE %s shard %d/%d already scored (restored or earlier run)", s, i, n)
                continue
            if (i, n) == (0, 1):
                if not (ce_parallel.usable(cfg) and ce_parallel.infer_halves(cfg, s)):
                    for h in (0, 1):  # one GPU, or a worker failed: the halves one after the other (resumable too)
                        infer_half(cfg, s, h)
                        sync_now(f"cross-encoder half {h} scored {s}")
                    merge_halves(cfg, s)
                sync_now(f"cross-encoder scores for {s}")
                continue
            infer(cfg, s, i, n)  # explicit --shard i/n (several machines): not resumable within a shard
            sync_now(f"cross-encoder scores for {s} shard {i}/{n}")
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
    ap.add_argument("--fresh-start", action="store_true",
                    help="delete this run's Hugging Face checkpoint first (with checkpoint.enabled)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    cfg["_cli"] = {"config": str(Path(args.config).resolve()), "overrides": list(args.overrides)}  # for worker processes
    if int(cfg.run.get("max_gpus", 0)) > 0:  # global GPU cap, read by every stage (and by worker processes via env)
        os.environ["BER_MAX_GPUS"] = str(int(cfg.run.max_gpus))
    global _CFG
    _CFG = cfg
    set_seed(int(cfg.run.seed))
    if args.start:
        stages = PIPELINE[PIPELINE.index(args.start):]
    elif args.stage in (None, "all"):
        stages = PIPELINE
    else:
        stages = [s.strip() for s in args.stage.split(",")]
    from ..checkpoint import activate
    ckpt = activate(cfg)
    if args.fresh_start:
        ckpt.reset()
    ckpt.restore()  # new session: pull finished stages from Hugging Face, so they are skipped below
    save_json(cfg, work_dir(cfg, None, "_last_config.json"))
    t_main = time.time()
    sess_start, sess_limit = session_from_env()

    def show_progress():
        for line in report(work_dir(cfg, None), plan_from_cfg(cfg), sess_start, sess_limit, t_main,
                           hint="commit again to resume from the checkpoint" if ckpt.enabled else ""):
            log().info(line)

    show_progress()  # 0% on a fresh start; after a checkpoint restore it shows what is already finished
    for st in stages:
        if inference_only(cfg) and st in TRAIN_ONLY:
            log().info("⏭  %s skipped (inference_only)", st)
            continue
        per_split = args.split not in (None, "both")
        marker_split = args.split if per_split else None
        if st not in ("predict", "outputs") and not args.force and not args.probe and stage_done(cfg, st, marker_split):
            log().info("⏭  %s already done (use --force to re-run)", st)
            continue
        # A stage being (re-)run is not done until it finishes: if this attempt dies half-way, a stale marker from an
        # earlier completion must not make the next run skip it (its outputs may be half replaced by now).
        work_dir(cfg, None, "_markers", f"{st}.{marker_split or 'all'}.done").unlink(missing_ok=True)
        t0 = time.time()
        try:
            with timer(f"stage {st}"):
                info = run_stage(cfg, st, args)
        except MemoryError:
            # Out of memory raised inside Python (rather than the kernel killing the process). The local work dir
            # is intact and finished sub-steps are kept; exit with a dedicated code so the Kaggle runner repeats the
            # stage with lower-memory settings.
            log().error("  stage %s ran out of memory -> exit %d (the runner retries with lower-memory settings)",
                        st, EXIT_OOM)
            ckpt.sync(f"stage {st} interrupted (out of memory): partial progress")
            sys.exit(EXIT_OOM)
        mark_done(cfg, st, marker_split, {"seconds": round(time.time() - t0, 1), "info": info})
        ckpt.sync(f"stage {st} done")  # outputs + completion marker in one commit
        show_progress()


if __name__ == "__main__":
    main()
