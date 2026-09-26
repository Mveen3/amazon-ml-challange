"""Kaggle notebook driver. The notebook keeps only its settings and the ``git clone``; every other cell
calls a method here, so fixes reach Kaggle with the next clone instead of a notebook re-import.

    R = KaggleRun(pkg=PKG, team="neural_nexus", ...)
    R.hardware(); R.install(); R.secrets(); R.data()
    R.smoke()                               # optional rehearsal of the full run on a 5k-entity sample
    R.stage("ingest,eda,mine,normalize"); ...; R.results(); R.package()
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# sample-scale overrides that let configs/kaggle.yaml (the real full-run config) run on 5k entities
SMOKE_SETS = ["mining.min_support=3", "blocking.keyblock.admin_df_min=5", "blocking.keyblock.admin_df_max=3000",
              "prerank.chunk_rows=50000", "features.shard_rows=3000", "features.join_rows=10000",
              "ce.max_pos=3000", "ce.infer_chunk=5000",
              "blocking.knn.mem_gb=0.002",  # tiny kNN memory budget -> many small query chunks, so both GPUs get work
              # the rehearsal is a self-contained run: with a track config it must never restore the full-data
              # base folder into the sample work dir
              "checkpoint.prefix=smoke-test", "checkpoint.restore_from=null", "checkpoint.start_from=null",
              "run.inference_only=false"]  # an inference-only track still rehearses a full sample run


def q(x) -> str:
    """Shell-quote a path/argument (paths may contain spaces outside Kaggle)."""
    return shlex.quote(str(x))


# Exit codes that mean "out of memory": the kernel's OOM killer (SIGKILL -> -9, or 137 via a shell), SIGABRT from a
# native allocation failure (-6 / 134), and EXIT_OOM (75) from ber.pipeline.run on a Python MemoryError.
OOM_EXIT_CODES = (-9, 137, -6, 134, 75)

# Settings for a stage retried after running out of memory. They only shrink memory use: fold models train one at a
# time, training samples shrink, fewer worker processes run at once. Chunk / shard sizes are NOT changed, because they
# are part of the resume plans (changing them would throw away finished sub-steps).
_SEQUENTIAL_FOLDS = [f"{s}.gbdt.parallel_folds=false" for s in ("prerank", "r1", "r2", "gate")]
LOW_MEMORY_LEVELS = [
    _SEQUENTIAL_FOLDS + ["prerank.max_train_rows=6000000", "r1.sample_entities=500000", "r2.sample_entities=500000",
                         "r1.max_train_rows=6000000", "r2.max_train_rows=6000000",
                         "ce.parallel_halves=false", "ce.infer_chunk=250000"],
    _SEQUENTIAL_FOLDS + ["prerank.max_train_rows=3000000", "r1.sample_entities=300000", "r2.sample_entities=300000",
                         "r1.max_train_rows=4000000", "r2.max_train_rows=4000000",
                         "ce.parallel_halves=false", "ce.infer_chunk=100000", "run.n_workers=2"],
]


class KaggleRun:
    def __init__(self, pkg, config="configs/kaggle.yaml", dataset_slug="amazon-ml", team="neural_nexus",
                 use_cross_encoder=True, use_hf_checkpoint=True, fresh_start=False, overrides=(),
                 working="/kaggle/working", input_root="/kaggle/input", scratch=None):
        self.pkg = Path(pkg)
        self.config = config
        self.slug = dataset_slug
        self.team = re.sub(r"[^A-Za-z0-9_-]+", "_", str(team).strip()).strip("_") or "team"
        self.use_ce = bool(use_cross_encoder)
        self.use_ckpt = bool(use_hf_checkpoint)
        self.fresh_pending = bool(fresh_start)
        self.overrides = list(overrides)
        self.working = Path(working)
        self.input_root = input_root
        self.logs = self.working / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()                 # session clock (Kaggle stops a session after 12 h)
        self.heartbeat_s = 300                # seconds between automatic progress reports while a stage runs
        self.session_limit_h = 12.0
        self._last_line = ""
        self._gpu: dict[str, dict] = {}       # per-GPU utilisation samples since the last report
        try:
            import psutil
            psutil.cpu_percent(interval=None)  # prime: later calls report the average since the previous call
        except ImportError:
            pass
        self._run_started = time.time()
        if str(self.pkg / "src") not in sys.path:  # only for ber.progress / ber.config in this process
            sys.path.insert(0, str(self.pkg / "src"))
        self.scratch_override = scratch
        self.scratch = self.data_dir = self.work_dir = None
        self.output_dir = self.pkg.parent.parent / "output"
        print(f"team: {self.team} | config: {config} | cross-encoder: {self.use_ce} | "
              f"HF checkpoints: {self.use_ckpt} | fresh start: {self.fresh_pending}")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def sh(cmd: str, check: bool = True):
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if r.stdout:
            print(r.stdout[-6000:])
        if r.stderr:
            print(r.stderr[-3000:])
        if check and r.returncode:
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
        return r

    def _env(self) -> dict:
        return dict(os.environ, PYTHONPATH=str(self.pkg / "src"), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                    HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
                    BER_SESSION_START=str(self.t0), BER_SESSION_LIMIT_H=str(self.session_limit_h))

    def _hours(self) -> str:
        return f"{(time.time() - self.t0) / 3600:.2f} h of 12"

    # ------------------------------------------------------------------ setup cells
    def hardware(self) -> None:
        self.sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv", check=False)
        print("CPU cores:", os.cpu_count())
        try:
            import psutil
            print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        except ImportError:
            pass
        for p in [str(self.working), "/kaggle/temp", "/tmp"]:
            try:
                os.makedirs(p, exist_ok=True)
                print(f"{p}: {shutil.disk_usage(p).free / 1e9:.0f} GB free")
            except OSError as e:
                print(p, e)
        import torch
        n = torch.cuda.device_count()
        print("torch", torch.__version__, "| CUDA GPUs:", n,
              "|", ", ".join(f"{torch.cuda.get_device_name(i)} (cc {torch.cuda.get_device_capability(i)})"
                             for i in range(n)))
        if n == 0:
            print("WARNING: no GPU - set Accelerator to 'GPU T4 x2' (CPU-only is far slower).")

    def install(self) -> None:
        self.sh(f"{q(sys.executable)} -m pip install -q -r {q(self.pkg / 'requirements-kaggle.txt')}")
        self.sh(f"{q(sys.executable)} -c \"import polars, rapidfuzz, xgboost, transformers, sklearn, huggingface_hub as h; "
                f"print('polars', polars.__version__, '| rapidfuzz', rapidfuzz.__version__, '| xgboost', "
                f"xgboost.__version__, '| transformers', transformers.__version__, '| sklearn', sklearn.__version__, "
                f"'| huggingface_hub', h.__version__)\"")

    def secrets(self) -> bool:
        """Load the HF_TOKEN Kaggle Secret (Add-ons -> Secrets) into the environment; the value is never shown."""
        if not self.use_ckpt:
            print("Hugging Face checkpoints switched off (USE_HF_CHECKPOINT = False).")
            return False
        tok, err = os.environ.get("HF_TOKEN"), None
        if not tok:
            try:
                from kaggle_secrets import UserSecretsClient
                tok = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception as e:  # secret missing or not attached to this notebook
                err = e
        if tok and tok.strip():
            os.environ["HF_TOKEN"] = tok.strip()
            print("HF_TOKEN loaded (value hidden) -> every finished stage is checkpointed to a private HF repo.")
            return True
        self.use_ckpt = False
        print(f"No HF_TOKEN secret available ({type(err).__name__ if err else 'empty'}): the run works, but without "
              "checkpoints. Add-ons -> Secrets -> add HF_TOKEN and tick it for this notebook to enable them.")
        return False

    def data(self) -> None:
        if str(self.pkg / "scripts") not in sys.path:
            sys.path.insert(0, str(self.pkg / "scripts"))
        import kaggle_prepare

        self.scratch = Path(self.scratch_override) if self.scratch_override else kaggle_prepare.pick_scratch()
        self.data_dir = self.scratch / "dataset"
        self.work_dir = self.scratch / "ber_work"
        kaggle_prepare.prepare(self.input_root, self.slug, str(self.data_dir))

    # ------------------------------------------------------------------ pipeline
    def _sets(self, data_dir=None, work_dir=None, output_dir=None) -> list[str]:
        sets = [f"paths.work_dir={work_dir or self.work_dir}", f"paths.data_dir={data_dir or self.data_dir}",
                f"paths.output_dir={output_dir or self.output_dir}"]
        if not self.use_ce:
            sets.append("ce.enabled=false")
        if not self.use_ckpt:
            sets.append("checkpoint.enabled=false")
        return sets

    def _sample_gpus(self) -> None:
        """One nvidia-smi reading per GPU, accumulated until the next report (a single snapshot can miss a busy GPU)."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                i, util, used, total = [x.strip() for x in line.split(",")]
                g = self._gpu.setdefault(i, {"util": [], "mem": 0.0, "total": float(total)})
                g["util"].append(float(util))
                g["mem"] = max(g["mem"], float(used))
        except Exception:
            pass

    def resources(self) -> str:
        """'GPU0 avg 61% peak 100% mem 9.8/15 GB | GPU1 ... | CPU 92% | RAM 14/34 GB' since the last report."""
        parts = []
        for i, g in sorted(self._gpu.items()):
            if g["util"]:
                parts.append(f"GPU{i} avg {sum(g['util']) / len(g['util']):.0f}% peak {max(g['util']):.0f}% "
                             f"mem {g['mem'] / 1024:.1f}/{g['total'] / 1024:.0f} GB")
        self._gpu = {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            parts.append(f"CPU {psutil.cpu_percent(interval=None):.0f}%")
            parts.append(f"RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.0f} GB")
        except ImportError:
            pass
        return " | ".join(parts)

    def status(self, sets: list[str] | None = None, activity: bool = False) -> None:
        """Print the pipeline + session progress bars (elapsed, remaining, projected finish)."""
        try:
            from ber.config import load_config
            from ber.progress import plan_from_cfg, report

            sets = sets if sets is not None else self._sets() + self.overrides
            cfg = load_config(self.pkg / self.config, sets)
            hint = "commit again to resume from the checkpoint" if self.use_ckpt else "set USE_CROSS_ENCODER = False"
            lines = report(cfg.paths.work_dir, plan_from_cfg(cfg), self.t0, self.session_limit_h, self._run_started,
                           hint=hint)
        except Exception as e:  # progress display must never break a run
            lines = [f"(progress unavailable: {type(e).__name__}: {e})"]
        stamp = time.strftime("%H:%M:%S")
        print(f"\n[{stamp}] " + "\n           ".join(lines), flush=True)
        if activity:
            res = self.resources()
            if res:
                print(f"           RESOURCES {res} (since the last report)", flush=True)
            if self._last_line:
                print(f"           last log line: {self._last_line[:110]}", flush=True)

    def _heartbeat(self, stop: threading.Event, sets: list[str]) -> None:
        tick = max(1.0, min(5.0, self.heartbeat_s / 2))
        next_beat = time.time() + self.heartbeat_s
        while not stop.wait(tick):
            self._sample_gpus()
            if time.time() >= next_beat:
                self.status(sets, activity=True)
                next_beat = time.time() + self.heartbeat_s

    def run(self, stages: str, sets: list[str], log_name: str = "pipeline.log", fresh: bool = False,
            _level: int = 0) -> None:
        cmd = [sys.executable, "-u", "-m", "ber.pipeline.run", "--config", self.config, "--stage", stages]
        for s in sets:
            cmd += ["--set", s]
        if fresh:
            cmd.append("--fresh-start")
        print("$", " ".join(cmd))
        t0 = self._run_started = time.time()
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(stop, sets), daemon=True)
        beat.start()
        try:
            with open(self.logs / log_name, "a") as log:
                p = subprocess.Popen(cmd, cwd=self.pkg, env=self._env(), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                    if line.strip():
                        self._last_line = line.strip()
                p.wait()
        finally:
            stop.set()
        print(f"\n[{stages}] exit {p.returncode} in {(time.time() - t0) / 60:.1f} min | session time used: {self._hours()}")
        self.status(sets)
        if p.returncode != 0:
            out_of_memory = p.returncode in OOM_EXIT_CODES
            if out_of_memory and _level < len(LOW_MEMORY_LEVELS):
                extra = LOW_MEMORY_LEVELS[_level]
                print(f"\n⚠ '{stages}' ran out of memory (exit {p.returncode}). The local work dir is intact and "
                      f"finished sub-steps are kept, so the retry continues where it stopped, with lower-memory "
                      f"settings (level {_level + 1}/{len(LOW_MEMORY_LEVELS)}):\n   " + " ".join(extra) + "\n")
                return self.run(stages, sets + extra, log_name=log_name, fresh=False, _level=_level + 1)
            hint = ("it still ran out of memory at the lowest-memory settings" if out_of_memory
                    else "see the error above")
            raise RuntimeError(f"stage(s) '{stages}' failed ({hint}; full log: {self.logs / log_name}). Everything "
                               "finished so far is checkpointed: Save & Run All again resumes from there.")

    def stage(self, stages: str) -> None:
        """One group of full-run stages (finished stages are skipped, also after a checkpoint restore)."""
        fresh, self.fresh_pending = self.fresh_pending, False
        self.run(stages, self._sets() + self.overrides, fresh=fresh)


    def smoke(self) -> None:
        """Rehearse the full run (same config, same code) on a 5k-entity sample; with HF checkpoints on, also
        prove the save -> restore -> skip cycle a timed-out session relies on, then delete the test checkpoint."""
        sd = self.scratch / "smoke"
        shutil.rmtree(sd, ignore_errors=True)  # stale stage markers would silently skip stages
        self.sh(f"cd {q(self.pkg)} && {q(sys.executable)} scripts/make_sample.py --data {q(self.data_dir)} "
                f"--out {q(sd / 'sample_data')}")
        sets = self._sets(f"{sd}/sample_data", f"{sd}/work", f"{sd}/output") + SMOKE_SETS
        log_file = self.logs / "smoke.log"
        start = log_file.stat().st_size if log_file.exists() else 0
        self.run("all", sets, log_name="smoke.log", fresh=True)
        self._check_multi_gpu(log_file.read_text(errors="replace")[start:])
        self.sh(f"cd {q(self.pkg)} && {q(sys.executable)} scripts/score_sample.py --out {q(sd / 'output')} "
                f"--truth {q(sd / 'sample_data/test_truth.tsv')} --s1 {q(sd / 'sample_data/test/test_source1.tsv')}")
        if not self.use_ckpt:
            print("SMOKE TEST PASSED (no checkpoint round-trip: HF checkpoints are off).")
            return
        print("\n--- checkpoint round-trip: new empty work dir, same checkpoint -> restore and skip every stage ---")
        sets2 = self._sets(f"{sd}/sample_data", f"{sd}/work_restored", f"{sd}/output_restored") + SMOKE_SETS
        self.run("all", sets2, log_name="smoke.log")
        same = all((sd / "output" / f).read_bytes() == (sd / "output_restored" / f).read_bytes()
                   for f in ("matching_results.tsv", "candidate_pairs.tsv"))
        cleanup = ("from ber.config import load_config; from ber.checkpoint import Checkpointer; "
                   f"Checkpointer(load_config({self.config!r}, {sets2 + ['checkpoint.enabled=true']!r})).reset()")
        r = subprocess.run([sys.executable, "-c", cleanup], cwd=self.pkg, env=self._env(), text=True, capture_output=True)
        print((r.stdout + r.stderr).strip()[-2000:])
        if r.returncode:
            print("note: could not delete the 'smoke-test' checkpoint folder; it is harmless and is replaced next time.")
        if not same:
            raise RuntimeError("checkpoint round-trip produced different outputs - do not start the full run")
        print("SMOKE TEST PASSED: full pipeline + HF checkpoint save/restore verified (test checkpoint deleted).")

    @staticmethod
    def gpu_paths_used(text: str) -> dict[str, bool]:
        """Which multi-GPU paths show up in a pipeline log."""
        return {"kNN search (block)": bool(re.search(r"knn: .*on cuda:0, cuda:1", text)),
                "XGBoost folds": bool(re.search(r"fold \d+: fit .* on cuda:1", text)),
                "cross-encoder (one half per GPU)": "workers running concurrently (GPU 0, 1)" in text}

    def _check_multi_gpu(self, text: str) -> None:
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            n = 0
        if n < 2:
            print(f"\nMulti-GPU check skipped: only {n} GPU visible.")
            return
        used = self.gpu_paths_used(text)
        print("\nMulti-GPU paths exercised by the rehearsal:")
        for name, ok in used.items():
            print(f"  {'OK     ' if ok else 'NOT USED'}  {name}")
        if not all(used.values()):
            print("  -> a path that shows NOT USED either fell back to one GPU (search the log for 'falling back' / "
                  "'failed') or was skipped. The run still works on one GPU; to make that explicit set "
                  "EXTRA_OVERRIDES = ['run.max_gpus=1'].")

    # ------------------------------------------------------------------ results
    def results(self) -> None:
        rep_path = self.work_dir / "models" / "oof_report.json"
        if rep_path.exists():  # an inference-only track does not re-train, so it may not have one
            rep = json.loads(rep_path.read_text())
            keys = ["macro_f05", "ceiling", "micro_precision", "micro_recall", "by_profile", "singleton_f05",
                    "non_singleton_f05", "cands_per_s1"]
            print("Out-of-fold validation on the full train set (best estimate before submitting):")
            print(json.dumps({k: rep.get(k) for k in keys}, indent=1))
        else:
            print("No out-of-fold report in this run (inference only: the models come from an earlier track).")
        reports = self.working / "reports"
        reports.mkdir(exist_ok=True)
        for f in ["oof_report.json", "thresholds.json", "stress_check.json", "prerank/floor.json",
                  "r1/oof_metrics.json", "r2/oof_metrics.json", "gate/oof_metrics.json",
                  "error_analysis.md", "error_analysis.json", "decision/report.json"]:
            src = self.work_dir / "models" / f
            if src.exists():
                shutil.copy(src, reports / f.replace("/", "_"))
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(self.output_dir / f, self.working / f)

    def package(self) -> None:
        env = (f"WORK_DIR={q(self.work_dir)} DATA_DIR={q(self.data_dir)} OUT_DIR={q(self.working)} "
               f"RESULTS_DIR={q(self.output_dir)}")
        self.sh(f"cd {q(self.pkg)} && {env} bash scripts/package_submission.sh {q(self.team)} --with-models")
        for p in sorted(self.working.iterdir()):
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
            print(f"{size / 1e6:10.1f} MB  {p}")
