"""Checkpoint the pipeline work dir to a private Hugging Face repo; restore it in a new session.

Kaggle stops a session after 12 h and wipes its scratch disk. With ``checkpoint.enabled``:
  * after every finished stage, the stage's outputs and its completion marker are pushed to a
    *private* HF dataset repo in one atomic commit (only files that changed since the last push);
  * a run that starts on an empty work dir first downloads the latest checkpoint, so finished
    stages are skipped and the pipeline continues where the previous session stopped.
The token is read from the environment (``HF_TOKEN``; on Kaggle from a Kaggle Secret) or, for
local runs, from a ``.env`` file -- only the token variable is read, and it is never logged.
Checkpointing never breaks a run: a failed upload is logged and retried after the next stage.
The one hard stop is a *public* repo, which would publish competition-derived files.
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import time
from pathlib import Path

from .utils import log

TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
MANIFEST = ".hf_sync.json"  # local record of what the remote holds: relpath -> [size, mtime_ns]
_ACTIVE: "Checkpointer | None" = None


def read_dotenv(start: Path, names: tuple[str, ...]) -> None:
    """Set ``names`` from the nearest ``.env`` in ``start`` or its parents (existing env vars win)."""
    for d in [start, *start.parents]:
        f = d / ".env"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.removeprefix("export ").partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key in names and val:
                os.environ.setdefault(key, val)
        return


def _token(names: tuple[str, ...]) -> str | None:
    for n in names:
        if os.environ.get(n, "").strip():
            return os.environ[n].strip()
    return None


class Checkpointer:
    def __init__(self, cfg):
        cc = cfg.get("checkpoint", {}) or {}
        self.enabled = bool(cc.get("enabled", False))
        self.work = Path(cfg.paths.work_dir)
        self.prefix = str(cc.get("prefix", "full")).strip("/")
        self.exclude = list(cc.get("exclude", []))
        self.repo = None
        if not self.enabled:
            return
        names = tuple(dict.fromkeys([str(cc.get("token_env", "HF_TOKEN")), *TOKEN_VARS]))
        read_dotenv(Path(cfg.paths.root), names)
        token = _token(names)
        if not token:
            log().warning("  checkpoint: enabled but no HF token in the environment (%s) -> checkpointing OFF",
                          names[0])
            self.enabled = False
            return
        try:
            from huggingface_hub import HfApi

            self.api = HfApi(token=token)
            self._token = token
            repo = cc.get("repo_id") or f"{self.api.whoami()['name']}/{cc.get('repo_name', 'amazon-ml-ber-work')}"
            self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            is_private = self.api.repo_info(repo, repo_type="dataset").private
        except Exception as e:  # bad token, no network, hub down: run without checkpoints
            log().warning("  checkpoint: Hugging Face unavailable (%s: %s) -> checkpointing OFF",
                          type(e).__name__, str(e).splitlines()[0][:200])
            self.enabled = False
            return
        if not is_private:
            raise RuntimeError(f"Hugging Face repo {repo} is PUBLIC; refusing to upload competition-derived "
                               "files. Make it private (repo Settings) or set checkpoint.repo_id.")
        self.repo = repo
        log().info("  checkpoint: private repo hf://datasets/%s, folder '%s'", repo, self.prefix)

    # ------------------------------------------------------------------ local state
    def _excluded(self, rel: str) -> bool:
        return rel == MANIFEST or any(fnmatch.fnmatch(rel, p) for p in self.exclude)

    def _scan(self) -> dict:
        out = {}
        if self.work.exists():
            for p in self.work.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(self.work).as_posix()
                    if not self._excluded(rel):
                        st = p.stat()
                        out[rel] = [st.st_size, st.st_mtime_ns]
        return out

    def _read_manifest(self) -> dict:
        f = self.work / MANIFEST
        return json.loads(f.read_text()) if f.exists() else {}

    def _write_manifest(self, state: dict) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / MANIFEST).write_text(json.dumps(state))

    def _remote_files(self) -> set[str]:
        pre = self.prefix + "/"
        return {f[len(pre):] for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.startswith(pre)}

    # ------------------------------------------------------------------ operations
    def reset(self) -> None:
        """Delete this run's checkpoint folder (fresh start). Other folders in the repo are untouched."""
        if not self.enabled:
            return
        if self._remote_files():
            self.api.delete_folder(self.prefix, repo_id=self.repo, repo_type="dataset",
                                   commit_message=f"fresh start: delete {self.prefix}/")
            try:  # drop the deleted files from history so they stop counting against storage
                self.api.super_squash_history(self.repo, repo_type="dataset")
            except Exception:
                pass
            log().info("  checkpoint: deleted previous checkpoint '%s' (fresh start)", self.prefix)
        (self.work / MANIFEST).unlink(missing_ok=True)

    def restore(self) -> None:
        """Empty work dir (no stage markers) -> download the latest checkpoint into it."""
        if not self.enabled:
            return
        markers = self.work / "_markers"
        if markers.exists() and any(markers.iterdir()):
            return  # this session already has local progress
        self._download_from_hf()

    def force_restore(self) -> None:
        """Re-download from HF even when local markers exist (crash recovery).

        This overwrites the local work dir with whatever is on HF, recovering
        any sub-stage progress that was pushed before the crash.
        """
        if not self.enabled:
            return
        log().info("  checkpoint: crash recovery — re-downloading latest HF state")
        self._download_from_hf()

    def _download_from_hf(self) -> None:
        """Download the remote checkpoint into the local work dir."""
        remote = self._remote_files()
        if not remote:
            log().info("  checkpoint: nothing saved yet -> starting from scratch")
            return
        from huggingface_hub import snapshot_download

        t0 = time.time()
        stage = self.work.parent / f".hf_restore_{self.work.name}"
        shutil.rmtree(stage, ignore_errors=True)
        snapshot_download(self.repo, repo_type="dataset", allow_patterns=[f"{self.prefix}/**"], local_dir=stage,
                          token=self._token, max_workers=8)
        n, size = 0, 0
        src = stage / self.prefix
        manifest = {}
        for p in sorted(src.rglob("*")):
            if p.is_file():
                rel = p.relative_to(src).as_posix()
                dst = self.work / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(p, dst)
                st = dst.stat()
                n, size = n + 1, size + st.st_size
                if not self._excluded(rel):
                    manifest[rel] = [st.st_size, st.st_mtime_ns]
        shutil.rmtree(stage, ignore_errors=True)
        # The manifest records what the REMOTE holds: only the files that came from it. Local-only files (outputs
        # not uploaded yet) must stay out of it, otherwise the next sync would think they are already on HF.
        self._write_manifest(manifest)
        markers = self.work / "_markers"
        done = sorted(p.name.split(".")[0] for p in markers.glob("*.done")) if markers.exists() else []
        log().info("  checkpoint: restored %d files (%.2f GB) in %.0fs; finished stages: %s", n, size / 1e9,
                   time.time() - t0, ", ".join(done) or "none")

    def sync(self, message: str) -> None:
        """Push files added/changed since the last push and delete removed ones, in one commit."""
        if not self.enabled:
            return
        try:
            from huggingface_hub import CommitOperationAdd, CommitOperationDelete

            now, old = self._scan(), self._read_manifest()
            changed = [r for r, st in now.items() if old.get(r) != st]
            gone = [r for r in old if r not in now]
            if gone:
                remote = self._remote_files()
                gone = [r for r in gone if r in remote]
            if not changed and not gone:
                return
            t0 = time.time()
            ops = [CommitOperationAdd(f"{self.prefix}/{r}", str(self.work / r)) for r in changed]
            ops += [CommitOperationDelete(f"{self.prefix}/{r}") for r in gone]
            self.api.create_commit(self.repo, operations=ops, commit_message=message, repo_type="dataset")
            self._write_manifest(now)
            log().info("  checkpoint: pushed %d files (%.2f GB), removed %d, in %.0fs", len(changed),
                       sum(now[r][0] for r in changed) / 1e9, len(gone), time.time() - t0)
        except Exception as e:
            log().warning("  checkpoint: upload failed (%s: %s); will retry after the next stage",
                          type(e).__name__, str(e).splitlines()[0][:200])


def activate(cfg) -> Checkpointer:
    global _ACTIVE
    _ACTIVE = Checkpointer(cfg)
    return _ACTIVE


def sync_now(message: str) -> None:
    """Mid-stage checkpoint (e.g. after each cross-encoder half); no-op when checkpointing is off."""
    if _ACTIVE is not None:
        _ACTIVE.sync(message)


def force_restore_now() -> None:
    """Re-download the latest HF checkpoint even when local markers already exist.

    Called after a crash (exit -9 / OOM) to recover whatever was last pushed.
    The local work dir may contain a mix of stale and fresh files, so we
    download the HF state on top, letting it overwrite anything already present.
    """
    if _ACTIVE is not None:
        _ACTIVE.force_restore()
