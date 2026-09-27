"""Kaggle helpers (stdlib only): locate the competition files and pick a scratch disk.

The dataset is added to the notebook as a Kaggle Dataset (e.g. uploaded as ``amazon-ml.zip`` and
named ``amazon-ml``). Kaggle usually extracts zips on upload, but the folder nesting depends on how
the archive was built, and some archives stay packed. ``prepare`` therefore:
  1. searches ``/kaggle/input/<slug>`` (or all of ``/kaggle/input``) for the 7 required TSVs,
  2. extracts any .zip / .tar(.gz|.bz2|.xz) archives it finds if files are still missing,
  3. symlinks everything into ``<link_root>/{train,test}/`` -- the layout the pipeline expects.
The links live on scratch disk, not under /kaggle/working, so they are never saved as output.

    python scripts/kaggle_prepare.py --slug amazon-ml --link-root /kaggle/temp/dataset
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REQUIRED = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _walk(roots: list[Path]):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith(".")]
            for f in filenames:
                yield Path(dirpath) / f


def _find(roots: list[Path], names: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in _walk(roots):
        if p.name in names and p.name not in found and p.stat().st_size > 0:
            found[p.name] = p
    return found


def _extract_archives(roots: list[Path], dest: Path) -> list[Path]:
    done = []
    for p in _walk(roots):
        if p.name.lower().endswith(ARCHIVES):
            target = dest / p.name.split(".")[0]
            marker = target / ".extracted"
            if not marker.exists():
                print(f"  extracting {p} -> {target}")
                target.mkdir(parents=True, exist_ok=True)
                shutil.unpack_archive(str(p), str(target))
                marker.touch()
            done.append(target)
    return done


def pick_scratch(candidates=("/kaggle/temp", "/tmp")) -> Path:
    """Writable candidate with the most free space (Kaggle's scratch disk is much larger than /kaggle/working)."""
    best, best_free = None, -1
    for c in candidates:
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test"
            probe.touch()
            probe.unlink()
        except OSError:
            continue
        free = shutil.disk_usage(p).free
        if free > best_free:
            best, best_free = p, free
    if best is None:
        raise RuntimeError(f"no writable scratch directory among {candidates}")
    print(f"  scratch disk: {best} ({best_free / 1e9:.0f} GB free)")
    return best


def prepare(input_root: str = "/kaggle/input", slug: str = "amazon-ml", link_root: str = "/kaggle/temp/dataset",
            extract_root: str | None = None) -> Path:
    inp = Path(input_root)
    roots = [inp / slug] if (inp / slug).exists() else [inp]
    names = {n for v in REQUIRED.values() for n in v}
    found = _find(roots, names)
    missing = names - set(found)
    if missing:
        extract = Path(extract_root) if extract_root else Path(link_root).parent / "dataset_extracted"
        extracted = _extract_archives(roots, extract)
        if extracted:
            found.update(_find(extracted, missing))
    missing = names - set(found)
    if missing:
        listing = "\n".join(f"    {p}" for p in list(_walk(roots))[:40])
        raise FileNotFoundError(f"missing {sorted(missing)} under {roots}. Is the dataset added to the notebook "
                                f"(Add Input -> your dataset)? First files seen:\n{listing}")
    out = Path(link_root)
    for split, files in REQUIRED.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for name in files:
            link = d / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(found[name].resolve())
            print(f"  {split}/{name:<24} <- {found[name]}  ({found[name].stat().st_size / 1e6:,.0f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--slug", default="amazon-ml")
    ap.add_argument("--link-root", default="/kaggle/temp/dataset")
    a = ap.parse_args()
    prepare(a.input_root, a.slug, a.link_root)


if __name__ == "__main__":
    main()
