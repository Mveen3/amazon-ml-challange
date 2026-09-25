"""Stage 0 driver: parse every record of a split into ``norm.parquet`` (parallel)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ..utils import chunks, ensure_dir, log, n_workers, pmap, work_dir
from .address import parse_address
from .names import parse_name
from .profiles import runtime_profile
from .tables import Tables

_TABLES: Tables | None = None

NORM_SCHEMA = {
    "uid": pl.Int64, "name_norm": pl.Utf8, "name_latin": pl.Utf8, "name_core": pl.Utf8,
    "name_core_sorted": pl.Utf8, "name_skel": pl.Utf8, "name_alts": pl.Utf8, "legal": pl.Utf8,
    "is_domain": pl.Boolean, "domain_body": pl.Utf8, "script": pl.Int8, "dict_cov": pl.Float32,
    "addr_latin": pl.Utf8, "addr_comps": pl.List(pl.Utf8), "addr_tokens": pl.List(pl.Utf8), "addr_key": pl.Utf8,
    "nums": pl.List(pl.Utf8), "num_primary": pl.Utf8, "pin": pl.Utf8, "phone": pl.Utf8, "landmark": pl.Utf8,
    "bis": pl.Boolean, "addr_missing": pl.Boolean,
}


def _init(tables_path: str | None) -> None:
    global _TABLES
    _TABLES = Tables.load(Path(tables_path) if tables_path else None)


def parse_records(uids, names, addrs, profs, tables: Tables) -> pl.DataFrame:
    cols: dict[str, list] = {k: [] for k in NORM_SCHEMA}
    for uid, n, a, p in zip(uids, names, addrs, profs):
        prof = runtime_profile(p, tables)
        row = {"uid": uid, **parse_name(n, prof, tables), **parse_address(a, prof, tables)}
        for k in NORM_SCHEMA:
            cols[k].append(row[k])
    return pl.DataFrame(cols, schema=NORM_SCHEMA)


def _task(args) -> str:
    out_path, uids, names, addrs, profs = args
    parse_records(uids, names, addrs, profs, _TABLES).write_parquet(out_path)
    return out_path


def normalize_split(cfg, split: str) -> None:
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"),
                          columns=["uid", "name_raw", "addr_raw", "prof"])
    tmp = ensure_dir(work_dir(cfg, split, "_norm_parts"))
    tables_path = work_dir(cfg, None, "tables.pkl")
    size = int(cfg.normalize.chunk_size)
    tasks = []
    for i, (s, e) in enumerate(chunks(rec.height, size)):
        part = rec.slice(s, e - s)
        tasks.append((str(tmp / f"part_{i:05d}.parquet"), part["uid"].to_list(), part["name_raw"].to_list(),
                      part["addr_raw"].to_list(), part["prof"].to_list()))
    del rec
    paths = pmap(_task, tasks, n_workers(cfg), initializer=_init,
                 initargs=(str(tables_path) if tables_path.exists() else None,), desc=f"normalize {split}")
    norm = pl.concat([pl.read_parquet(p) for p in paths])
    norm.write_parquet(work_dir(cfg, split, "norm.parquet"))
    for p in paths:
        Path(p).unlink()
    tmp.rmdir()
    log().info("  %s: normalized %d records", split, norm.height)
