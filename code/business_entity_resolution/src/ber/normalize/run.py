"""Stage 0 driver: parse every record of a split into ``norm.parquet`` (parallel)."""
from __future__ import annotations

import shutil
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
    out_path, rec_path, offset, length = args
    part = (pl.scan_parquet(rec_path).slice(offset, length)
            .select(["uid", "name_raw", "addr_raw", "prof"]).collect())
    parse_records(part["uid"].to_list(), part["name_raw"].to_list(), part["addr_raw"].to_list(),
                  part["prof"].to_list(), _TABLES).write_parquet(out_path)
    return out_path


def normalize_split(cfg, split: str) -> None:
    """Workers read their own slice of records.parquet; parts are streamed into norm.parquet."""
    rec_path = work_dir(cfg, split, "records.parquet")
    n = pl.scan_parquet(rec_path).select(pl.len()).collect().item()
    tmp = work_dir(cfg, split, "_norm_parts")
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)
    tables_path = work_dir(cfg, None, "tables.pkl")
    tasks = [(str(tmp / f"part_{i:05d}.parquet"), str(rec_path), s, e - s)
             for i, (s, e) in enumerate(chunks(n, int(cfg.normalize.chunk_size)))]
    paths = pmap(_task, tasks, n_workers(cfg), initializer=_init,
                 initargs=(str(tables_path) if tables_path.exists() else None,), desc=f"normalize {split}")
    pl.scan_parquet(paths).sink_parquet(work_dir(cfg, split, "norm.parquet"))
    shutil.rmtree(tmp)
    log().info("  %s: normalized %d records", split, n)
