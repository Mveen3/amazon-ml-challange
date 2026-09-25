"""Stage 1a: multi-channel candidate union, per country partition, both directions.

Channels
  A1  address TF-IDF (RP) kNN      S1->records k_fwd, records->S1 k_rev
  N1  [name, beta*address] kNN     S1->records k_fwd, records->S1 k_rev
  A2/N2 exact keys (see keys.py)
  D1  optional dense bi-encoder hits (models/biencoder.py) if present on disk
Output: ``cands/union.parquet`` with per-channel ranks/sims, key mask, and RP cosines.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

from ..utils import ensure_dir, log, n_workers, work_dir
from . import keys as K
from .knn import topk
from .vectors import build, rowwise_cos

CHANNELS = {"a1f": 0, "a1r": 1, "n1f": 2, "n1r": 3, "d1f": 4, "d1r": 5}
NORM_COLS = ["uid", "name_core", "name_latin", "name_core_sorted", "name_skel", "is_domain", "domain_body",
             "addr_latin", "addr_tokens", "addr_key", "addr_missing", "nums", "pin"]


def safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name) or "unknown"


def load_partitioned(cfg, split: str) -> pl.DataFrame:
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src", "country", "prof"])
    norm = pl.read_parquet(work_dir(cfg, split, "norm.parquet"), columns=NORM_COLS)
    return rec.join(norm, on="uid").sort("uid")


def _hits(q_uids, i_uids, idx, sim, ch: str, min_score: float, s1_is_query: bool) -> pl.DataFrame:
    m, k = idx.shape
    rank = np.tile(np.arange(1, k + 1, dtype=np.int16), m)
    q = np.repeat(q_uids, k)
    flat_idx = idx.ravel()
    flat_sim = sim.ravel()
    ok = (flat_idx >= 0) & (flat_sim >= min_score)
    other = i_uids[np.clip(flat_idx[ok], 0, None)]
    s1, rec = (q[ok], other) if s1_is_query else (other, q[ok])
    return pl.DataFrame({"s1_uid": s1, "rec_uid": rec, "ch": np.full(ok.sum(), CHANNELS[ch], np.int8),
                         "rank": rank[ok], "sim": flat_sim[ok].astype(np.float32)})


def _knn_both(vec, s1_pos, rec_pos, uids, kf, kr, min_score, prefix, cfg) -> list[pl.DataFrame]:
    kc = cfg.blocking.knn
    out = []
    idx, sim = topk(vec[rec_pos], vec[s1_pos], kf, kc.device, float(kc.mem_gb))
    out.append(_hits(uids[s1_pos], uids[rec_pos], idx, sim, f"{prefix}f", min_score, True))
    idx, sim = topk(vec[s1_pos], vec[rec_pos], kr, kc.device, float(kc.mem_gb))
    out.append(_hits(uids[rec_pos], uids[s1_pos], idx, sim, f"{prefix}r", min_score, False))
    return out


def combined_vectors(name_rp: np.ndarray, addr_rp: np.ndarray, beta: float) -> np.ndarray:
    C = np.hstack([name_rp.astype(np.float32), beta * addr_rp.astype(np.float32)])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return C.astype(np.float16)


def block_split(cfg, split: str) -> pl.DataFrame:
    bc = cfg.blocking
    df = load_partitioned(cfg, split)
    workers = n_workers(cfg)
    frames = []
    for country in sorted(df["country"].unique().to_list()):
        part = df.filter(pl.col("country") == country)
        uids = part["uid"].to_numpy()
        src = part["src"].to_numpy()
        has_addr = ~part["addr_missing"].to_numpy()
        log().info("  [%s/%s] %d records (%d S1)", split, country, len(uids), int((src == 1).sum()))
        vdir = ensure_dir(work_dir(cfg, split, "vec", safe(country)))
        addr_docs = part["addr_latin"].str.replace_all(",", " ").to_list()
        name_docs = pl.when(pl.col("name_core").str.len_chars() > 0).then(pl.col("name_core")) \
            .otherwise(pl.col("name_latin"))
        name_docs = part.select(name_docs.alias("d"))["d"].to_list()
        A = build(addr_docs, vdir, "addr", cfg, workers)
        N = build(name_docs, vdir, "name", cfg, workers)
        C = combined_vectors(N, A, float(bc.name_addr_weight))
        np.save(vdir / "uids.npy", uids)
        np.save(vdir / "addr_rp.npy", A)
        np.save(vdir / "comb_rp.npy", C)

        s1_all, rec_all = np.where(src == 1)[0], np.where(src != 1)[0]
        s1_a, rec_a = np.where((src == 1) & has_addr)[0], np.where((src != 1) & has_addr)[0]
        hits = []
        hits += _knn_both(A, s1_a, rec_a, uids, int(bc.a1.k_fwd), int(bc.a1.k_rev), float(bc.a1.min_score), "a1", cfg)
        hits += _knn_both(C, s1_all, rec_all, uids, int(bc.n1.k_fwd), int(bc.n1.k_rev), float(bc.n1.min_score),
                          "n1", cfg)
        dense_path = work_dir(cfg, split, "cands", f"dense_hits_{safe(country)}.parquet")
        if bc.get("dense", {}).get("enabled", False) and dense_path.exists():
            hits.append(pl.read_parquet(dense_path))
        knn = pl.concat(hits)
        aggs = []
        for name, code in CHANNELS.items():
            aggs.append(pl.col("rank").filter(pl.col("ch") == code).min().alias(f"{name}_rank"))
        aggs += [pl.col("sim").filter(pl.col("ch").is_in([0, 1])).max().alias("a1_sim"),
                 pl.col("sim").filter(pl.col("ch").is_in([2, 3])).max().alias("n1_sim"),
                 pl.col("sim").filter(pl.col("ch").is_in([4, 5])).max().alias("d1_sim")]
        knn = knn.group_by(["s1_uid", "rec_uid"]).agg(aggs)

        kp = K.key_pairs(K.make_keys(part, cfg), cfg)
        bits = {i: 1 << i for i in range(len(K.KEY_TYPES))}
        kagg = (kp.with_columns(pl.col("kt").replace_strict(bits, return_dtype=pl.Int32).alias("bit"))
                .group_by(["s1_uid", "rec_uid"])
                .agg([pl.col("bit").unique().sum().alias("key_mask"), pl.col("block").min().alias("key_block"),
                      pl.col("kt").n_unique().cast(pl.Int8).alias("key_n")]))
        union = knn.join(kagg, on=["s1_uid", "rec_uid"], how="full", coalesce=True)
        pos_s = np.searchsorted(uids, union["s1_uid"].to_numpy())
        pos_r = np.searchsorted(uids, union["rec_uid"].to_numpy())
        a1 = rowwise_cos(A, A, pos_s, pos_r)
        a1[~(has_addr[pos_s] & has_addr[pos_r])] = np.nan
        union = union.with_columns([pl.Series("a1_cos", a1), pl.Series("n1_cos", rowwise_cos(C, C, pos_s, pos_r)),
                                    pl.lit(country).alias("country")])
        log().info("  [%s/%s] union pairs: %d (%.1f per S1)", split, country, union.height,
                   union.height / max(1, len(s1_all)))
        frames.append(union)
    union = pl.concat(frames, how="diagonal_relaxed").sort(["s1_uid", "rec_uid"])
    out = ensure_dir(work_dir(cfg, split, "cands"))
    union.write_parquet(out / "union.parquet")
    return union
