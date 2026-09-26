"""Stage 2 round-1 pair features (sharded, multiprocess).

Main process: context features over the final candidate graph, join both sides'
parsed fields, write input shards of ``shard_rows`` pairs (contiguous S1 ranges, per country).
Workers: vectorised rapidfuzz similarities + exact TF-IDF cosines + a per-pair
Python pass for token/IDF, component, number, domain, legal and diff-signature
features. Output: ``feats/r1/*.parquet``.
"""
from __future__ import annotations

import gc
import math
import pickle
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cdist, cpdist

from ..blocking.candidates import countries, safe
from ..blocking.prerank import PRE_FEATURES, attach_norm
from ..blocking.vectors import tfidf_rows
from ..normalize.profiles import ALL_LEGAL_TOKENS, LEGAL_FAMILY
from ..normalize.tables import Tables
from ..normalize.translit import skeleton
from ..utils import chunks, ensure_dir, load_json, log, n_workers, pmap, save_json, work_dir
from . import diffsig
from .admin import build_admin_vocab, density_match, density_ref, strip_pairs, unlabelled_countries
from .context import context_names, density_normalize, score_context

NORM_COLS = ["name_norm", "name_latin", "name_core", "name_core_sorted", "name_skel", "name_alts", "legal",
             "is_domain", "domain_body", "script", "dict_cov", "addr_latin", "addr_comps", "addr_tokens", "nums",
             "num_primary", "pin", "phone", "landmark", "bis", "addr_missing"]
# List-size counts: their scale follows the candidate density of a country and split (train 10.8 candidates
# per S1, test France 20.4), so they are divided by the country's median before any model sees them
# (features.density_norm). Ranks, gaps, shares and margins are scale-free and stay as they are.
DENSITY_COLS_R1 = ["s1_ncand", "rec_ncand", "pre_s1_n", "pre_rec_n"]
CARRY = ["s1_uid", "rec_uid", "fold", "label", "country", "prof"]
PASS = PRE_FEATURES + ["p_pre"] + context_names("pre")

VEC_FEATURES = ["nm_jw", "nm_lev", "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_norm_tset",
                "nm_latin_tset", "nm_skel_ratio", "nm_skel_tset", "ad_ratio", "ad_tset", "ad_tsort", "ad_partial",
                "lm_sim", "nm_tfidf", "ad_tfidf"]
LOOP_FEATURES = (
    ["nm_idf_jacc", "nm_cov_s", "nm_cov_r", "nm_me_s", "nm_me_r", "nm_first_eq", "nm_core_eq", "nm_sorted_eq",
     "nm_ntok_s", "nm_ntok_r", "nm_len_ratio", "nm_alt_best",
     "legal_both_missing", "legal_same", "legal_family", "legal_s_only", "legal_r_only", "legal_diff",
     "dom_is", "dom_concat", "dom_prefix2", "dom_initials", "dom_partial", "dom_ratio",
     "ad_idf_jacc", "ad_cov_s", "ad_cov_r", "ad_me_s", "ad_me_r",
     "cp_mean", "cp_min", "cp_n_match", "cp_s_missing", "cp_r_extra", "cp_n_s", "cp_n_r",
     "num_hit", "num_jacc", "num_eq", "num_cov", "num_absdiff", "num_edit", "num_n_s", "num_n_r",
     "pin_same", "pin_differ", "pin_s_only", "pin_r_only", "ph_same", "ph_differ", "bis_differ",
     "rec_dict_cov"]
    + diffsig.names("dn") + diffsig.names("da"))
R1_FEATURES = list(dict.fromkeys(PASS + VEC_FEATURES + LOOP_FEATURES))  # num_hit appears in PASS and LOOP

_CTX: dict = {}
NAN = float("nan")


# ----------------------------------------------------------------------------- idf tables
def build_token_idf(cfg, split: str) -> None:
    rare_df = int(cfg.features.rare_df)
    for country in countries(cfg, split):
        sub = (pl.scan_parquet(work_dir(cfg, split, "records.parquet")).select(["uid", "country"])
               .filter(pl.col("country") == country)
               .join(pl.scan_parquet(work_dir(cfg, split, "norm.parquet")).select(["uid", "name_core", "addr_tokens"]),
                     on="uid").collect())
        n = sub.height
        out = {}
        for field, expr in (("name", pl.col("name_core").str.split(" ")), ("addr", pl.col("addr_tokens"))):
            toks = (sub.select(["uid", expr.alias("t")]).explode("t").drop_nulls().unique()
                    .filter(~pl.col("t").str.contains(r"^\d+$") & (pl.col("t").str.len_chars() > 0))
                    .group_by("t").agg(pl.len().alias("df")).filter(pl.col("df") >= 2))
            idf = np.log((n + 1) / (toks["df"].to_numpy() + 1)) + 1
            out[field] = dict(zip(toks["t"].to_list(), idf.astype(np.float32).tolist()))
            out[f"{field}_default"] = float(math.log((n + 1) / 2) + 1)
            out[f"{field}_rare"] = float(math.log((n + 1) / (rare_df + 1)) + 1)
        with open(work_dir(cfg, split, f"tokidf_{safe(country)}.pkl"), "wb") as f:
            pickle.dump(out, f)
        del sub, out


# ----------------------------------------------------------------------------- inputs
def _labelled(pairs: pl.DataFrame, s1: pl.DataFrame, truth: pl.DataFrame | None) -> pl.DataFrame:
    pairs = pairs.drop([c for c in ("fold", "prof", "label") if c in pairs.columns]).join(s1, on="s1_uid", how="left")
    if truth is not None:
        return pairs.join(truth, on=["s1_uid", "rec_uid"], how="left").with_columns(pl.col("label").fill_null(0))
    return pairs.with_columns(pl.lit(None, dtype=pl.Int8).alias("label"))


def write_shards(cfg, country: str, pairs: pl.DataFrame, norm: pl.DataFrame | pl.LazyFrame, in_dir: Path,
                 out_dir: Path, admin: list[str] | None = None) -> list[tuple[str, str]]:
    """Join both sides' parsed fields in ``join_rows`` chunks and cut ``shard_rows`` input shards.

    ``pairs`` is sorted by (s1_uid, rec_uid), so every shard covers a contiguous S1 range. ``norm`` may be lazy:
    only the rows this country's pairs need are then loaded. Shards whose output already exists (an interrupted
    earlier attempt with the same plan) are not written again and are not returned as tasks.
    """
    fc = cfg.features
    join_rows, shard_rows = int(fc.get("join_rows", 4_000_000)), int(fc.get("shard_rows", 500_000))
    names = []  # shard names in order, from the same arithmetic as the loop below
    for s, e in chunks(pairs.height, join_rows):
        names += [None for _ in chunks(e - s, shard_rows)]
    names = [f"{safe(country)}_{i:05d}.parquet" for i in range(len(names))]
    todo = {n for n in names if not (out_dir / n).exists()}
    if not todo:
        return []
    need = pl.concat([pairs.select(pl.col("s1_uid").alias("uid")), pairs.select(pl.col("rec_uid").alias("uid"))]).unique()
    norm_c = (norm.lazy() if isinstance(norm, pl.DataFrame) else norm).join(need.lazy(), on="uid", how="semi").collect()
    del need
    tasks, idx = [], 0
    for s, e in chunks(pairs.height, join_rows):
        block = [names[idx + j] for j, _ in enumerate(chunks(e - s, shard_rows))]
        if not todo.intersection(block):
            idx += len(block)
            continue
        joined = attach_norm(pairs.slice(s, e - s), norm_c, NORM_COLS)
        joined = strip_pairs(joined, admin or [])  # no-op for profiles with training labels
        for s2, e2 in chunks(joined.height, shard_rows):
            name = names[idx]
            idx += 1
            if name in todo:
                joined.slice(s2, e2 - s2).write_parquet(in_dir / name)
                tasks.append((str(in_dir / name), str(out_dir / name)))
        del joined
    return tasks


# ----------------------------------------------------------------------------- worker
def _init(split_dir: str, tables_path: str, cfg_feat: dict) -> None:
    _CTX.clear()
    _CTX.update(split_dir=Path(split_dir), tables=Tables.load(Path(tables_path)), cfg=cfg_feat, idf={}, vec={})


def _country_ctx(country: str) -> tuple[dict, dict]:
    if country not in _CTX["idf"]:
        with open(_CTX["split_dir"] / f"tokidf_{safe(country)}.pkl", "rb") as f:
            _CTX["idf"][country] = pickle.load(f)
        vdir = _CTX["split_dir"] / "vec" / safe(country)
        _CTX["vec"][country] = {f: np.load(vdir / f"{f}_idf.npy") for f in ("name", "addr")}
    return _CTX["idf"][country], _CTX["vec"][country]


def _rowcos(a_docs, b_docs, idf, cfg_feat) -> np.ndarray:
    A = tfidf_rows(a_docs, idf, cfg_feat["ngram"], cfg_feat["hash_features"])
    B = tfidf_rows(b_docs, idf, cfg_feat["ngram"], cfg_feat["hash_features"])
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def _sim(a, b, scorer, scale=100.0) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32) / scale


def _tok_feats(s: list[str], r: list[str], idf: dict, dflt: float):
    if not s or not r:
        return (NAN,) * 5
    ss, rs = set(s), set(r)
    # sorted, never raw set order: set order depends on the per-process string-hash seed, and float sums in a
    # different order differ in the last bit -> features would not be byte-reproducible across processes
    sl, rl = sorted(ss), sorted(rs)
    w = {t: idf.get(t, dflt) for t in sorted(ss | rs)}
    jacc = sum(w[t] for t in sorted(ss & rs)) / max(1e-6, sum(w.values()))
    M = cdist(sl, rl, scorer=JaroWinkler.normalized_similarity, workers=1)
    sk_r, sk_s = {skeleton(t) for t in rl}, {skeleton(t) for t in sl}
    ms = [1.0 if (t in rs or skeleton(t) in sk_r or M[i].max() >= 0.9) else 0.0 for i, t in enumerate(sl)]
    mr = [1.0 if (t in ss or skeleton(t) in sk_s or M[:, j].max() >= 0.9) else 0.0 for j, t in enumerate(rl)]
    cov_s = sum(w[t] * m for t, m in zip(sl, ms)) / max(1e-6, sum(w[t] for t in sl))
    cov_r = sum(w[t] * m for t, m in zip(rl, mr)) / max(1e-6, sum(w[t] for t in rl))
    return jacc, cov_s, cov_r, float(M.max(axis=1).mean()), float(M.max(axis=0).mean())


def _legal(sl: str, rl: str):
    s = set(sl.split("+")) - {""}
    r = set(rl.split("+")) - {""}
    fs, fr = {LEGAL_FAMILY.get(x, x) for x in s}, {LEGAL_FAMILY.get(x, x) for x in r}
    same = bool(s) and s == r
    fam = bool(s) and bool(r) and not same and bool(fs & fr)
    return (float(not s and not r), float(same), float(fam), float(bool(s) and not r), float(bool(r) and not s),
            float(bool(s) and bool(r) and not (fs & fr)))


def _dom(s_core: list[str], is_dom: bool, body: str):
    if not is_dom or not body or not s_core:
        return (0.0, NAN, NAN, NAN, NAN, NAN)
    concat = "".join(s_core)
    init = "".join(t[0] for t in s_core if t)
    eq = float(body == concat or (len(concat) >= 6 and (body.startswith(concat) or concat.startswith(body))))
    pre2 = float(len(s_core) >= 2 and body.startswith(s_core[0] + s_core[1]))
    ini = float(len(init) >= 2 and body.startswith(init))
    return (1.0, eq, pre2, ini, fuzz.partial_ratio(body, concat) / 100.0, fuzz.ratio(body, concat) / 100.0)


def _comp(sc: list[str], rc: list[str]):
    if not sc or not rc:
        return (NAN, NAN, NAN, NAN, NAN, float(len(sc)), float(len(rc)))
    M = cdist(sc, rc, scorer=fuzz.token_set_ratio, workers=1) / 100.0
    bs, br = M.max(axis=1), M.max(axis=0)
    return (float(bs.mean()), float(bs.min()), float((bs >= 0.9).sum()), float((bs < 0.6).sum()),
            float((br < 0.6).sum()), float(len(sc)), float(len(rc)))


def _nums(s_nums: list[str], r_nums: list[str], prim: str):
    ss, rs = set(s_nums), set(r_nums)
    if not ss and not rs:
        return (NAN,) * 6 + (0.0, 0.0)
    hit = float(bool(prim) and prim in rs)
    jac = len(ss & rs) / len(ss | rs)
    eq = float(bool(ss) and ss == rs)
    cov = len(ss & rs) / len(ss) if ss else NAN
    if prim and r_nums:
        pv = int(prim[:12])
        absd = math.log1p(min(abs(pv - int(x[:12])) for x in r_nums))
        edit = float(min(Levenshtein.distance(prim, x) for x in r_nums))
    else:
        absd = edit = NAN
    return (hit, jac, eq, cov, absd, edit, float(len(ss)), float(len(rs)))


def compute(df: pl.DataFrame) -> pl.DataFrame:
    cf = _CTX["cfg"]
    tables: Tables = _CTX["tables"]
    country = df["country"][0]
    prof = df["prof"][0]
    tidf, vidf = _country_ctx(country)
    n = df.height
    col = {c: df[c].to_list() for c in df.columns if c.startswith(("s_", "r_"))}
    F: dict[str, np.ndarray] = {}
    sn, rn = col["s_name_core"], col["r_name_core"]
    F["nm_jw"] = _sim(sn, rn, JaroWinkler.normalized_similarity, 1.0)
    F["nm_lev"] = _sim(sn, rn, Levenshtein.normalized_similarity, 1.0)
    F["nm_ratio"] = _sim(sn, rn, fuzz.ratio)
    F["nm_tsort"] = _sim(sn, rn, fuzz.token_sort_ratio)
    F["nm_tset"] = _sim(sn, rn, fuzz.token_set_ratio)
    F["nm_partial"] = _sim(sn, rn, fuzz.partial_ratio)
    F["nm_norm_tset"] = _sim(col["s_name_norm"], col["r_name_norm"], fuzz.token_set_ratio)
    F["nm_latin_tset"] = _sim(col["s_name_latin"], col["r_name_latin"], fuzz.token_set_ratio)
    F["nm_skel_ratio"] = _sim(col["s_name_skel"], col["r_name_skel"], fuzz.ratio)
    F["nm_skel_tset"] = _sim(col["s_name_skel"], col["r_name_skel"], fuzz.token_set_ratio)
    sa, ra = col["s_addr_latin"], col["r_addr_latin"]
    F["ad_ratio"] = _sim(sa, ra, fuzz.ratio)
    F["ad_tset"] = _sim(sa, ra, fuzz.token_set_ratio)
    F["ad_tsort"] = _sim(sa, ra, fuzz.token_sort_ratio)
    F["ad_partial"] = _sim(sa, ra, fuzz.partial_ratio)
    lm = _sim(col["s_landmark"], col["r_landmark"], fuzz.token_set_ratio)
    lm[[not (a and b) for a, b in zip(col["s_landmark"], col["r_landmark"])]] = np.nan
    F["lm_sim"] = lm
    s_name_docs = [a or b for a, b in zip(sn, col["s_name_latin"])]
    r_name_docs = [a or b for a, b in zip(rn, col["r_name_latin"])]
    F["nm_tfidf"] = _rowcos(s_name_docs, r_name_docs, vidf["name"], cf)
    adt = _rowcos([x.replace(",", " ") for x in sa], [x.replace(",", " ") for x in ra], vidf["addr"], cf)
    adt[np.array(col["s_addr_missing"]) | np.array(col["r_addr_missing"])] = np.nan
    F["ad_tfidf"] = adt

    L = np.full((n, len(LOOP_FEATURES)), np.nan, dtype=np.float32)
    n_affix = tables.name_affix.get(prof, set())
    a_affix = tables.addr_affix.get(prof, set())
    ni, nd, nr_ = tidf["name"], tidf["name_default"], tidf["name_rare"]
    ai, ad, ar_ = tidf["addr"], tidf["addr_default"], tidf["addr_rare"]
    empty: set = set()
    for i in range(n):
        s_core, r_core = sn[i].split(), rn[i].split()
        row = list(_tok_feats(s_core, r_core, ni, nd))
        row += [float(bool(s_core) and bool(r_core) and s_core[0] == r_core[0]), float(sn[i] == rn[i]),
                float(col["s_name_core_sorted"][i] == col["r_name_core_sorted"][i]),
                float(len(s_core)), float(len(r_core)),
                min(len(sn[i]), len(rn[i])) / max(1, len(sn[i]), len(rn[i]))]
        alts = [x for x in col["r_name_alts"][i].split(" || ") if x] or [rn[i]]
        row.append(max(fuzz.token_set_ratio(sn[i], x) for x in alts) / 100.0)
        row += list(_legal(col["s_legal"][i], col["r_legal"][i]))
        row += list(_dom(s_core, col["r_is_domain"][i], col["r_domain_body"][i]))
        row += list(_tok_feats(col["s_addr_tokens"][i] or [], col["r_addr_tokens"][i] or [], ai, ad))
        row += list(_comp(col["s_addr_comps"][i] or [], col["r_addr_comps"][i] or []))
        row += list(_nums(col["s_nums"][i] or [], col["r_nums"][i] or [], col["s_num_primary"][i]))
        sp, rp = col["s_pin"][i], col["r_pin"][i]
        row += [float(bool(sp) and sp == rp), float(bool(sp) and bool(rp) and sp != rp),
                float(bool(sp) and not rp), float(bool(rp) and not sp)]
        sph, rph = col["s_phone"][i], col["r_phone"][i]
        both = bool(sph) and bool(rph)
        row += [float(both and sph[-7:] == rph[-7:]), float(both and sph[-7:] != rph[-7:]),
                float(col["s_bis"][i] != col["r_bis"][i])]
        row.append(float(col["r_dict_cov"][i]))
        dn = diffsig.diff_counts(s_core, r_core, ni, nd, nr_, n_affix, ALL_LEGAL_TOKENS)
        da = diffsig.diff_counts(col["s_addr_tokens"][i] or [], col["r_addr_tokens"][i] or [], ai, ad, ar_,
                                 a_affix, empty)
        L[i, :len(row)] = row
        L[i, len(row):len(row) + diffsig.N_OUT] = dn
        L[i, len(row) + diffsig.N_OUT:] = da
    out = df.select([c for c in CARRY + PASS if c in df.columns])
    out = out.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in F.items()])
    return out.with_columns([pl.Series(name, L[:, j]) for j, name in enumerate(LOOP_FEATURES)])


def _task(args) -> str:
    in_path, out_path = args
    tmp = Path(out_path).with_suffix(".tmp")
    compute(pl.read_parquet(in_path)).write_parquet(tmp)
    tmp.rename(out_path)  # atomic: a killed worker never leaves a half-written shard that looks finished
    Path(in_path).unlink()
    return out_path


# ----------------------------------------------------------------------------- driver
def run_features(cfg, split: str) -> None:
    """Round-1 features, sharded. Resumable: with the same plan (pair counts, shard sizes), shards finished by an
    interrupted earlier attempt are kept, and finished shards are checkpointed every ``features.sync_every``."""
    from ..checkpoint import sync_now

    build_token_idf(cfg, split)
    admin = build_admin_vocab(cfg, split)  # {} unless a country has no training labels (e.g. France)
    unlabelled, ref = unlabelled_countries(cfg, split), density_ref(cfg)
    fc = cfg.features
    in_dir, out_dir = work_dir(cfg, split, "feats", "r1_in"), work_dir(cfg, split, "feats", "r1")
    final = work_dir(cfg, split, "cands", "final.parquet")
    plan = {"join_rows": int(fc.get("join_rows", 4_000_000)), "shard_rows": int(fc.get("shard_rows", 500_000)),
            "pairs": {c: pl.scan_parquet(final).filter(pl.col("country") == c).select(pl.len()).collect().item()
                      for c in countries(cfg, split)},
            "final_bytes": int(final.stat().st_size),  # size, not mtime: a checkpoint restore changes mtimes
            "admin": admin, "density_norm": bool(fc.get("density_norm", False)),
            "density_match": sorted(unlabelled) if ref else []}
    plan_path = work_dir(cfg, split, "feats", "r1_plan.json")
    if plan_path.exists() and out_dir.exists() and load_json(plan_path) == plan:
        log().info("  %s: resuming round-1 features (%d shards already done)", split, len(list(out_dir.glob("*.parquet"))))
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
        save_json(plan, plan_path)
    shutil.rmtree(in_dir, ignore_errors=True)  # inputs are always rebuilt (only for shards still to do)
    ensure_dir(in_dir)
    ensure_dir(out_dir)
    s1 = pl.read_parquet(work_dir(cfg, split, "s1.parquet"), columns=["s1_uid", "prof", "fold"])
    truth = None
    if split == "train":
        truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet")).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    norm = pl.scan_parquet(work_dir(cfg, split, "norm.parquet")).select(["uid"] + NORM_COLS)  # lazy: per country
    tasks = []
    for country in countries(cfg, split):  # pairs never cross countries -> context per country is exact
        if plan["pairs"].get(country, 0) == 0:
            continue
        pairs = pl.scan_parquet(final).filter(pl.col("country") == country).collect().sort(["s1_uid", "rec_uid"])
        pairs = _labelled(score_context(pairs, "p_pre", "pre"), s1, truth)
        if bool(fc.get("density_norm", False)):
            pairs = density_normalize(pairs, DENSITY_COLS_R1)
        elif country in admin or country in unlabelled:  # a profile without labels: counts on the train scale
            pairs = density_match(pairs, DENSITY_COLS_R1, ref)
        # one dtype for every country (a rescaled count is fractional): shards of all countries read together
        pairs = pairs.with_columns([pl.col(c).cast(pl.Float32) for c in DENSITY_COLS_R1 if c in pairs.columns])
        tasks += write_shards(cfg, country, pairs, norm, in_dir, out_dir, admin.get(country))
        del pairs
        gc.collect()
    del norm, s1, truth
    gc.collect()
    cf = {"ngram": list(cfg.blocking.ngram), "hash_features": int(cfg.blocking.hash_features)}
    step = max(1, int(fc.get("sync_every", 20)))
    for b in range(0, len(tasks), step):  # batches: finished shards reach the checkpoint as the stage goes
        pmap(_task, tasks[b:b + step], n_workers(cfg), initializer=_init,
             initargs=(str(work_dir(cfg, split)), str(work_dir(cfg, None, "tables.pkl")), cf),
             desc=f"r1 feats {split} [{b + 1}-{min(len(tasks), b + step)} of {len(tasks)} to do]")
        if b + step < len(tasks):
            sync_now(f"features {split}: {b + step}/{len(tasks)} shards")
    shutil.rmtree(in_dir, ignore_errors=True)
    log().info("  %s: %d feature shards (%d computed now)", split, len(list(out_dir.glob("*.parquet"))), len(tasks))


def load_r1(cfg, split: str, columns=None) -> pl.DataFrame:
    df = pl.read_parquet(str(work_dir(cfg, split, "feats", "r1")) + "/*.parquet", columns=columns)
    return df.sort(["s1_uid", "rec_uid"]) if {"s1_uid", "rec_uid"} <= set(df.columns) else df
