"""Mine normalisation tables from matched *training* pairs (no external data).

Three passes, each re-parsing a sample of truth pairs with the tables mined so far:
  A  Indic->Latin token dictionary (names) and abbreviation maps (names, addresses)
  B  address-component equivalences (state codes/names/native script, city aliases)
  C  affix tokens that noise tends to add to names / addresses (diff-signature typing)

Leak safety: a mapping is kept only if >= ``min_support`` *distinct* S1 entities
support it, so tables hold generic knowledge (``tn -> tamil nadu``), not entity names.
Because the same rule on 4/5 of the data would give the same entries, one global
mining run is equivalent to fold-internal mining for OOF validation.
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler

from ..normalize.address import parse_address
from ..normalize.names import name_tokens_for_mining, parse_name
from ..normalize.profiles import runtime_profile
from ..normalize.tables import EMPTY, Tables
from ..normalize.text import has_indic
from ..normalize.translit import romanize_token, skeleton
from ..utils import ensure_dir, log, n_workers, pmap, work_dir

_T: Tables = EMPTY


def _init(path: str | None) -> None:
    global _T
    _T = Tables.load(Path(path)) if path else EMPTY


def _is_abbrev(short: str, long: str) -> bool:
    """True abbreviations only (2-5 letters, >= 2 shorter): excludes doubled-letter typos like city/ccity."""
    if not (2 <= len(short) <= 5) or len(long) < len(short) + 2 or not short.isalpha() or not long.isalpha():
        return False
    if short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def _abbr_pairs(s_toks: list[str], r_toks: list[str]):
    s_set, r_set = set(s_toks), set(r_toks)
    s_only, r_only = s_set - r_set, r_set - s_set
    for short_side, long_side in ((r_only, s_only), (s_only, r_only)):
        for t in short_side:
            cands = [x for x in long_side if _is_abbrev(t, x)]
            if len(cands) == 1:
                yield t, cands[0]


# ----------------------------------------------------------------------------- pass A
def _pass_a(rows) -> tuple:
    tok_cnt, tok_tot = Counter(), Counter()
    nab, aab = Counter(), Counter()
    seen = set()
    for s1, sn, sa, rn, ra, p in rows:
        prof = runtime_profile(p, None)
        s_norm, _ = name_tokens_for_mining(sn, prof)
        r_norm, _ = name_tokens_for_mining(rn, prof)
        if any(has_indic(t) for t in r_norm):
            pairs = []
            if len(s_norm) == len(r_norm):
                pairs = [(r, s) for r, s in zip(r_norm, s_norm) if has_indic(r) and not has_indic(s)]
            else:
                lat = [t for t in s_norm if not has_indic(t)]
                for r in r_norm:
                    if not has_indic(r) or not lat:
                        continue
                    sk = skeleton(romanize_token(r))
                    best = max(lat, key=lambda x: JaroWinkler.normalized_similarity(sk, skeleton(x)))
                    if JaroWinkler.normalized_similarity(sk, skeleton(best)) >= 0.8:
                        pairs.append((r, best))
            for r, s in pairs:
                tok_tot[r] += 1
                if (s1, r, s) not in seen:
                    seen.add((s1, r, s))
                    tok_cnt[(r, s)] += 1
        elif not any(has_indic(t) for t in s_norm):
            for short, long in _abbr_pairs(s_norm, r_norm):
                if (s1, "n", short, long) not in seen:
                    seen.add((s1, "n", short, long))
                    nab[(p, short, long)] += 1
        if sa and ra:
            st = parse_address(sa, prof, EMPTY, use_comp_map=False)["addr_tokens"]
            rt = parse_address(ra, prof, EMPTY, use_comp_map=False)["addr_tokens"]
            for short, long in _abbr_pairs(st, rt):
                if (s1, "a", short, long) not in seen:
                    seen.add((s1, "a", short, long))
                    aab[(p, short, long)] += 1
    return tok_cnt, tok_tot, nab, aab


# ----------------------------------------------------------------------------- pass B
def _pass_b(rows) -> Counter:
    cnt = Counter()
    seen = set()
    for s1, sn, sa, rn, ra, p in rows:
        if not sa or not ra:
            continue
        prof = runtime_profile(p, _T)
        sc = parse_address(sa, prof, _T, use_comp_map=False)["addr_comps"]
        rc = parse_address(ra, prof, _T, use_comp_map=False)["addr_comps"]
        s_rem = [c for c in sc if c not in rc and not any(ch.isdigit() for ch in c)]
        r_rem = [c for c in rc if c not in sc and not any(ch.isdigit() for ch in c)]
        if len(s_rem) == 1 and len(r_rem) == 1 and len(r_rem[0].split()) <= 4 and len(s_rem[0].split()) <= 4:
            key = (p, r_rem[0], s_rem[0])
            if (s1, key) not in seen:
                seen.add((s1, key))
                cnt[key] += 1
    return cnt


# ----------------------------------------------------------------------------- pass C
def _pass_c(rows) -> tuple:
    n_extra, n_tot, a_extra, a_tot = Counter(), Counter(), Counter(), Counter()
    for s1, sn, sa, rn, ra, p in rows:
        prof = runtime_profile(p, _T)
        s = parse_name(sn, prof, _T)
        r = parse_name(rn, prof, _T)
        if r["script"] == 0 and not r["is_domain"]:
            s_toks = s["name_core"].split()
            s_sk = {skeleton(t) for t in s_toks}
            for t in set(r["name_core"].split()):
                n_tot[(p, t)] += 1
                if t not in s_toks and skeleton(t) not in s_sk:
                    n_extra[(p, t)] += 1
        if sa and ra:
            st = set(parse_address(sa, prof, _T)["addr_tokens"])
            for t in set(parse_address(ra, prof, _T)["addr_tokens"]):
                if t.isdigit():
                    continue
                a_tot[(p, t)] += 1
                if t not in st:
                    a_extra[(p, t)] += 1
    return n_extra, n_tot, a_extra, a_tot


def _merge(results, idx):
    out = Counter()
    for r in results:
        out.update(r[idx] if isinstance(r, tuple) else r)
    return out


def _dominant(cnt: Counter, min_support: int, min_share: float, key_len: int = 2) -> dict:
    """Keep ``src -> dst`` mappings whose support and share of ``src`` are high enough."""
    by_src = defaultdict(Counter)
    for key, c in cnt.items():
        by_src[key[:-1]][key[-1]] += c
    out = {}
    for src, dsts in by_src.items():
        dst, c = dsts.most_common(1)[0]
        if c >= min_support and c / sum(dsts.values()) >= min_share:
            out[src] = dst
    return out


def _canonical_groups(edges: dict, max_group: int = 25) -> dict:
    """Union-find over component equivalences; canonical = most-used S1-side form."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    target_use = Counter()
    for (r,), s in edges.items():
        parent[find(r)] = find(s)
        target_use[s] += 1
    groups = defaultdict(list)
    for x in list(parent):
        groups[find(x)].append(x)
    out = {}
    for members in groups.values():
        if len(members) > max_group:
            continue
        canon = max(members, key=lambda m: (target_use[m], -len(m)))
        for m in members:
            if m != canon:
                out[m] = canon
    return out


def mine(cfg) -> Tables:
    mc = cfg.mining
    rec = pl.read_parquet(work_dir(cfg, "train", "records.parquet"), columns=["uid", "name_raw", "addr_raw", "prof"])
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1_ids = truth["s1_uid"].unique()
    frac = min(1.0, int(mc.max_pairs) / max(1, truth.height))
    rng = np.random.default_rng(int(cfg.run.seed))
    keep = s1_ids.filter(pl.Series(rng.random(len(s1_ids)) < frac))
    pairs = truth.filter(pl.col("s1_uid").is_in(keep.implode())).sort("s1_uid")
    s = rec.rename({"uid": "s1_uid", "name_raw": "sn", "addr_raw": "sa"})
    r = rec.rename({"uid": "rec_uid", "name_raw": "rn", "addr_raw": "ra", "prof": "rp"})
    pairs = pairs.join(s, on="s1_uid").join(r, on="rec_uid").select(["s1_uid", "sn", "sa", "rn", "ra", "prof"])
    rows = pairs.rows()
    log().info("  mining on %d pairs", len(rows))
    size = max(1000, len(rows) // (n_workers(cfg) * 4) + 1)
    batches = [rows[i:i + size] for i in range(0, len(rows), size)]
    ms, sh = int(mc.min_support), float(mc.min_share)
    tmp = ensure_dir(work_dir(cfg, None, "_mining"))

    res = pmap(_pass_a, batches, n_workers(cfg), initializer=_init, initargs=(None,), desc="mine A")
    tok_cnt, tok_tot = _merge(res, 0), _merge(res, 1)
    token_map = {}
    for (rtok, stok), c in tok_cnt.items():
        if c >= ms and c / max(1, tok_tot[rtok]) >= sh and token_map.get(rtok, ("", 0))[1] < c:
            token_map[rtok] = (stok, c)
    token_map = {k: v[0] for k, v in token_map.items()}
    name_abbr, addr_abbr = defaultdict(dict), defaultdict(dict)
    for (p, short), long in _dominant(_merge(res, 2), ms, sh).items():
        name_abbr[p][short] = long
    for (p, short), long in _dominant(_merge(res, 3), ms, sh).items():
        addr_abbr[p][short] = long
    tables = Tables(token_map=token_map, name_abbr=dict(name_abbr), addr_abbr=dict(addr_abbr))
    tables.save(tmp / "a.pkl")
    log().info("  pass A: %d token maps, %d name abbr, %d addr abbr", len(token_map),
               sum(map(len, name_abbr.values())), sum(map(len, addr_abbr.values())))

    res = pmap(_pass_b, batches, n_workers(cfg), initializer=_init, initargs=(str(tmp / "a.pkl"),), desc="mine B")
    comp_cnt = _merge(res, None)
    comp_map = {}
    for prof_name in {k[0] for k in comp_cnt}:
        sub = Counter({k[1:]: c for k, c in comp_cnt.items() if k[0] == prof_name})
        edges = _dominant(sub, ms, sh)
        comp_map[prof_name] = _canonical_groups(edges)
    tables.comp_map = comp_map
    tables.save(tmp / "b.pkl")
    log().info("  pass B: %d component equivalences", sum(map(len, comp_map.values())))

    res = pmap(_pass_c, batches, n_workers(cfg), initializer=_init, initargs=(str(tmp / "b.pkl"),), desc="mine C")
    n_extra, n_tot, a_extra, a_tot = (_merge(res, i) for i in range(4))
    rate = float(mc.affix_rate)
    name_affix, addr_affix = defaultdict(set), defaultdict(set)
    for (p, t), c in n_extra.items():
        if c >= ms and c / n_tot[(p, t)] >= rate:
            name_affix[p].add(t)
    for (p, t), c in a_extra.items():
        if c >= ms and c / a_tot[(p, t)] >= rate:
            addr_affix[p].add(t)
    tables.name_affix, tables.addr_affix = dict(name_affix), dict(addr_affix)
    tables.save(work_dir(cfg, None, "tables.pkl"))
    log().info("  pass C: %d name affixes, %d addr affixes", sum(map(len, name_affix.values())),
               sum(map(len, addr_affix.values())))
    with open(tmp / "summary.pkl", "wb") as f:
        pickle.dump({"n_pairs": len(rows)}, f)
    return tables
