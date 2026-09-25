""""What changed" (diff-signature) features.

Planted traps and noisy true matches can score the same on every generic
similarity; what separates them is *which kind of edit* turns the S1 field into
the record field. Tokens are aligned greedily and every edit is typed; counts per
type become features (and error-analysis buckets).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from rapidfuzz.distance import JaroWinkler

from ..normalize.translit import skeleton

TYPES = ["exact", "phon", "typo", "abbr", "numchg", "miss_legal", "miss_num", "miss_generic", "miss_rare",
         "extra_affix", "extra_legal", "extra_num", "extra_generic", "extra_rare"]
IDX = {t: i for i, t in enumerate(TYPES)}
N_OUT = len(TYPES) + 2  # + missing idf mass, extra idf mass


def _is_abbrev(short: str, long: str) -> bool:
    if len(short) >= len(long) or len(long) < 3 or short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def diff_counts(a: list[str], b: list[str], idf: dict, default_idf: float, rare_idf: float,
                affix: set, legal_vocab: set) -> np.ndarray:
    """a: S1 tokens (reference), b: record tokens."""
    out = np.zeros(N_OUT, dtype=np.float32)
    ca, cb = Counter(a), Counter(b)
    common = ca & cb
    out[IDX["exact"]] = sum(common.values())
    ra = list((ca - common).elements())
    rb = list((cb - common).elements())
    cand = []
    if ra and rb:
        sk_b = [skeleton(y) if not y.isdigit() else y for y in rb]
        for i, x in enumerate(ra):
            xd = x.isdigit()
            sx = x if xd else skeleton(x)
            for j, y in enumerate(rb):
                yd = y.isdigit()
                if xd and yd:
                    cand.append((0.5, i, j, IDX["numchg"]))
                elif xd or yd:
                    continue
                elif sx == sk_b[j]:
                    cand.append((0.95, i, j, IDX["phon"]))
                else:
                    jw = JaroWinkler.normalized_similarity(x, y)
                    if jw >= 0.88:
                        cand.append((jw, i, j, IDX["typo"]))
                    elif _is_abbrev(y, x) or _is_abbrev(x, y):
                        cand.append((0.7, i, j, IDX["abbr"]))
    cand.sort(reverse=True)
    ui, uj = set(), set()
    for _, i, j, t in cand:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        out[t] += 1
    miss_idf = extra_idf = 0.0
    for i, x in enumerate(ra):
        if i in ui:
            continue
        w = idf.get(x, default_idf)
        miss_idf += w
        if x in legal_vocab:
            out[IDX["miss_legal"]] += 1
        elif x.isdigit():
            out[IDX["miss_num"]] += 1
        elif w >= rare_idf:
            out[IDX["miss_rare"]] += 1
        else:
            out[IDX["miss_generic"]] += 1
    for j, y in enumerate(rb):
        if j in uj:
            continue
        w = idf.get(y, default_idf)
        extra_idf += w
        if y in affix:
            out[IDX["extra_affix"]] += 1
        elif y in legal_vocab:
            out[IDX["extra_legal"]] += 1
        elif y.isdigit():
            out[IDX["extra_num"]] += 1
        elif w >= rare_idf:
            out[IDX["extra_rare"]] += 1
        else:
            out[IDX["extra_generic"]] += 1
    out[-2], out[-1] = miss_idf, extra_idf
    return out


def names(prefix: str) -> list[str]:
    return [f"{prefix}_{t}" for t in TYPES] + [f"{prefix}_miss_idf", f"{prefix}_extra_idf"]
