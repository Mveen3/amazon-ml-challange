"""Business-name parsing into several comparable views.

Output fields (all strings unless noted):
  name_norm         cleaned tokens, original script kept
  name_latin        romanised + abbreviation-expanded tokens (legal forms included)
  name_core         name_latin minus legal forms, leading honorifics, trailing store numbers
  name_core_sorted  sorted core tokens (word-order invariant)
  name_skel         consonant skeleton of each core token
  name_alts         " || "-joined cores of DBA / "|" alternatives (empty if none)
  legal             "+"-joined canonical legal-form classes
  is_domain (bool), domain_body   website / hashtag style names reduced to their body
  script (int8)     0 Latin, 1 all-Indic, 2 mixed
  dict_cov (float)  share of Indic tokens resolved by the mined dictionary
"""
from __future__ import annotations

import re

from .profiles import Profile
from .tables import EMPTY, Tables
from .text import clean, collapse_single_letters, has_indic, tokens
from .translit import romanize_token, skeleton

_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*?)\.(?:co\.in|com|in|net|org|fr|biz|info|io|co|us|uk)/?$")
_ALT_SPLIT = re.compile(r"\s*\|\s*|\s+(?:d/b/a|d\.b\.a\.?|dba|t/a|aka|a\.k\.a\.?|a/k/a|f/k/a|fka|"
                        r"doing business as|formerly known as|formerly)\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z]")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _fix_leet(t: str) -> str:
    """Undo digit-for-letter noise inside words: n0se -> nose, 5ervices -> services (3m stays)."""
    n_alpha = sum(c.isalpha() for c in t)
    if n_alpha == 0 or n_alpha == len(t) or (t[0].isdigit() and n_alpha < 3):
        return t
    return t.translate(_LEET)


def _domain_body(part: str) -> str | None:
    p = part.strip()
    m = _DOMAIN_RE.match(p)
    if m:
        return _NON_ALNUM.sub("", m.group(1))
    if p.startswith("www.") and " " not in p:
        return _NON_ALNUM.sub("", p[4:].split(".")[0])
    if p.startswith("#") and " " not in p and len(p) > 3:
        return _NON_ALNUM.sub("", p[1:])
    return None


def _extract_legal(toks: list[str], prof: Profile) -> tuple[list[str], list[str]]:
    core, legal, i, n = [], [], 0, len(toks)
    while i < n:
        for L in range(min(prof.legal_max, n - i), 0, -1):
            cls = prof.legal.get(tuple(toks[i:i + L]))
            if cls is not None:
                legal.append(cls)
                i += L
                break
        else:
            core.append(toks[i])
            i += 1
    return core, legal


def _parse_part(part: str, prof: Profile, tables: Tables) -> dict:
    dom = _domain_body(part)
    if dom:
        norm = [dom]
    else:
        norm = collapse_single_letters(tokens(part.replace("&", f" {prof.amp} ")))
    latin, n_indic, n_dict = [], 0, 0
    for t in norm:
        if has_indic(t):
            n_indic += 1
            mapped = tables.token_map.get(t)
            if mapped:
                n_dict += 1
                latin.extend(mapped.split())
            else:
                latin.append(romanize_token(t))
        else:
            latin.append(_fix_leet(t) if t.isascii() else t)
    abbr = prof.name_abbr
    latin = [x for t in latin for x in abbr.get(t, t).split()]
    core, legal = _extract_legal(latin, prof)
    while len(core) > 1 and core[0] in prof.honorific:
        core = core[1:]
    while len(core) > 2 and core[-1].isdigit() and len(core[-1]) >= 3:
        core = core[:-1]
    if not core:
        core = list(latin)
    return {"norm": norm, "latin": latin, "core": core, "legal": legal, "dom": dom,
            "n_indic": n_indic, "n_dict": n_dict, "n_tok": len(norm)}


def parse_name(raw: str, prof: Profile, tables: Tables = EMPTY) -> dict:
    s = clean(raw)
    parts = [p for p in _ALT_SPLIT.split(s) if p and p.strip()] or [""]
    alts = [_parse_part(p, prof, tables) for p in parts]
    main = alts[0]
    doms = [a["dom"] for a in alts if a["dom"]]
    n_indic = sum(a["n_indic"] for a in alts)
    n_tok = sum(a["n_tok"] for a in alts)
    core = main["core"]
    return {
        "name_norm": " ".join(main["norm"]),
        "name_latin": " ".join(main["latin"]),
        "name_core": " ".join(core),
        "name_core_sorted": " ".join(sorted(core)),
        "name_skel": " ".join(skeleton(t) for t in core),
        "name_alts": " || ".join(" ".join(a["core"]) for a in alts) if len(alts) > 1 else "",
        "legal": "+".join(sorted(set(main["legal"]))),
        "is_domain": bool(doms),
        "domain_body": doms[0] if doms else "",
        "script": 0 if n_indic == 0 else (1 if n_indic == n_tok else 2),
        "dict_cov": (sum(a["n_dict"] for a in alts) / n_indic) if n_indic else 1.0,
    }


def name_tokens_for_mining(raw: str, prof: Profile, tables: Tables = EMPTY) -> tuple[list[str], list[str]]:
    """(original-script tokens, Latin tokens) of the main alternative, for table mining."""
    s = clean(raw)
    parts = [p for p in _ALT_SPLIT.split(s) if p and p.strip()] or [""]
    a = _parse_part(parts[0], prof, tables)
    return a["norm"], a["latin"]
