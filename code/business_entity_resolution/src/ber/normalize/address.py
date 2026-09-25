"""Address parsing: comma components, typed number/PIN/phone/landmark fields.

Noise shuffles *components* (comma-separated parts) but rarely the words inside
one, so the unit of comparison is the component. Numbers are pulled out into a
normalised list (leading zeros stripped, ordinals like 45th/45nd -> 45 kept in
the street text but excluded from house numbers), and phone numbers, PIN /
postcodes and landmarks go to their own fields so they never pollute the
house-number features.
"""
from __future__ import annotations

import re

from .profiles import Profile
from .tables import EMPTY, Tables
from .text import NULL_COMPS, clean, has_indic, tokens
from .translit import romanize_token

_COMP_SPLIT = re.compile(r"[,;|\n]+")
_NUM_SPLIT = re.compile(r"[-/]")
_ORD = re.compile(r"^(\d+)(st|nd|rd|th)$")
_RUNS = re.compile(r"\d+|\D+")

STREET_WORDS = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "place", "terrace", "trail", "highway",
    "parkway", "circle", "square", "way", "rue", "chemin", "impasse", "allee", "route", "quai", "cours",
    "passage", "marg", "nagar", "colony", "cross", "main", "layout", "sector", "block", "gali", "path",
    "bazar", "bazaar", "chowk", "pally", "para", "sarani", "salai", "veedhi", "peth", "wadi", "society",
    "apartment", "apartments", "building", "tower", "complex", "floor", "plot", "house", "flat", "unit",
    "suite", "box", "residence", "lotissement", "hameau", "lieu", "dit", "cite", "villa", "sentier", "pike",
    "run", "loop", "row", "walk", "point", "cove", "ridge", "crossing", "extension", "estate", "enclave",
    "vihar", "puram", "pura", "ganj", "bagh", "halli", "palya", "kottai", "pet", "wada",
}


def _tokenize_component(comp: str, prof: Profile, tables: Tables) -> list[list]:
    """Return items ``[token, kind]`` with kind 'w' word, 'n' number, 'o' ordinal."""
    items: list[list] = []
    for raw_tok in tokens(comp.replace("&", f" {prof.amp} "), keep_num_seps=True):
        if has_indic(raw_tok):
            mapped = tables.token_map.get(raw_tok)
            subs = mapped.split() if mapped else [romanize_token(raw_tok)]
        else:
            subs = [raw_tok]
        for tok in subs:
            for part in _NUM_SPLIT.split(tok):
                if not part:
                    continue
                if not any(ch.isdigit() for ch in part):
                    items.append([part, "w"])
                    continue
                m = _ORD.match(part)
                if m:
                    items.append([str(int(m.group(1))), "o"])
                    continue
                for run in _RUNS.findall(part):
                    if run.isdigit():
                        items.append([run.lstrip("0") or "0", "n"])
                    else:
                        items.append([run, "w"])
    # collapse runs of single letters ("c i t" -> "cit")
    out: list[list] = []
    run: list[str] = []
    for tok, kind in items:
        if kind == "w" and len(tok) == 1 and tok.isascii() and tok.isalpha():
            run.append(tok)
            continue
        if run:
            out.append(["".join(run), "w"])
            run = []
        out.append([tok, kind])
    if run:
        out.append(["".join(run), "w"])
    return out


def _take_phone(items: list[list], prof: Profile) -> tuple[list[list], list[str]]:
    keep, phone, i = [], [], 0
    while i < len(items):
        tok, kind = items[i]
        if kind == "n" and len(tok) >= 7:
            phone.append(tok)
        elif kind == "w" and tok in prof.phone and i + 1 < len(items) and items[i + 1][1] == "n":
            j = i + 1
            while j < len(items) and items[j][1] == "n" and len(items[j][0]) >= 3:
                phone.append(items[j][0])
                j += 1
            if j == i + 1:  # "ph 2" is probably "phase 2" -> keep
                keep.append(items[i])
            else:
                i = j
                continue
        else:
            keep.append(items[i])
        i += 1
    return keep, phone


def _take_landmark(items: list[list], prof: Profile) -> tuple[list[list], list[str]]:
    for i, (tok, kind) in enumerate(items):
        if kind != "w":
            continue
        nxt = items[i + 1][0] if i + 1 < len(items) else ""
        if tok in prof.landmark or (tok == "next" and nxt == "to") or (tok == "in" and nxt == "front"):
            start = i + (2 if tok in ("next", "in") else 1)
            return items[:i], [t for t, _ in items[start:]]
    return items, []


def _drop_prefixes(items: list[list], prof: Profile) -> list[list]:
    out, i, n = [], 0, len(items)
    while i < n:
        tok, kind = items[i]
        nxt = items[i + 1] if i + 1 < n else None
        if kind == "w" and tok in prof.addr_prefix_head and nxt is not None and nxt[0] == "no":
            i += 2
            continue
        if kind == "w" and tok in prof.addr_prefix and (nxt is None or nxt[1] in ("n", "w")) and n > 1:
            if tok in ("no", "hno", "dno", "hn") or (nxt is not None and nxt[1] == "n"):
                i += 1
                continue
        out.append(items[i])
        i += 1
    return out or items


def parse_address(raw: str, prof: Profile, tables: Tables = EMPTY, use_comp_map: bool = True) -> dict:
    s = clean(raw)
    comp_map = tables.comp_map.get(prof.name, {}) if use_comp_map else {}
    abbr = prof.addr_abbr
    comps, nums, phone, land = [], [], [], []
    pin, bis = "", False
    raw_comps = [c.strip() for c in _COMP_SPLIT.split(s)]
    raw_comps = [c for c in raw_comps if c and c.strip(" .-#") not in NULL_COMPS]
    for c in raw_comps:
        items = _tokenize_component(c, prof, tables)
        items = [it for it in items if not (it[1] == "w" and it[0] in NULL_COMPS)]
        items, ph = _take_phone(items, prof)
        phone.extend(ph)
        items, lm = _take_landmark(items, prof)
        land.extend(lm)
        items = _drop_prefixes(items, prof)
        # bis/ter house-number suffixes
        for k, (tok, kind) in enumerate(items):
            if kind == "w" and (tok in prof.number_suffix or
                                (prof.letter_bis and tok == "b" and k > 0 and items[k - 1][1] == "n")):
                bis = True
                items[k][0] = "bis" if tok == "b" else tok
        # abbreviation expansion
        expanded: list[list] = []
        for tok, kind in items:
            if kind == "w" and tok in abbr:
                expanded.extend([[t, "w"] for t in abbr[tok].split()])
            else:
                expanded.append([tok, kind])
        items = expanded
        # PIN / postcode
        if prof.postcode_len:
            has_street = any(k == "w" and t in STREET_WORDS for t, k in items)
            keep = []
            for tok, kind in items:
                if (kind == "n" and len(tok) == prof.postcode_len and not pin
                        and (prof.postcode_always or not has_street)):
                    pin = tok
                else:
                    keep.append([tok, kind])
            items = keep
        nums.extend(t for t, k in items if k == "n")
        if not items:
            continue
        text = " ".join(t for t, _ in items)
        comps.append(comp_map.get(text, text))
    toks = [t for c in comps for t in c.split()]
    return {
        "addr_latin": ", ".join(comps),
        "addr_comps": comps,
        "addr_tokens": toks,
        "addr_key": " ".join(sorted(set(toks))),
        "nums": nums,
        "num_primary": nums[0] if nums else "",
        "pin": pin,
        "phone": "".join(phone),
        "landmark": " ".join(land),
        "bis": bis,
        "addr_missing": not comps,
    }
