"""Dependency-free Indic -> Latin romaniser and a consonant-skeleton key.

All nine major Indic Unicode blocks share the ISCII-derived layout, so one
offset table (code point minus the block base) romanises Devanagari, Bengali,
Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam alike.

``skeleton`` collapses what transliteration cannot preserve (vowel quality,
voicing, aspiration, gemination) so that e.g. ``प्राइवेट`` and ``private`` both map
to ``prvt`` and Tamil ``டிரேடர்ஸ்`` and ``traders`` both map to ``trtrs``.
"""
from __future__ import annotations

import re

_BASES = {0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr", 0x0B00: "orya",
          0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda", 0x0D00: "mlym"}
_SCHWA_DROP = {"deva", "beng", "guru", "gujr"}  # word-final inherent vowel is silent

_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
           0x60: "ri", 0x61: "li", 0x72: "i", 0x73: "u"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
         0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_SIGNS = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri", 0x45: "e",
          0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4E: "e",
          0x4F: "aw", 0x62: "li", 0x63: "li"}
_MODS = {0x01: "n", 0x02: "n", 0x03: "h", 0x3D: "", 0x50: "om", 0x70: "n", 0x71: "", 0x55: "", 0x56: "", 0x57: "au"}
_SPECIAL = {0x09CE: "t", 0x0B83: "h", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l",
            0x0D7F: "k", 0x09F0: "r", 0x09F1: "v", 0x09DC: "r", 0x09DD: "rh", 0x09DF: "y"}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_LONG_V = re.compile(r"(aa|ii|uu)")


def romanize(text: str) -> str:
    """Romanise every Indic span in ``text``; other characters pass through."""
    res: list[str] = []
    pending = False  # a consonant was emitted and its vowel is still undecided
    script = None
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0DFF:
            if cp in _SPECIAL:
                if pending:
                    res.append("a")
                res.append(_SPECIAL[cp])
                pending = False
                continue
            base = cp & ~0x7F
            off = cp - base
            script = _BASES.get(base, script)
            if off in _CONS:
                if pending:
                    res.append("a")
                res.append(_CONS[off])
                pending = True
                continue
            if off == _NUKTA:
                continue
            if off == _VIRAMA:
                pending = False
                continue
            if off in _SIGNS:
                res.append(_SIGNS[off])
                pending = False
                continue
            if pending:
                res.append("a")
                pending = False
            if off in _VOWELS:
                res.append(_VOWELS[off])
            elif off in _MODS:
                res.append(_MODS[off])
            elif 0x66 <= off <= 0x6F:
                res.append(str(off - 0x66))
            continue
        if pending:
            if script not in _SCHWA_DROP:
                res.append("a")
            pending = False
        res.append(ch)
    if pending and script not in _SCHWA_DROP:
        res.append("a")
    return "".join(res)


def romanize_token(tok: str) -> str:
    """Romanise one token and collapse long vowels (``raam`` -> ``ram``)."""
    return _LONG_V.sub(lambda m: m.group(0)[0], romanize(tok))


_DIGRAPHS = (("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("gh", "g"), ("kh", "k"), ("th", "t"),
             ("dh", "d"), ("bh", "b"), ("jh", "j"), ("zh", "l"), ("ck", "k"), ("qu", "k"))
# Bengali has no /v/ (প্রাইভেট = "praibhet"), so b/v/w/p share one class.
_SKEL_MAP = str.maketrans({"c": "k", "q": "k", "g": "k", "d": "t", "b": "p", "v": "p", "w": "p", "z": "s", "x": "k"})
_REPEAT = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant skeleton of an ASCII token; digits are returned unchanged."""
    if not tok:
        return ""
    if tok.isdigit():
        return tok
    t = tok
    for a, b in _DIGRAPHS:
        if a in t:
            t = t.replace(a, b)
    t = t.translate(_SKEL_MAP)
    s = t[0] + "".join(c for c in t[1:] if c not in "aeiouyh")
    return _REPEAT.sub(r"\1", s)
