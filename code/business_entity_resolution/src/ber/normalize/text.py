"""Low-level text utilities shared by name and address parsing.

Python's ``re`` ``\\w`` does not match Indic vowel signs / viramas (Mn/Mc), so
tokenisation uses a translate table built from Unicode categories instead:
punctuation, symbols, separators and controls become spaces; letters, marks and
digits of every script survive.
"""
from __future__ import annotations

import re
import unicodedata

INDIC_RE = re.compile("[ऀ-෿]")
NULL_COMPS = {"null", "none", "nil", "n/a", "na", "nan", "unknown", "not available", "n.a", "n.a.", "-", "--", "0"}


def _build_tables():
    fold = {}
    for cp in list(range(0x00C0, 0x0250)) + list(range(0x1E00, 0x1F00)):
        ch = chr(cp)
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        if base and base != ch:
            fold[cp] = base
    for k, v in {"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "đ": "d",
                 "Đ": "D", "ł": "l", "Ł": "L", "þ": "th", "Þ": "TH", "ð": "d", "Ð": "D", "ı": "i", "ħ": "h"}.items():
        fold[ord(k)] = v
    for cp in range(0x0300, 0x0370):  # stray Latin combining diacritics
        fold[cp] = None
    sep = {}
    for cp in range(0, 0x3100):
        if unicodedata.category(chr(cp))[0] in "PSZC":
            sep[cp] = " "
    sep_keep = dict(sep)
    for ch in "-/":
        sep_keep.pop(ord(ch), None)
    return fold, sep, sep_keep


FOLD, SEP, SEP_KEEP = _build_tables()


def clean(s: str | None) -> str:
    """NFKC, fold Latin accents (Indic marks untouched), casefold."""
    if not s:
        return ""
    return unicodedata.normalize("NFKC", s).translate(FOLD).casefold()


def tokens(s: str, keep_num_seps: bool = False) -> list[str]:
    return s.translate(SEP_KEEP if keep_num_seps else SEP).split()


def has_indic(s: str) -> bool:
    return INDIC_RE.search(s) is not None


def collapse_single_letters(toks: list[str]) -> list[str]:
    """Merge runs of >=2 single ASCII letters: ``c i t`` -> ``cit``, ``s a s`` -> ``sas``."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isascii() and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def is_num(t: str) -> bool:
    return t.isascii() and t.isdigit()
