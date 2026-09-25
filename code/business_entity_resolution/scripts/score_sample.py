"""Score smoke-test outputs against the hidden truth of the sample test split.

    python scripts/score_sample.py --out output_smoke --truth sample_data/test_truth.tsv \
        --s1 sample_data/test/test_source1.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.eval.metric import f05  # noqa: E402


def load(path: str) -> dict[str, set]:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return {r[0]: {x for x in r[1].split(",") if x} for r in df.iter_rows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_smoke")
    ap.add_argument("--truth", default="sample_data/test_truth.tsv")
    ap.add_argument("--s1", default="sample_data/test/test_source1.tsv")
    a = ap.parse_args()
    pred = load(f"{a.out}/matching_results.tsv")
    cand = load(f"{a.out}/candidate_pairs.tsv")
    truth = load(a.truth)
    s1 = pl.read_csv(a.s1, separator="\t", quote_char=None, infer_schema=False)
    country = dict(zip(s1["entity_id"], s1["country"]))
    by: dict[str, list] = {}
    for e, t in truth.items():
        by.setdefault(country[e], []).append((f05(pred.get(e, set()), t), f05(t & cand.get(e, set()), t)))
    rows = [x for v in by.values() for x in v]
    print(f"macro F0.5 {sum(a for a, _ in rows) / len(rows):.5f}  ceiling {sum(b for _, b in rows) / len(rows):.5f}  n={len(rows)}")
    for c, v in sorted(by.items()):
        print(f"  {c:8s} F0.5 {sum(a for a, _ in v) / len(v):.5f}  ceiling {sum(b for _, b in v) / len(v):.5f}  n={len(v)}")


if __name__ == "__main__":
    main()
