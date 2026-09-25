"""One cross-encoder half as its own process (pinned to one GPU by ``CUDA_VISIBLE_DEVICES``).

    python -m ber.models.ce_worker --config C --set k=v ... --op train --half 0
    python -m ber.models.ce_worker --config C --set k=v ... --op infer --half 1 --split test

Started by ``ber.models.ce_parallel``; not meant to be run by hand.
"""
from __future__ import annotations

import argparse


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--op", required=True, choices=["train", "infer"])
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)

    from ..config import load_config
    from ..utils import set_seed
    from . import cross_encoder as CE

    cfg = load_config(a.config, a.overrides)
    set_seed(int(cfg.run.seed))
    if a.op == "train":
        CE.train_half(cfg, a.half)
    else:
        CE.infer_half(cfg, a.split, a.half)


if __name__ == "__main__":
    main()
