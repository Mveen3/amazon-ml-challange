"""Multilingual cross-encoder on the uncertain band (2-fold cross-fitted).

Model h is trained on S1 entities whose fold is in ``halves[h]``; train pairs are
scored by the model that did *not* see their entity (OOF), test pairs by the mean
of both. Only pairs in the round-1 uncertainty band (plus each S1's top-k) are
scored; others get NaN and ``ce_scored = 0``. Strictly additive: the pipeline
runs without it (``ce.enabled: false`` or no GPU).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..features.pair import load_r1
from ..utils import ensure_dir, log, torch_device, work_dir

BUCKET_COLS = ["nm_core_eq", "nm_tset", "ad_tset", "num_hit", "legal_diff", "legal_s_only", "legal_r_only"]


def _texts(cfg, split: str, df: pl.DataFrame) -> tuple[list[str], list[str]]:
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "name_raw", "addr_raw", "src"])
    txt = rec.with_columns((pl.col("name_raw") + " | " + pl.col("addr_raw")).alias("t")).select(["uid", "t", "src"])
    d = (df.select(["s1_uid", "rec_uid"])
         .join(txt.rename({"uid": "s1_uid", "t": "a"}).drop("src"), on="s1_uid", how="left", maintain_order="left")
         .join(txt.rename({"uid": "rec_uid", "t": "b"}), on="rec_uid", how="left", maintain_order="left"))
    b = [f"S{s}: {t}" for s, t in zip(d["src"].to_list(), d["b"].to_list())]
    return [f"S1: {t}" for t in d["a"].to_list()], b


def band(cfg, split: str) -> pl.DataFrame:
    cc = cfg.ce
    p1 = pl.read_parquet(work_dir(cfg, split, "preds", "r1.parquet"))
    p1 = p1.with_columns(pl.col("p1").rank("ordinal", descending=True).over("s1_uid").alias("_r"))
    lo, hi = cc.band
    return p1.filter(((pl.col("p1") > float(lo)) & (pl.col("p1") < float(hi))) | (pl.col("_r") <= int(cc.top_rank))).drop("_r")


def _training_pairs(cfg, half: int) -> pl.DataFrame:
    cc = cfg.ce
    halves = [list(h) for h in cfg.folds.neural_halves]
    df = load_r1(cfg, "train", columns=["s1_uid", "rec_uid", "fold", "label", "p_pre"] + BUCKET_COLS)
    df = df.filter(pl.col("fold").is_in(halves[half]))
    rng = np.random.default_rng(int(cfg.run.seed) + half)
    pos = df.filter(pl.col("label") == 1)
    if pos.height > int(cc.max_pos):
        pos = pos.sample(int(cc.max_pos), seed=int(cfg.run.seed))
    neg = df.filter(pl.col("label") == 0)
    buckets = [
        neg.filter((pl.col("nm_core_eq") == 1) | (pl.col("nm_tset") >= 0.95)),          # same name, other place
        neg.filter((pl.col("ad_tset") >= 0.9) & (pl.col("nm_tset") < 0.7)),            # same address, other business
        neg.filter((pl.col("num_hit") == 0) & (pl.col("ad_tset") >= 0.8)),             # sibling house number
        neg.filter((pl.col("legal_diff") == 1) | (pl.col("legal_s_only") == 1) | (pl.col("legal_r_only") == 1)),
        neg.filter(pl.col("p_pre") >= 0.2),                                             # generally confusable
        neg,                                                                            # anything
    ]
    quota = int(pos.height * float(cc.neg_ratio) / len(buckets))
    picked = [b.sample(min(quota, b.height), seed=int(rng.integers(1 << 30))) for b in buckets if b.height]
    negs = pl.concat(picked).unique(["s1_uid", "rec_uid"]) if picked else neg.head(0)
    out = pl.concat([pos, negs]).select(["s1_uid", "rec_uid", "label"])
    return out.sample(fraction=1.0, shuffle=True, seed=int(cfg.run.seed))


def _load_model(name_or_path: str, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return tok, model


def _amp_dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def train_half(cfg, half: int) -> None:
    import torch

    cc = cfg.ce
    device = torch_device(cc.device)
    pairs = _training_pairs(cfg, half)
    a, b = _texts(cfg, "train", pairs)
    y = pairs["label"].to_numpy().astype(np.float32)
    log().info("  CE half %d: %d training pairs (%d pos)", half, len(y), int(y.sum()))
    tok, model = _load_model(cc.model_name, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cc.lr), weight_decay=0.01)
    bs = int(cc.batch_size)
    steps = int(cc.epochs) * math.ceil(len(y) / bs)
    warm = max(1, int(steps * float(cc.warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and _amp_dtype() == torch.float16)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0 = 0, time.time()
    rng = np.random.default_rng(int(cfg.run.seed) + 7 * half)
    for _ in range(int(cc.epochs)):
        order = rng.permutation(len(y))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                      max_length=int(cc.max_len), return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(), enabled=use_amp):
                logits = model(**enc).logits.view(-1)
                loss = loss_fn(logits.float(), torch.from_numpy(y[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log().info("   CE half %d step %d/%d loss %.4f (%.0f pairs/s)", half, step, steps, loss.item(),
                           step * bs / (time.time() - t0))
    out = ensure_dir(work_dir(cfg, None, "models", "ce", f"half{half}"))
    (model.module if hasattr(model, "module") else model).save_pretrained(out)
    tok.save_pretrained(out)


def _score(cfg, model_dir: Path, a: list[str], b: list[str]) -> np.ndarray:
    import torch

    cc = cfg.ce
    device = torch_device(cc.device)
    tok, model = _load_model(str(model_dir), device)
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    bs = int(cc.infer_batch_size)
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                      max_length=int(cc.max_len), return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(), enabled=device.type == "cuda"):
                out[idx] = model(**enc).logits.view(-1).float().cpu().numpy()
    return out


def infer(cfg, split: str, shard: int = 0, nshards: int = 1) -> None:
    df = band(cfg, split)
    df = df.with_row_index("_i").filter(pl.col("_i") % nshards == shard).drop("_i")
    a, b = _texts(cfg, split, df)
    mdir = work_dir(cfg, None, "models", "ce")
    halves = [list(h) for h in cfg.folds.neural_halves]
    if split == "train":
        fold = df["fold"].to_numpy() if "fold" in df.columns else \
            df.join(pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "fold"]), on="s1_uid",
                    how="left", maintain_order="left")["fold"].to_numpy()
        ce = np.full(len(a), np.nan, dtype=np.float32)
        for h in (0, 1):
            m = ~np.isin(fold, halves[h])  # pairs this model never saw
            if m.any():
                sel = np.where(m)[0]
                ce[sel] = _score(cfg, mdir / f"half{h}", [a[i] for i in sel], [b[i] for i in sel])
    else:
        ce = np.mean([_score(cfg, mdir / f"half{h}", a, b) for h in (0, 1)], axis=0)
    out = ensure_dir(work_dir(cfg, split, "preds"))
    df.select(["s1_uid", "rec_uid"]).with_columns(pl.Series("ce", ce)).write_parquet(out / f"ce_part{shard:03d}.parquet")
    log().info("  CE %s shard %d/%d: scored %d pairs", split, shard, nshards, len(ce))


def load_ce(cfg, split: str) -> pl.DataFrame | None:
    parts = sorted(work_dir(cfg, split, "preds").glob("ce_part*.parquet"))
    if not parts:
        return None
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["s1_uid", "rec_uid"])
