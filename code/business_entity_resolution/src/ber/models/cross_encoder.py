"""Multilingual cross-encoder on the uncertain band (2-fold cross-fitted).

Model h is trained on S1 entities whose fold is in ``halves[h]``; train pairs are
scored by the model that did *not* see their entity (OOF), test pairs by the mean
of both. Only pairs in the round-1 uncertainty band (plus each S1's top-k) are
scored; others get NaN and ``ce_scored = 0``. Strictly additive: the pipeline
runs without it (``ce.enabled: false`` or no GPU).
"""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..features.pair import load_r1
from ..utils import chunks, ensure_dir, gpu_cap, load_json, log, save_json, torch_device, work_dir

# Model loading in recent transformers prints two progress lines per weight tensor; that floods logs
# (and notebook output) for no information. Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BUCKET_COLS = ["nm_core_eq", "nm_tset", "ad_tset", "num_hit", "legal_diff", "legal_s_only", "legal_r_only"]


def text_table(cfg, split: str) -> pl.DataFrame:
    """uid -> "name | address" (raw text: the model sees exactly what the data provides) + source."""
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "name_raw", "addr_raw", "src"])
    return rec.with_columns((pl.col("name_raw") + " | " + pl.col("addr_raw")).alias("t")).select(["uid", "t", "src"])


def _texts(txt: pl.DataFrame, df: pl.DataFrame) -> tuple[list[str], list[str]]:
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
    """-> (tokenizer, net, hf_model).

    ``net(**enc)`` returns the logits as a plain tensor and runs on every visible GPU (DataParallel), so
    the multi-GPU gather never has to rebuild a transformers ``ModelOutput`` (whose internals change
    between transformers versions). ``hf_model`` is the underlying model, used for ``save_pretrained``.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:  # very old transformers
        pass
    tok = AutoTokenizer.from_pretrained(name_or_path)
    hf_model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1).to(device)

    class _Logits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, **enc):
            return self.m(**enc).logits

    net = _Logits(hf_model)
    n = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_cap() > 0:
        n = min(n, gpu_cap())
    if n > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(n)))
    return tok, net, hf_model


def _amp_dtype():
    """bf16 only on GPUs with native support (Ampere+, compute capability >= 8); fp16 otherwise.

    ``torch.cuda.is_bf16_supported()`` also returns True when bf16 is merely *emulated* (e.g. on a
    T4, capability 7.5), which runs without tensor cores and is several times slower than fp16.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def train_half(cfg, half: int) -> None:
    import torch

    cc = cfg.ce
    device = torch_device(cc.device)
    pairs = _training_pairs(cfg, half)
    a, b = _texts(text_table(cfg, "train"), pairs)
    y = pairs["label"].to_numpy().astype(np.float32)
    log().info("  CE half %d: %d training pairs (%d pos)", half, len(y), int(y.sum()))
    tok, net, hf_model = _load_model(cc.model_name, device)
    log().info("  CE half %d: %s, %s on %d GPU(s)", half, cc.model_name, _amp_dtype() if device.type == "cuda" else "fp32",
               torch.cuda.device_count() if device.type == "cuda" else 0)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cc.lr), weight_decay=0.01)
    bs = int(cc.batch_size)
    steps = int(cc.epochs) * math.ceil(len(y) / bs)
    warm = max(1, int(steps * float(cc.warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and _amp_dtype() == torch.float16)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    net.train()
    step, t0 = 0, time.time()
    rng = np.random.default_rng(int(cfg.run.seed) + 7 * half)
    for _ in range(int(cc.epochs)):
        order = rng.permutation(len(y))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                      max_length=int(cc.max_len), return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(), enabled=use_amp):
                logits = net(**enc).view(-1)
                loss = loss_fn(logits.float(), torch.from_numpy(y[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                log().info("   CE half %d step %d/%d loss %.4f (%.0f pairs/s)", half, step, steps, loss.item(),
                           step * bs / (time.time() - t0))
    # Save next to the final directory, then rename: the stage treats an existing half{h}/ as trained, so a
    # process killed while saving must not leave a half-written one. trained_at identifies this model in the
    # scoring plans (scores from another model are never reused).
    out = work_dir(cfg, None, "models", "ce", f"half{half}")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    hf_model.save_pretrained(ensure_dir(tmp))
    tok.save_pretrained(tmp)
    save_json({"trained_at": time.time()}, tmp / "ber_trained.json")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


class _Scorer:
    """A fine-tuned half model loaded once; scores text pairs in length-sorted batches."""

    def __init__(self, cfg, model_dir: Path):
        self.cc = cfg.ce
        self.device = torch_device(self.cc.device)
        self.tok, self.model, _ = _load_model(str(model_dir), self.device)
        self.model.eval()

    def __call__(self, a: list[str], b: list[str]) -> np.ndarray:
        import torch

        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
        out = np.empty(len(a), dtype=np.float32)
        bs = int(self.cc.infer_batch_size)
        with torch.inference_mode():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                enc = self.tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True,
                               max_length=int(self.cc.max_len), return_tensors="pt").to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=_amp_dtype(),
                                    enabled=self.device.type == "cuda"):
                    out[idx] = self.model(**enc).view(-1).float().cpu().numpy()
        return out


def infer(cfg, split: str, shard: int = 0, nshards: int = 1) -> None:
    """Score the uncertainty band in chunks (texts for one chunk at a time; each model loaded once)."""
    df = band(cfg, split)
    df = df.with_row_index("_i").filter(pl.col("_i") % nshards == shard).drop("_i")
    txt = text_table(cfg, split)
    mdir = work_dir(cfg, None, "models", "ce")
    halves = [list(h) for h in cfg.folds.neural_halves]
    scorers = [_Scorer(cfg, mdir / f"half{h}") for h in (0, 1)]
    ce = np.full(df.height, np.nan, dtype=np.float32)
    fold = df["fold"].to_numpy()
    for s, e in chunks(df.height, int(cfg.ce.get("infer_chunk", 1_000_000))):
        a, b = _texts(txt, df.slice(s, e - s))
        if split == "train":
            for h in (0, 1):
                m = ~np.isin(fold[s:e], halves[h])  # pairs this model never saw
                if m.any():
                    sel = np.where(m)[0]
                    ce[s + sel] = scorers[h]([a[i] for i in sel], [b[i] for i in sel])
        else:
            ce[s:e] = (scorers[0](a, b) + scorers[1](a, b)) / 2.0
        log().info("   CE %s: %d/%d pairs scored", split, e, df.height)
    out = ensure_dir(work_dir(cfg, split, "preds"))
    df.select(["s1_uid", "rec_uid"]).with_columns(pl.Series("ce", ce)).write_parquet(out / f"ce_part{shard:03d}.tmp")
    (out / f"ce_part{shard:03d}.tmp").rename(out / f"ce_part{shard:03d}.parquet")  # atomic: existence = scored
    log().info("  CE %s shard %d/%d: scored %d pairs", split, shard, nshards, len(ce))


def infer_half(cfg, split: str, half: int) -> Path:
    """Score this half's share of the band with this half's model only (used by the per-GPU workers).

    Train pairs are scored by the model that never saw their entity (so each pair by exactly one half); test pairs
    are scored by both halves and averaged later by ``merge_halves``.

    Resumable: scores are written in parts of ``ce.part_rows`` pairs under a plan (band size, part size, model id);
    a later attempt with the same plan keeps the finished parts and scores only the rest. ``ce.infer_chunk`` (texts
    held in memory at a time) is not part of the plan, so a lower-memory retry still resumes.
    """
    out = ensure_dir(work_dir(cfg, split, "preds")) / f"ce_half{half}.parquet"
    df = band(cfg, split)
    halves = [list(h) for h in cfg.folds.neural_halves]
    if split == "train":
        df = df.filter(pl.Series(~np.isin(df["fold"].to_numpy(), halves[half])))
    mdir = work_dir(cfg, None, "models", "ce") / f"half{half}"
    stamp = mdir / "ber_trained.json"
    part_rows = int(cfg.ce.get("part_rows", 250_000))
    plan = {"rows": df.height, "part_rows": part_rows, "model": load_json(stamp) if stamp.exists() else None}
    done = out.with_suffix(".json")
    if out.exists() and done.exists() and load_json(done) == plan:
        log().info("   CE half %d %s: already scored", half, split)
        return out
    pdir = out.with_name(f"ce_half{half}_parts")
    if (pdir / "plan.json").exists() and load_json(pdir / "plan.json") == plan:
        log().info("   CE half %d %s: resuming (%d parts already scored)", half, split, len(list(pdir.glob("*.parquet"))))
    else:
        shutil.rmtree(pdir, ignore_errors=True)
        save_json(plan, ensure_dir(pdir) / "plan.json")
    txt, scorer = None, None
    sub = max(1, min(part_rows, int(cfg.ce.get("infer_chunk", 1_000_000))))
    for i, (s, e) in enumerate(chunks(df.height, part_rows)):
        part = pdir / f"part_{i:05d}.parquet"
        if part.exists():
            continue
        if scorer is None:  # loaded only when something is left to score
            txt, scorer = text_table(cfg, split), _Scorer(cfg, mdir)
        ce = np.empty(e - s, dtype=np.float32)
        for s2, e2 in chunks(e - s, sub):
            a, b = _texts(txt, df.slice(s + s2, e2 - s2))
            ce[s2:e2] = scorer(a, b)
        tmp = part.with_suffix(".tmp")
        df.slice(s, e - s).select(["s1_uid", "rec_uid"]).with_columns(pl.Series("ce", ce)).write_parquet(tmp)
        tmp.rename(part)  # atomic: a killed worker never leaves a half-written part that looks finished
        log().info("   CE half %d %s: %d/%d pairs scored", half, split, e, df.height)
    parts = sorted(pdir.glob("part_*.parquet"))
    res = pl.read_parquet(parts) if parts else df.select(["s1_uid", "rec_uid"]).with_columns(pl.lit(0.0, pl.Float32).alias("ce"))
    tmp = out.with_suffix(".tmp")
    res.write_parquet(tmp)
    tmp.rename(out)
    save_json(plan, done)
    shutil.rmtree(pdir, ignore_errors=True)
    return out


def merge_halves(cfg, split: str) -> None:
    """Combine the per-half scores into ``ce_part000.parquet`` (train: disjoint union; test: mean of the two)."""
    pdir = work_dir(cfg, split, "preds")
    parts = [pl.read_parquet(pdir / f"ce_half{h}.parquet") for h in (0, 1)]
    both = pl.concat(parts)
    merged = both if split == "train" else both.group_by(["s1_uid", "rec_uid"]).agg(pl.col("ce").mean())
    expected = band(cfg, split).height
    if merged.height != expected:
        raise RuntimeError(f"CE {split}: {merged.height} scored pairs but the band has {expected}; check folds.neural_halves")
    merged.write_parquet(pdir / "ce_part000.tmp")
    (pdir / "ce_part000.tmp").rename(pdir / "ce_part000.parquet")  # its existence marks the split as scored
    for h in (0, 1):
        (pdir / f"ce_half{h}.parquet").unlink(missing_ok=True)
        (pdir / f"ce_half{h}.json").unlink(missing_ok=True)
    log().info("  CE %s: merged the two halves -> %d scored pairs", split, merged.height)


def load_ce(cfg, split: str) -> pl.DataFrame | None:
    parts = sorted(work_dir(cfg, split, "preds").glob("ce_part*.parquet"))
    if not parts:
        return None
    return pl.concat([pl.read_parquet(p) for p in parts]).unique(["s1_uid", "rec_uid"])
