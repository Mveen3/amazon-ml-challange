"""Optional dense channel D1: fine-tuned multilingual bi-encoder (docs §7.7).

Enable only if the ceiling analysis shows lexical channels miss true pairs.
Cross-fitted like the cross-encoder: on train, S1 entities of half h are searched
with the model trained on the *other* half; on test the two models' embeddings
are concatenated (cosine = mean of the two cosines).
Hits are written to ``cands/dense_hits_<country>.parquet`` and merged by the
blocking stage (run ``dense`` after ``normalize`` and before ``block``).
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..blocking.candidates import CHANNELS, _hits, safe
from ..blocking.knn import topk
from ..utils import ensure_dir, log, torch_device, work_dir


def _text(df: pl.DataFrame) -> list[str]:
    return ("query: " + df["name_raw"] + " | " + df["addr_raw"]).to_list()


def _encode(model, tok, texts: list[str], device, bs: int, max_len: int) -> np.ndarray:
    import torch

    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return out


def train(cfg) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    dc = cfg.blocking.dense
    device = torch_device(dc.device)
    rec = pl.read_parquet(work_dir(cfg, "train", "records.parquet"), columns=["uid", "name_raw", "addr_raw", "country"])
    truth = pl.read_parquet(work_dir(cfg, "train", "truth.parquet"))
    s1 = pl.read_parquet(work_dir(cfg, "train", "s1.parquet"), columns=["s1_uid", "fold"])
    halves = [list(h) for h in cfg.folds.neural_halves]
    for h in (0, 1):
        pairs = truth.join(s1, on="s1_uid").filter(pl.col("fold").is_in(halves[h]))
        if pairs.height > int(dc.max_pairs):
            pairs = pairs.sample(int(dc.max_pairs), seed=int(cfg.run.seed))
        a = pairs.join(rec.rename({"uid": "s1_uid"}), on="s1_uid", how="left", maintain_order="left")
        b = pairs.join(rec.rename({"uid": "rec_uid"}), on="rec_uid", how="left", maintain_order="left")
        # batches within one country -> harder in-batch negatives
        order = np.lexsort((np.random.default_rng(h).random(a.height), a["country"].to_numpy()))
        ta, tb = _text(a), _text(b)
        tok = AutoTokenizer.from_pretrained(dc.model_name)
        model = AutoModel.from_pretrained(dc.model_name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(dc.lr))
        bs, scale = int(dc.batch_size), float(dc.scale)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        np.random.default_rng(int(cfg.run.seed) + h).shuffle(batches)
        model.train()
        for step, idx in enumerate(batches):
            ea = tok([ta[i] for i in idx], padding=True, truncation=True, max_length=int(dc.max_len), return_tensors="pt").to(device)
            eb = tok([tb[i] for i in idx], padding=True, truncation=True, max_length=int(dc.max_len), return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                ha, hb = model(**ea).last_hidden_state, model(**eb).last_hidden_state
            ma, mb = ea["attention_mask"].unsqueeze(-1), eb["attention_mask"].unsqueeze(-1)
            va = torch.nn.functional.normalize(((ha * ma).sum(1) / ma.sum(1)).float(), dim=-1)
            vb = torch.nn.functional.normalize(((hb * mb).sum(1) / mb.sum(1)).float(), dim=-1)
            logits = va @ vb.T * scale
            lbl = torch.arange(len(idx), device=device)
            loss = (torch.nn.functional.cross_entropy(logits, lbl) + torch.nn.functional.cross_entropy(logits.T, lbl)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                log().info("   bi-encoder half %d step %d/%d loss %.4f", h, step, len(batches), loss.item())
        out = ensure_dir(work_dir(cfg, None, "models", "biencoder", f"half{h}"))
        model.save_pretrained(out)
        tok.save_pretrained(out)


def hits(cfg, split: str) -> None:
    from transformers import AutoModel, AutoTokenizer

    dc, bc = cfg.blocking.dense, cfg.blocking
    device = torch_device(dc.device)
    rec = pl.read_parquet(work_dir(cfg, split, "records.parquet"), columns=["uid", "src", "name_raw", "addr_raw", "country"])
    folds = pl.read_parquet(work_dir(cfg, split, "s1.parquet"), columns=["s1_uid", "fold"])
    rec = rec.join(folds.rename({"s1_uid": "uid"}), on="uid", how="left", maintain_order="left")
    halves = [list(h) for h in cfg.folds.neural_halves]
    models = []
    for h in (0, 1):
        p = work_dir(cfg, None, "models", "biencoder", f"half{h}")
        models.append((AutoTokenizer.from_pretrained(p), AutoModel.from_pretrained(p).to(device)))
    for country in rec["country"].unique().to_list():
        part = rec.filter(pl.col("country") == country)
        uids, src, fold = part["uid"].to_numpy(), part["src"].to_numpy(), part["fold"].fill_null(-1).to_numpy()
        texts = _text(part)
        embs = [_encode(m, t, texts, device, int(dc.infer_batch_size), int(dc.max_len)) for t, m in models]
        rec_pos = np.where(src != 1)[0]
        out = []
        if split == "train":
            for h in (0, 1):
                E = embs[h]  # model h never saw S1s outside halves[h]
                s1_pos = np.where((src == 1) & ~np.isin(fold, halves[h]))[0]
                idx, sim = topk(E[rec_pos], E[s1_pos], int(dc.k_fwd), bc.knn.device, float(bc.knn.mem_gb))
                out.append(_hits(uids[s1_pos], uids[rec_pos], idx, sim, "d1f", float(dc.min_score), True))
                idx, sim = topk(E[s1_pos], E[rec_pos], int(dc.k_rev), bc.knn.device, float(bc.knn.mem_gb))
                out.append(_hits(uids[rec_pos], uids[s1_pos], idx, sim, "d1r", float(dc.min_score), False))
        else:
            E = np.hstack(embs) / math.sqrt(2)
            s1_pos = np.where(src == 1)[0]
            idx, sim = topk(E[rec_pos], E[s1_pos], int(dc.k_fwd), bc.knn.device, float(bc.knn.mem_gb))
            out.append(_hits(uids[s1_pos], uids[rec_pos], idx, sim, "d1f", float(dc.min_score), True))
            idx, sim = topk(E[s1_pos], E[rec_pos], int(dc.k_rev), bc.knn.device, float(bc.knn.mem_gb))
            out.append(_hits(uids[rec_pos], uids[s1_pos], idx, sim, "d1r", float(dc.min_score), False))
        pl.concat(out).write_parquet(ensure_dir(work_dir(cfg, split, "cands")) / f"dense_hits_{safe(country)}.parquet")
        log().info("  dense hits %s/%s written", split, country)


assert CHANNELS["d1f"] == 4 and CHANNELS["d1r"] == 5
