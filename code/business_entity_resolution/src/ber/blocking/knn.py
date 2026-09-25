"""Exact top-k inner-product search with torch (GPU if present, else multithreaded CPU).

The index is split into row blocks that fit the memory budget; per-block top-k
results are merged, so arbitrarily large partitions work on any device.
"""
from __future__ import annotations

import numpy as np

from ..utils import log, torch_device


def topk(index: np.ndarray, queries: np.ndarray, k: int, device: str = "auto", mem_gb: float = 4.0,
         index_block_rows: int = 8_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx int64 [m,k], sim float32 [m,k]); rows padded with -1 / -inf if index < k."""
    import torch

    m, n = len(queries), len(index)
    k_eff = min(k, n)
    out_idx = np.full((m, k), -1, dtype=np.int64)
    out_sim = np.full((m, k), -np.inf, dtype=np.float32)
    if m == 0 or n == 0 or k_eff == 0:
        return out_idx, out_sim
    dev = torch_device(device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    nbytes = 2 if dtype == torch.float16 else 4
    blocks = [(s, min(n, s + index_block_rows)) for s in range(0, n, index_block_rows)]
    with torch.inference_mode():
        for bs, be in blocks:
            I = torch.from_numpy(np.ascontiguousarray(index[bs:be])).to(dev, dtype)
            rows = be - bs
            kb = min(k_eff, rows)
            q_chunk = int(max(1, min(65536, mem_gb * 1e9 / max(1, rows * nbytes * 2))))
            for qs in range(0, m, q_chunk):
                qe = min(m, qs + q_chunk)
                q = torch.from_numpy(np.ascontiguousarray(queries[qs:qe])).to(dev, dtype)
                v, ix = torch.topk(q @ I.T, kb, dim=1)
                v = v.float().cpu().numpy()
                ix = ix.cpu().numpy() + bs
                if len(blocks) == 1:
                    out_sim[qs:qe, :kb], out_idx[qs:qe, :kb] = v, ix
                else:  # merge with running best
                    cat_v = np.concatenate([out_sim[qs:qe], v], axis=1)
                    cat_i = np.concatenate([out_idx[qs:qe], ix], axis=1)
                    order = np.argsort(-cat_v, axis=1)[:, :k]
                    out_sim[qs:qe] = np.take_along_axis(cat_v, order, 1)
                    out_idx[qs:qe] = np.take_along_axis(cat_i, order, 1)
            del I
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    log().info("   knn: %d queries x %d index (k=%d) on %s", m, n, k, dev)
    return out_idx, out_sim
