"""Exact top-k inner-product search with torch: every visible GPU (query chunks are shared out), else multithreaded CPU.

The index is split into row blocks that fit the memory budget; per-block top-k results are merged, so arbitrarily
large partitions work on any device. With several GPUs the index block is copied to each and the query chunks are
handed to whichever GPU is free, so the result is identical to a single-GPU run (exact top-k, independent chunks).
If a multi-GPU run fails for any reason (e.g. out of memory on one card) the call is repeated on one GPU.
"""
from __future__ import annotations

import queue
import threading

import numpy as np

from ..utils import gpu_cap, log, torch_device


def _devices(device: str, max_gpus: int = 0) -> list:
    import torch

    dev = torch_device(device)
    if dev.type != "cuda":
        return [dev]
    n = torch.cuda.device_count()
    for cap in (max_gpus, gpu_cap()):
        if cap and cap > 0:
            n = min(n, int(cap))
    return [torch.device(f"cuda:{i}") for i in range(max(1, n))]


def _topk_on(devs: list, index: np.ndarray, queries: np.ndarray, k: int, mem_gb: float, index_block_rows: int):
    import torch

    m, n = len(queries), len(index)
    k_eff = min(k, n)
    out_idx = np.full((m, k), -1, dtype=np.int64)
    out_sim = np.full((m, k), -np.inf, dtype=np.float32)
    cuda = devs[0].type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    nbytes = 2 if cuda else 4
    blocks = [(s, min(n, s + index_block_rows)) for s in range(0, n, index_block_rows)]
    for bs, be in blocks:
        rows = be - bs
        kb = min(k_eff, rows)
        q_chunk = int(max(1, min(65536, mem_gb * 1e9 / max(1, rows * nbytes * 2))))
        chunks = queue.SimpleQueue()
        for qs in range(0, m, q_chunk):
            chunks.put((qs, min(m, qs + q_chunk)))
        block = np.ascontiguousarray(index[bs:be])
        errors: list = []

        def work(dev):
            try:
                with torch.inference_mode():
                    I = torch.from_numpy(block).to(dev, dtype)
                    while True:
                        try:
                            qs, qe = chunks.get_nowait()
                        except queue.Empty:
                            break
                        q = torch.from_numpy(np.ascontiguousarray(queries[qs:qe])).to(dev, dtype)
                        v, ix = torch.topk(q @ I.T, kb, dim=1)
                        v = v.float().cpu().numpy()
                        ix = ix.cpu().numpy() + bs
                        if len(blocks) == 1:
                            out_sim[qs:qe, :kb], out_idx[qs:qe, :kb] = v, ix
                        else:  # merge with running best (each chunk is owned by exactly one worker)
                            cat_v = np.concatenate([out_sim[qs:qe], v], axis=1)
                            cat_i = np.concatenate([out_idx[qs:qe], ix], axis=1)
                            order = np.argsort(-cat_v, axis=1)[:, :k]
                            out_sim[qs:qe] = np.take_along_axis(cat_v, order, 1)
                            out_idx[qs:qe] = np.take_along_axis(cat_i, order, 1)
                    del I
            except BaseException as e:  # noqa: BLE001 - reported to the caller, which may retry on one device
                errors.append(e)

        if len(devs) == 1:
            work(devs[0])
        else:
            threads = [threading.Thread(target=work, args=(d,), daemon=True) for d in devs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if errors:
            raise errors[0]
        if cuda:
            for d in devs:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
    return out_idx, out_sim


def topk(index: np.ndarray, queries: np.ndarray, k: int, device: str = "auto", mem_gb: float = 4.0,
         index_block_rows: int = 8_000_000, max_gpus: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx int64 [m,k], sim float32 [m,k]); rows padded with -1 / -inf if index < k.

    ``max_gpus``: 0 = use every visible GPU, 1 = force a single GPU, N = at most N.
    """
    m, n = len(queries), len(index)
    if m == 0 or n == 0 or min(k, n) == 0:
        return (np.full((m, k), -1, dtype=np.int64), np.full((m, k), -np.inf, dtype=np.float32))
    devs = _devices(device, max_gpus)
    try:
        out = _topk_on(devs, index, queries, k, mem_gb, index_block_rows)
    except Exception as e:  # noqa: BLE001
        if len(devs) == 1:
            raise
        log().warning("   knn: multi-GPU run failed (%s: %s) -> repeating on one GPU", type(e).__name__,
                      str(e).splitlines()[0][:150])
        devs = devs[:1]
        out = _topk_on(devs, index, queries, k, mem_gb, index_block_rows)
    log().info("   knn: %d queries x %d index (k=%d) on %s", m, n, k, ", ".join(str(d) for d in devs))
    return out
