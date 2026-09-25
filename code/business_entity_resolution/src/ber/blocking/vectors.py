"""Hashed char n-gram TF-IDF, random-projected to dense float16 vectors.

Random projection preserves cosine in expectation (JL lemma) while keeping the
rare-n-gram emphasis of TF-IDF (unlike SVD), and dense vectors make exact
top-k search a plain GPU matmul. Exact sparse TF-IDF cosines for candidate
pairs are recomputed later from the saved IDF (``tfidf_rows``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from ..utils import chunks, ensure_dir, pmap


def _hv(ngram, n_features) -> HashingVectorizer:
    return HashingVectorizer(analyzer="char_wb", ngram_range=tuple(ngram), n_features=int(n_features),
                             alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)


def tfidf_rows(docs: list[str], idf: np.ndarray, ngram, n_features) -> sparse.csr_matrix:
    X = _hv(ngram, n_features).transform(docs).tocsr()
    X.data = 1.0 + np.log(X.data)
    X = X @ sparse.diags(idf.astype(np.float32))
    return normalize(X, norm="l2", copy=False).tocsr()


def _df_task(args):
    docs, ngram, nf = args
    X = _hv(ngram, nf).transform(docs)
    return np.bincount(X.indices, minlength=int(nf)).astype(np.int64)


def _proj_task(args):
    docs, idf_path, r_path, ngram, nf = args
    idf = np.load(idf_path)
    R = np.load(r_path, mmap_mode="r")
    Y = np.asarray(tfidf_rows(docs, idf, ngram, nf) @ R, dtype=np.float32)
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    Y /= np.maximum(norms, 1e-12)
    return Y.astype(np.float16)


def random_matrix(path: Path, n_features: int, dim: int, seed: int) -> Path:
    if not path.exists():
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((int(n_features), int(dim)), dtype=np.float32) / np.sqrt(dim)
        ensure_dir(path.parent)
        np.save(path, R)
    return path


def build(docs: list[str], out_dir: Path, field: str, cfg, workers: int) -> np.ndarray:
    """Fit IDF on ``docs`` (one partition), save ``{field}_idf.npy`` and return RP vectors."""
    bc = cfg.blocking
    ngram, nf, dim = list(bc.ngram), int(bc.hash_features), int(bc.rp_dim)
    size = int(bc.vector_chunk)
    tasks = [(docs[s:e], ngram, nf) for s, e in chunks(len(docs), size)]
    df = np.sum(pmap(_df_task, tasks, workers, desc=f"df {field}"), axis=0)
    n = len(docs)
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    ensure_dir(out_dir)
    idf_path = out_dir / f"{field}_idf.npy"
    np.save(idf_path, idf)
    r_path = random_matrix(Path(cfg.paths.work_dir) / "rp_matrix.npy", nf, dim, int(cfg.run.seed))
    tasks = [(docs[s:e], str(idf_path), str(r_path), ngram, nf) for s, e in chunks(len(docs), size)]
    parts = pmap(_proj_task, tasks, workers, desc=f"project {field}")
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, dim), np.float16)


def rowwise_cos(A: np.ndarray, B: np.ndarray, ia: np.ndarray, ib: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """cos(A[ia[k]], B[ib[k]]) for all k (vectors already L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s, e in chunks(len(ia), batch):
        a = A[ia[s:e]].astype(np.float32)
        b = B[ib[s:e]].astype(np.float32)
        out[s:e] = np.einsum("ij,ij->i", a, b)
    return out
