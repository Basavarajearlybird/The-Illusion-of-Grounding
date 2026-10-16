"""Text embedding + nearest-neighbour search for the hard-negative miner.

Three backends, tried in order, all exposing the same two methods. The intent is
that the released build uses the strong encoder, while the pipeline stays
executable without a 1.3 GB model download:

* ``sentence-transformers`` with ``BAAI/bge-large-en-v1.5`` (the architecture's
  choice) or ``intfloat/e5-large-v2``.
* ``TF-IDF`` (scikit-learn) -- lexical rather than semantic, but it still
  surfaces on-topic documents, which is the property the miner needs.
* hashed bag-of-words -- last resort, no optional deps at all.

The index side prefers FAISS and falls back to exact numpy search, which is fine
at 15K documents (a 15K x 15K similarity pass is seconds).
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Text -> unit-norm dense vectors."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", prefer: str = "auto"):
        self.model_name = model_name
        self.backend = "none"
        self._model = None
        self._vectorizer = None

        if prefer in ("auto", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                return
            except Exception:
                pass

        if prefer in ("auto", "tfidf"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=200_000
                )
                self.backend = "tfidf"
                return
            except Exception:
                pass

        self.backend = "hashing"

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
            )
            return np.asarray(vecs, dtype=np.float32)

        if self.backend == "tfidf":
            mat = self._vectorizer.fit_transform(texts)
            vecs = np.asarray(mat.todense(), dtype=np.float32)
            return _l2_normalize(vecs)

        return _l2_normalize(_hash_encode(texts))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _hash_encode(texts: list[str], dim: int = 2048) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
    return out


class NeighbourIndex:
    """Top-k cosine search over the corpus of ground-truth documents."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss = index
        except Exception:
            self._faiss = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self._faiss is not None:
            return self._faiss.search(q, k)
        # Chunked exact search; keeps peak memory bounded on the full corpus.
        sims_out = np.empty((q.shape[0], k), dtype=np.float32)
        idx_out = np.empty((q.shape[0], k), dtype=np.int64)
        chunk = max(1, int(5e7 // max(1, self.vectors.shape[0])))
        for start in range(0, q.shape[0], chunk):
            block = q[start : start + chunk] @ self.vectors.T
            part = np.argpartition(-block, kth=min(k, block.shape[1] - 1) - 1, axis=1)[:, :k]
            rows = np.arange(block.shape[0])[:, None]
            order = np.argsort(-block[rows, part], axis=1)
            top = part[rows, order]
            idx_out[start : start + chunk] = top
            sims_out[start : start + chunk] = block[rows, top]
        return sims_out, idx_out
