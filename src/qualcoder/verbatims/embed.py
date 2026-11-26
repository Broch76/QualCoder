# -*- coding: utf-8 -*-
"""Embedding backends. Every result records which backend produced it, so each figure states its method.

- "precomputed": a .npy array (rows aligned with the units), e.g. computed on another machine.
- "transformer": a sentence-transformers model loaded from a LOCAL folder (no network; HF_HUB_OFFLINE is set).
- "lexical": offline TF-IDF over unigrams and bigrams, reduced by truncated SVD (needs scipy).
  Lexical vectors join comments that share words; they do not join paraphrases with no words in common.
"""
import os
import numpy as np
from .text import tokenize

# Words that carry meaning only next to another word ('not delivered', 'never arrived'): used in bigrams only,
# so on their own they do not pull unrelated comments together in the lexical backend.
WEAK = set("not no never always still very get got went sent wrong one nothing every all".split())


class LexicalEmbedder:
    def __init__(self, dim=300, min_df=2, bigrams=True, seed=42):
        self.dim, self.min_df, self.bigrams, self.seed = dim, min_df, bigrams, seed

    def _features(self, text):
        toks = tokenize(text)
        feats = [t for t in toks if t not in WEAK]
        if self.bigrams:
            feats += [a + " " + b for a, b in zip(toks, toks[1:])]
        return feats

    def _matrix(self, texts, fit):
        from scipy import sparse
        rows, cols, vals = [], [], []
        docs = [self._features(t) for t in texts]
        if fit:
            df = {}
            for d in docs:
                for f in set(d):
                    df[f] = df.get(f, 0) + 1
            vocab = sorted(f for f, c in df.items() if c >= self.min_df)
            self.vocab = {f: j for j, f in enumerate(vocab)}
            n = len(docs)
            self.idf = np.array([np.log((1 + n) / (1 + df[f])) + 1 for f in vocab])
        for i, d in enumerate(docs):
            counts = {}
            for f in d:
                j = self.vocab.get(f)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i); cols.append(j); vals.append((1 + np.log(c)) * self.idf[j])
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.diags(1 / norms) @ X

    def fit_transform(self, texts):
        from scipy.sparse.linalg import svds
        X = self._matrix(texts, fit=True)
        k = max(2, min(self.dim, min(X.shape) - 1))
        rng = np.random.default_rng(self.seed)
        v0 = rng.standard_normal(min(X.shape))
        _, s, vt = svds(X, k=k, v0=v0)
        order = np.argsort(-s)
        vt = vt[order]
        signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(1)])
        self.components = vt * signs[:, None]   # fixed sign: identical runs give identical vectors
        return self._project(X)

    def _project(self, X):
        Z = np.asarray(X @ self.components.T)
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return Z / n

    def transform(self, texts):
        return self._project(self._matrix(texts, fit=False))


def embed(texts, backend="lexical", model_path=None, npy_path=None, dim=300, seed=42):
    """Return (embeddings, info, embedder). `embedder.transform(texts)` embeds new units the same way."""
    if backend == "precomputed":
        E = np.load(npy_path)
        if len(E) != len(texts):
            raise ValueError("precomputed embeddings have %d rows for %d units" % (len(E), len(texts)))
        return E, {"backend": "precomputed", "source": os.path.basename(npy_path)}, None
    if backend == "transformer":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError("transformer backend needs a local model folder (model_path); none at %r" % model_path)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
        prefix = "query: " if "e5" in os.path.basename(model_path.rstrip("/")).lower() else ""

        class _T:
            def transform(self, t):
                return model.encode([prefix + x for x in t], normalize_embeddings=True, show_progress_bar=False)
        e = _T()
        return e.transform(texts), {"backend": "transformer", "model": os.path.basename(model_path.rstrip("/"))}, e
    if backend == "lexical":
        e = LexicalEmbedder(dim=dim, seed=seed)
        E = e.fit_transform(texts)
        return E, {"backend": "lexical", "method": "TF-IDF unigrams+bigrams, truncated SVD", "dim": int(E.shape[1]),
                   "vocabulary": len(e.vocab)}, e
    raise ValueError("unknown backend %r" % backend)
