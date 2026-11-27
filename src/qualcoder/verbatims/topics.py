# -*- coding: utf-8 -*-
"""Clause-level topic model for survey verbatims.

Each comment is split into clauses; clauses are embedded and clustered; a comment carries every topic
one of its clauses falls in (multi-label, comparable with a multi-label keyword frame). The number of
topics is chosen from a grid by cosine silhouette, subject to seed stability (adjusted Rand index).
Topic words are class-based TF-IDF over unigrams and bigrams; exemplars are the clauses nearest the
centroid, in original text. A fitted model assigns new months to the same topics with assign().
"""
import numpy as np
from .text import split_clauses, tokenize
from .embed import embed


def _normalise(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def kmeans(X, k, seed=42, n_init=8, max_iter=200):
    """Spherical-ish k-means on L2-normalised rows. scikit-learn when present, else numpy k-means++."""
    X = _normalise(X)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)
        return km.labels_.astype(int), _normalise(km.cluster_centers_)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    best = None
    sq = (X * X).sum(1)
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        d2 = np.maximum(sq - 2 * X @ C[0] + C[0] @ C[0], 0)
        for _ in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else None
            C.append(X[rng.choice(len(X), p=p)])
            d2 = np.minimum(d2, np.maximum(sq - 2 * X @ C[-1] + C[-1] @ C[-1], 0))
        C = np.array(C)
        for _ in range(max_iter):
            lab = np.argmax(X @ C.T - 0.5 * (C * C).sum(1), axis=1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1].astype(int), _normalise(best[2])


def adjusted_rand(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    s = c2(M).sum(); sa = c2(M.sum(1)).sum(); sb = c2(M.sum(0)).sum(); n = c2(len(a))
    exp = sa * sb / n if n else 0
    mx = (sa + sb) / 2
    return float((s - exp) / (mx - exp)) if mx != exp else 1.0


def silhouette_cosine(X, labels, sample=3000, seed=0):
    X = _normalise(X); labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs, ls = X[idx], labels[idx]
    D = 1 - Xs @ Xs.T
    uniq = np.unique(ls)
    if len(uniq) < 2:
        return 0.0
    means = np.stack([D[:, ls == u].mean(1) for u in uniq], axis=1)
    own = np.searchsorted(uniq, ls)
    cnt = np.array([(ls == u).sum() for u in uniq])
    a = means[np.arange(len(ls)), own] * cnt[own] / np.maximum(cnt[own] - 1, 1)
    means[np.arange(len(ls)), own] = np.inf
    b = means.min(1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[cnt[own] == 1] = 0
    return float(s.mean())


def choose_k(X, ks=range(6, 31, 2), seeds=(1, 2, 3), min_stability=0.75):
    """Grid over k: silhouette and mean pairwise seed ARI. Pick the best silhouette among stable k."""
    rows = []
    for k in ks:
        if k >= len(X):
            break
        labs = [kmeans(X, k, seed=s)[0] for s in seeds]
        ari = np.mean([adjusted_rand(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))])
        rows.append({"k": int(k), "silhouette": silhouette_cosine(X, labs[0]), "seed_ari": float(ari)})
    stable = [r for r in rows if r["seed_ari"] >= min_stability]
    pick = max(stable, key=lambda r: r["silhouette"]) if stable else max(rows, key=lambda r: r["seed_ari"])
    return pick["k"], rows


def ctfidf_words(unit_texts, labels, top_n=8):
    labels = np.asarray(labels)
    out, df, counts = {}, {}, {}
    for t in np.unique(labels):
        c = {}
        for txt in (unit_texts[i] for i in np.where(labels == t)[0]):
            toks = tokenize(txt)
            for f in toks + [a + " " + b for a, b in zip(toks, toks[1:])]:
                c[f] = c.get(f, 0) + 1
        counts[t] = c
        for f in c:
            df[f] = df.get(f, 0) + 1
    nt = len(counts)
    for t, c in counts.items():
        tot = sum(c.values()) or 1
        sc = {f: (v / tot) * np.log(1 + nt / df[f]) * (1.3 if " " in f else 1.0) for f, v in c.items() if v >= 2}
        out[int(t)] = [f for f, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    return out


class TopicModel:
    def __init__(self, backend="lexical", model_path=None, npy_path=None, k=None, ks=range(8, 41, 4), seed=42,
                 min_similarity=0.0):
        self.backend, self.model_path, self.npy_path = backend, model_path, npy_path
        self.k, self.ks, self.seed, self.min_similarity = k, ks, seed, min_similarity
        self.names = {}

    @staticmethod
    def units(comments):
        U = []
        for ci, c in enumerate(comments):
            if c.get("no_content"):
                continue
            for cl in split_clauses(c["text"]):
                U.append((ci, cl))
        return U

    def fit(self, comments, balance=True):
        """Fit on all comments, or (balance=True) on all detractor and passive comments plus an equal-sized
        random sample of promoter comments, so the many short praise comments do not use up most topics.
        Every comment is then assigned to the fitted topics."""
        fit_comments = comments
        if balance:
            rng = np.random.default_rng(self.seed)
            non = [c for c in comments if c.get("group") != "promoter"]
            pro = [c for c in comments if c.get("group") == "promoter"]
            m = min(len(pro), sum(1 for c in non if c.get("group") == "detractor"))
            pick = rng.choice(len(pro), size=m, replace=False) if m else []
            fit_comments = non + [pro[i] for i in sorted(pick)]
        self.fit_n = len(fit_comments)
        U = self.units(fit_comments)
        texts = [u[1] for u in U]
        E, self.embed_info, self.embedder = embed(texts, backend=self.backend, model_path=self.model_path,
                                                  npy_path=self.npy_path, seed=self.seed)
        self.k_grid = None
        if self.k is None:
            self.k, self.k_grid = choose_k(E, ks=self.ks)
        self.unit_labels, self.centroids = kmeans(E, self.k, seed=self.seed)
        others = [kmeans(E, self.k, seed=s)[0] for s in (self.seed + 1, self.seed + 2)]
        self.stability = float(np.mean([adjusted_rand(self.unit_labels, o) for o in others]))
        sim = _normalise(E) @ self.centroids.T
        self.unit_sim = sim[np.arange(len(E)), self.unit_labels]
        self.words = ctfidf_words(texts, self.unit_labels)
        self.exemplars = {}
        for t in range(self.k):
            idx = np.where(self.unit_labels == t)[0]
            seen, ex = set(), []
            for i in idx[np.argsort(-self.unit_sim[idx])]:
                key = texts[i].strip().lower()
                if key not in seen:
                    seen.add(key); ex.append(texts[i])
                if len(ex) == 5:
                    break
            self.exemplars[t] = ex
        self.names = {t: "t%d: %s" % (t, " / ".join(self.words.get(t, [])[:3])) for t in range(self.k)}
        self.units_ = U
        self.comment_topics = self.assign(comments) if self.embedder is not None else \
            self._collect(len(comments), U, self.unit_labels, self.unit_sim)
        return self

    def _collect(self, n, U, labels, sims):
        ct = [set() for _ in range(n)]
        for (ci, _), t, s in zip(U, labels, sims):
            if s >= self.min_similarity:
                ct[ci].add(int(t))
        return ct

    def assign(self, comments):
        """Assign new comments (e.g. a later month) to the fitted topics without refitting."""
        if self.embedder is None:
            raise ValueError("precomputed embeddings cannot embed new text; supply a transformer or lexical backend")
        U = self.units(comments)
        E = _normalise(self.embedder.transform([u[1] for u in U]))
        sim = E @ self.centroids.T
        lab = sim.argmax(1)
        return self._collect(len(comments), U, lab, sim[np.arange(len(E)), lab])

    def rename(self, mapping):
        """mapping: {topic id: theme name}. Topics mapped to the same name are merged in every output."""
        self.names.update({int(k): v for k, v in mapping.items()})

    def themes(self, comment_topics=None):
        """Per comment, the set of theme names (after rename/merge)."""
        ct = self.comment_topics if comment_topics is None else comment_topics
        return [{self.names[t] for t in s} for s in ct]

    def summary(self):
        sizes = np.bincount(self.unit_labels, minlength=self.k)
        return [{"topic": t, "name": self.names[t], "clauses": int(sizes[t]), "words": self.words.get(t, []),
                 "exemplars": self.exemplars.get(t, [])} for t in np.argsort(-sizes)]
