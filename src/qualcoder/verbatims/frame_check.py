# -*- coding: utf-8 -*-
"""Check topics against an existing keyword code frame.

Answers three questions: which topics the keyword frame barely covers (candidate new codes, ranked by
detractor volume); which code each topic corresponds to and how well they agree (Cohen's kappa at
comment level); and gives a stratified sample of comments per topic for hand validation.
"""
import importlib.util
import re
import numpy as np


def load_codes(path):
    """Load a code frame from a Python file. Accepts CODES = {name: regex} or the
    ALL = {name: (label, tier, compiled_regex, ...)} layout; regexes may be strings or compiled."""
    spec = importlib.util.spec_from_file_location("code_frame", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = getattr(mod, "CODES", None)
    if raw is None and hasattr(mod, "ALL"):
        raw = {k: v[2] for k, v in mod.ALL.items()}
    if raw is None:
        raise ValueError("code frame file defines neither CODES nor ALL")
    return {k: (re.compile(v, re.I) if isinstance(v, str) else v) for k, v in raw.items()}


def apply_codes(comments, codes):
    out = []
    for c in comments:
        t = c["text"].lower().replace("’", "'")
        out.append({k for k, rx in codes.items() if rx.search(t)})
    return out


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def topic_vs_codes(comments, topic_themes, code_sets, group="detractor"):
    idx = [i for i, c in enumerate(comments) if group is None or c["group"] == group]
    names = sorted({t for i in idx for t in topic_themes[i]})
    codes = sorted({k for i in idx for k in code_sets[i]})
    rows = []
    for t in names:
        it = np.array([t in topic_themes[i] for i in idx])
        n = int(it.sum())
        if n == 0:
            continue
        covered = sum(1 for i, f in zip(idx, it) if f and code_sets[i])
        best, best_k, best_share = None, float("nan"), 0.0
        for k in codes:
            ik = np.array([k in code_sets[i] for i in idx])
            share = (it & ik).sum() / n
            if share > best_share:
                best, best_share, best_k = k, share, kappa(it, ik)
        rows.append(dict(theme=t, detractor_comments=n, pct_with_any_code=100 * covered / n,
                         best_code=best or "", pct_with_best_code=100 * best_share, kappa_best=best_k))
    rows.sort(key=lambda r: (r["pct_with_any_code"] >= 50, -r["detractor_comments"]))
    for r in rows:
        r["status"] = "frame misses this" if r["pct_with_any_code"] < 50 else "covered"
    return rows


def validation_sample(comments, topic_themes, per_topic=20, seed=7, group="detractor"):
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(comments):
        if group is None or c["group"] == group:
            for t in topic_themes[i]:
                by.setdefault(t, []).append(i)
    rows = []
    for t in sorted(by):
        pick = rng.choice(by[t], size=min(per_topic, len(by[t])), replace=False)
        for i in pick:
            rows.append(dict(theme=t, id=comments[i]["id"], business=comments[i]["business"],
                             channel=comments[i]["channel"], text=comments[i]["text"], correct_yes_no=""))
    return rows
