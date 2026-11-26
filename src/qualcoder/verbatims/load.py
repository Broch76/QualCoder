# -*- coding: utf-8 -*-
"""Load survey rows into analysable comments, with a row-count trail for every base in the report."""
import re
import numpy as np
from .text import is_no_content

BLANKS = {"", "nan", "none", "skipped", "null", "-", "n/a"}
DEFAULT_CHANNEL_MAP = {"TELEPHONE": "TELEPHONIC", "PHONE": "TELEPHONIC", "CALL CENTRE": "TELEPHONIC",
                       "E-MAIL": "EMAIL", "MAIL": "EMAIL", "COUNTER": "COUNTER CALLS", "WALK-IN": "COUNTER CALLS",
                       "WALK IN": "COUNTER CALLS"}


def nps_group(score):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    s = float(score)
    return "detractor" if s <= 6 else "passive" if s <= 8 else "promoter"


def load_verbatims(df, business=None, text_col="Free Response", score_col="QNet Promoter Score",
                   channel_col="Interaction Channel", month_col="Month", id_col="Response ID",
                   business_col="business", months=None, channel_map=None, min_chars=3):
    """Return (comments, trail).

    comments: list of dicts {id, business, channel, month, score, group, text, no_content}
    trail: ordered list of (step, rows remaining, rows removed) so each base can be traced.
    `business` labels a single-business frame (as in voc_<business>.pkl); otherwise `business_col` is read.
    """
    import pandas as pd
    cmap = dict(DEFAULT_CHANNEL_MAP)
    cmap.update(channel_map or {})
    trail = []
    n0 = len(df)
    trail.append(("rows read", n0, 0))
    d = df.copy()
    if months is not None:
        keep = d[month_col].astype(str).str.strip().isin(months)
        trail.append(("outside the month window", int(keep.sum()), int((~keep).sum())))
        d = d[keep]
    score = pd.to_numeric(d[score_col], errors="coerce")
    keep = score.notna()
    trail.append(("no NPS score", int(keep.sum()), int((~keep).sum())))
    d, score = d[keep], score[keep]
    txt = d[text_col].astype(str).map(lambda t: re.sub(r"\s+", " ", t).strip())
    blank = txt.str.lower().isin(BLANKS) | (txt.str.len() < min_chars)
    trail.append(("blank or skipped comment", int((~blank).sum()), int(blank.sum())))
    d, score, txt = d[~blank], score[~blank], txt[~blank]
    ch = d[channel_col].astype(str).str.upper().str.strip().map(lambda c: cmap.get(c, c)) if channel_col in d else "ALL"
    bus = business if business is not None else d[business_col].astype(str)
    comments = []
    for i, (idx, row) in enumerate(d.iterrows()):
        t = txt.loc[idx]
        comments.append({
            "id": str(row[id_col]) if id_col in d else str(idx),
            "business": bus if isinstance(bus, str) else bus.loc[idx],
            "channel": ch if isinstance(ch, str) else ch.loc[idx],
            "month": str(row[month_col]).strip() if month_col in d else "",
            "score": float(score.loc[idx]),
            "group": nps_group(score.loc[idx]),
            "text": t,
            "no_content": bool(is_no_content(t)),
        })
    nc = sum(c["no_content"] for c in comments)
    trail.append(("analysable comments (no-content comments kept, counted, not clustered)", len(comments) - nc, nc))
    return comments, trail


def merge_trails(trails):
    """Sum per-business trails step by step (same steps, same order)."""
    out = []
    for steps in zip(*trails):
        out.append((steps[0][0], sum(s[1] for s in steps), sum(s[2] for s in steps)))
    return out
