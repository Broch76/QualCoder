# -*- coding: utf-8 -*-
"""Statistics for theme prevalence in survey comments: how big, how sure, tested not shown.

Unit of analysis is the respondent comment (a comment counts once per theme however often a word recurs).
- Prevalence: share of comments in a cell carrying a theme, with a Wilson 95% interval.
- Base flags: n < 30 suppressed, 30 to 99 flagged "caution".
- Business effect: business v the rest, Mantel-Haenszel odds ratio stratified by channel, so a business
  that simply takes more of a channel is not credited with that channel's themes.
- Channel effect: channel v the rest, stratified by business.
- Cell effect: within one business, one channel v that business's other channels.
- Detractor v promoter lift per theme.
All p-values in one run are adjusted together (Benjamini-Hochberg); "significant" means q < 0.05 and
both compared groups have n >= 30.
"""
import math
import numpy as np

SUPPRESS, CAUTION = 30, 100


def base_flag(n):
    return "suppress" if n < SUPPRESS else "caution" if n < CAUTION else ""


def wilson(x, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def chi2_p(stat):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, 1))
    except ImportError:
        return float(math.erfc(math.sqrt(max(stat, 0) / 2)))


def mantel_haenszel(y, e, strata):
    """y, e: 0/1 arrays (theme present, exposed); strata: labels. Returns (OR_MH, chi2, p)."""
    y, e, strata = np.asarray(y), np.asarray(e), np.asarray(strata)
    num = den = a_sum = ea_sum = var = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = float(((y == 1) & (e == 1) & m).sum()); b = float(((y == 0) & (e == 1) & m).sum())
        c = float(((y == 1) & (e == 0) & m).sum()); d = float(((y == 0) & (e == 0) & m).sum())
        n = a + b + c + d
        if n < 2 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * d / n; den += b * c / n
        r1, c1 = a + b, a + c
        a_sum += a; ea_sum += r1 * c1 / n
        var += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
    if var == 0:
        return float("nan"), 0.0, 1.0
    stat = (abs(a_sum - ea_sum)) ** 2 / var
    orr = num / den if den > 0 else float("inf")
    return orr, stat, chi2_p(stat)


def two_prop(x1, n1, x2, n2):
    """Pooled two-proportion chi-square (1 df) and odds ratio with 0.5 correction for zero cells."""
    if min(n1, n2) == 0:
        return float("nan"), 1.0
    p = (x1 + x2) / (n1 + n2)
    if p in (0, 1):
        return float("nan"), 1.0
    stat = (x1 / n1 - x2 / n2) ** 2 / (p * (1 - p) * (1 / n1 + 1 / n2))
    a, b, c, d = x1 + .5, n1 - x1 + .5, x2 + .5, n2 - x2 + .5
    return (a * d) / (b * c), chi2_p(stat)


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1)
    return out


def _indicator(themes, t):
    return np.array([1 if t in s else 0 for s in themes])


def prevalence(comments, themes, by=("business",), group="detractor"):
    idx = [i for i, c in enumerate(comments) if group is None or c["group"] == group]
    all_themes = sorted({t for i in idx for t in themes[i]})
    cells = {}
    for i in idx:
        cells.setdefault(tuple(comments[i][b] for b in by), []).append(i)
    rows = []
    for key in sorted(cells):
        ii = cells[key]; n = len(ii)
        for t in all_themes:
            x = sum(1 for i in ii if t in themes[i])
            lo, hi = wilson(x, n)
            rows.append(dict(zip(by, key), theme=t, n=n, x=x, pct=100 * x / n, lo=100 * lo, hi=100 * hi,
                             flag=base_flag(n)))
    return rows


def effects(comments, themes, group="detractor"):
    """Business, channel and business x channel effects for every theme, BH-adjusted together."""
    idx = [i for i, c in enumerate(comments) if c["group"] == group]
    C = [comments[i] for i in idx]; T = [themes[i] for i in idx]
    bus = np.array([c["business"] for c in C]); ch = np.array([c["channel"] for c in C])
    rows = []
    for t in sorted({t for s in T for t in s}):
        y = _indicator(T, t)
        for b in np.unique(bus):
            e = (bus == b).astype(int)
            n1, n2 = int(e.sum()), int((1 - e).sum()); x1, x2 = int(y[e == 1].sum()), int(y[e == 0].sum())
            or_raw, p_raw = two_prop(x1, n1, x2, n2)
            or_mh, _, p_mh = mantel_haenszel(y, e, ch)
            rows.append(dict(kind="business", theme=t, group_a=b, group_b="other businesses", n_a=n1, n_b=n2,
                             pct_a=100 * x1 / n1, pct_b=100 * x2 / n2 if n2 else float("nan"),
                             or_unadjusted=or_raw, p_unadjusted=p_raw, odds_ratio=or_mh, p=p_mh,
                             adjusted_for="channel"))
        for c in np.unique(ch):
            e = (ch == c).astype(int)
            n1, n2 = int(e.sum()), int((1 - e).sum()); x1, x2 = int(y[e == 1].sum()), int(y[e == 0].sum())
            or_raw, p_raw = two_prop(x1, n1, x2, n2)
            or_mh, _, p_mh = mantel_haenszel(y, e, bus)
            rows.append(dict(kind="channel", theme=t, group_a=c, group_b="other channels", n_a=n1, n_b=n2,
                             pct_a=100 * x1 / n1, pct_b=100 * x2 / n2 if n2 else float("nan"),
                             or_unadjusted=or_raw, p_unadjusted=p_raw, odds_ratio=or_mh, p=p_mh,
                             adjusted_for="business"))
        for b in np.unique(bus):
            mb = bus == b
            chs = np.unique(ch[mb])
            if len(chs) < 2:
                continue
            for c in chs:
                a = mb & (ch == c); o = mb & (ch != c)
                n1, n2 = int(a.sum()), int(o.sum()); x1, x2 = int(y[a].sum()), int(y[o].sum())
                orr, p = two_prop(x1, n1, x2, n2)
                rows.append(dict(kind="cell", theme=t, group_a="%s %s" % (b, c), group_b="%s other channels" % b,
                                 n_a=n1, n_b=n2, pct_a=100 * x1 / n1 if n1 else float("nan"),
                                 pct_b=100 * x2 / n2 if n2 else float("nan"), or_unadjusted=orr, p_unadjusted=p,
                                 odds_ratio=orr, p=p, adjusted_for="none (within one business)"))
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
        r["flag"] = base_flag(min(r["n_a"], r["n_b"]))
        r["significant"] = bool(qq < 0.05 and r["flag"] != "suppress")
        r["direction"] = "higher" if r["odds_ratio"] > 1 else "lower"
    return rows


def lift(comments, themes):
    """Theme prevalence among detractors v promoters (which themes separate the two)."""
    D = [themes[i] for i, c in enumerate(comments) if c["group"] == "detractor"]
    P = [themes[i] for i, c in enumerate(comments) if c["group"] == "promoter"]
    rows = []
    for t in sorted({t for s in D + P for t in s}):
        xd = sum(t in s for s in D); xp = sum(t in s for s in P)
        orr, p = two_prop(xd, len(D), xp, len(P))
        rows.append(dict(theme=t, pct_detractor=100 * xd / max(len(D), 1), pct_promoter=100 * xp / max(len(P), 1),
                         odds_ratio=orr, p=p))
    q = bh([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
    return rows
