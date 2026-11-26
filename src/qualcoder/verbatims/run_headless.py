# -*- coding: utf-8 -*-
"""End-to-end verbatim run, offline, outputs written locally.

python -m qualcoder.verbatims.run_headless --data DIR_WITH_voc_PKLS --out OUT_DIR
    [--backend lexical|transformer|precomputed] [--model-path DIR] [--npy FILE]
    [--codes CODE_FRAME.py] [--names NAMES.json] [--k K] [--months "August 2025,..."] [--model-in model.pkl]
    [--llm off|api|handoff] [--llm-scope all|detractor] [--llm-model-bulk M] [--llm-model-reason M] [--channel-families]
    [--context BUSINESS_CONTEXT.json] [--reference REFERENCE.csv]

Two-step use: run once (writes model.pkl); read topics.json and write NAMES.json ({topic id: theme name},
same name = merged); run again with --model-in OUT/model.pkl --names NAMES.json for the named tables.

Reads voc_<business>.pkl (columns: Response ID, QNet Promoter Score, Free Response, Month,
Interaction Channel); voc_all.pkl is skipped. Writes CSV and JSON tables and METHODS.md. Raw comment text appears only in
topics.json (exemplars), validation_sample.csv and llm/*.jsonl: keep OUT_DIR with the raw data.

--llm adds the Claude layer (see llm.py): topic names when --names is not given, new codes induced from the
uncoded detractor comments, and Claude coding of every comment against keyword labels plus induced codes,
with its own effects and prevalence tables and a code-by-code comparison with the keyword frame.
"""
import argparse, csv, glob, json, os, pickle, time, warnings
import numpy as np
from .load import load_verbatims, merge_trails
from .topics import TopicModel
from . import stats, frame_check


def read_pickle(path):
    """pd.read_pickle that also opens frames pickled by pandas 3 (string columns) under pandas 2.x."""
    import pandas as pd
    try:
        return pd.read_pickle(path)
    except NotImplementedError:
        from pandas.core.arrays.string_ import StringArray
        from pandas._libs.arrays import NDArrayBacked
        orig = NDArrayBacked.__setstate__

        def _setstate(self, state):
            if isinstance(state, tuple) and len(state) == 2 and not hasattr(state[0], "shape"):
                return orig(self, {"_ndarray": state[1], "_dtype": state[0]})
            return orig(self, state)
        StringArray.__setstate__ = _setstate
        try:
            return pd.read_pickle(path)
        finally:
            del StringArray.__setstate__


def _csv(path, rows):
    if not rows:
        open(path, "w").close(); return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _length_table(comments):
    """Comment length (characters) by business and NPS group: Kobus reads long comments as a signal."""
    rows = []
    keys = sorted({(c["business"], c["group"]) for c in comments if c["group"]})
    for b, g in keys:
        L = np.array([len(c["text"]) for c in comments if c["business"] == b and c["group"] == g and not c["no_content"]])
        if len(L):
            rows.append({"business": b, "group": g, "n": len(L), "median_chars": float(np.median(L)),
                         "p90_chars": float(np.percentile(L, 90)), "share_over_300_chars": float((L > 300).mean())})
    return rows


def _expectation_table(sc, exp, th):
    """Where the expectation was set, share of detractor comments with a substantive code, by business."""
    rows = []
    for b in sorted({c["business"] for c in sc}):
        idx = [i for i, c in enumerate(sc) if c["business"] == b and c["group"] == "detractor"
               and th[i] - {"praise", "no_content", "other"}]
        n = len(idx)
        for src in sorted({exp.get(i, "unclear") for i in idx}):
            k = sum(1 for i in idx if exp.get(i, "unclear") == src)
            lo, hi = stats.wilson(k, n)
            rows.append({"business": b, "expectation": src, "n_base": n, "n": k, "pct": 100 * k / n,
                         "ci_low": 100 * lo, "ci_high": 100 * hi, "flag": stats.base_flag(n)})
    return rows


def _tier_table(cc, lsc, book):
    """Share of detractor comments carrying at least one code in each controllability tier, by business.
    Tiers overlap (a comment can name several), so rows do not sum to 100. "no tier code" = substantive codes
    only from codes without a tier (induced codes); "no substantive code" = praise, other or nothing."""
    tier = {b["id"]: b.get("tier") for b in book if b.get("tier")}
    rows = []
    for bus in sorted({c["business"] for c in cc}):
        idx = [k for k, c in enumerate(cc) if c["business"] == bus and c["group"] == "detractor"]
        n = len(idx)
        cats = {t: (lambda k, t=t: any(tier.get(x) == t for x in lsc[k])) for t in sorted(set(tier.values()))}
        cats["no tier code"] = lambda k: bool(lsc[k] - {"praise", "other"}) and not any(x in tier for x in lsc[k])
        cats["no substantive code"] = lambda k: not (lsc[k] - {"praise", "other"})
        for t, fn in cats.items():
            x = sum(1 for k in idx if fn(k))
            lo, hi = stats.wilson(x, n)
            rows.append({"business": bus, "tier": t, "n_base": n, "n": x, "pct": 100 * x / n if n else 0,
                         "ci_low": 100 * lo, "ci_high": 100 * hi, "flag": stats.base_flag(n)})
    return rows


def _reference_tier_check(cc, path, kw, cl, book, exclude=("unresolved",)):
    """Tier-level agreement with a hand-coded reference whose CSV has columns id and tier. A comment's coder
    tier is the most controllable tier among its codes (Health controls > Health influences > Scheme board
    decides), else "neither". Reported with and without the codes in `exclude`, and against the constant guess."""
    from collections import Counter
    rank = ["Health controls", "Health influences", "Scheme board decides"]
    tier = {b["id"]: b.get("tier") for b in book if b.get("tier")}

    def top(codes, drop=()):
        ts = {tier[c] for c in codes if c in tier and c not in drop}
        return next((t for t in rank if t in ts), "neither")
    ref = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r.get("tier"):
            ref[r["id"].strip()] = r["tier"].strip()
    pos = {c["id"]: k for k, c in enumerate(cc)}
    ids = [i for i in ref if i in pos]
    rows = []
    if not ids:
        return [{"note": "reference has no tier column or no ids match", "n_reference": len(ref)}]
    modal, modal_n = Counter(ref[i] for i in ids).most_common(1)[0]
    rows.append({"coder": "constant guess (%s)" % modal, "n": len(ids), "agree": modal_n, "kappa": 0.0})
    for name, sets in (("keyword", kw), ("claude", cl)):
        for drop in ((), tuple(exclude)):
            pred = [top(sets[pos[i]], drop) for i in ids]
            truth = [ref[i] for i in ids]
            agree = sum(p == t for p, t in zip(pred, truth))
            cats = sorted(set(pred) | set(truth))
            po = agree / len(ids)
            pe = sum((pred.count(c) / len(ids)) * (truth.count(c) / len(ids)) for c in cats)
            rows.append({"coder": name + (" without " + ",".join(drop) if drop else ""), "n": len(ids),
                         "agree": agree, "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else ""})
    return rows


def _context_swap(claude, llm, sc_all, th, book, ctx, model, notes, per_business=60, seed=5):
    """Bias check: re-code a stratified sample of answered detractor comments twice, once with the true
    business context and once with another business's context, and compare code rates with the blind codes."""
    import random
    rng = random.Random(seed)
    buses = sorted(ctx)
    sample = []
    for b in buses:
        idx = [j for j, c in enumerate(sc_all) if c["business"] == b and c["group"] == "detractor" and th[j] is not None]
        sample += rng.sample(idx, min(per_business, len(idx)))
    sub = [sc_all[j] for j in sample]
    rot = {b: ctx[buses[(k + 1) % len(buses)]] for k, b in enumerate(buses)}
    out = {}
    for name, cx in (("own context", ctx), ("swapped context", rot)):
        try:
            t2, _, _ = llm.code(claude, sub, book, model=model, job="code_" + name.split()[0], context=cx)
            out[name] = t2
        except llm.NeedResponses as e:
            notes.append("Context-swap check pending (%s): %s" % (name, e))
            return []
    rows = []
    for code in [b["id"] for b in book]:
        for b in buses:
            ks = [k for k, j in enumerate(sample) if sc_all[j]["business"] == b]
            r = {"code": code, "business": b, "n": len(ks),
                 "blind": sum(1 for k in ks if code in th[sample[k]])}
            for name, t2 in out.items():
                r[name] = sum(1 for k in ks if t2[k] is not None and code in t2[k])
            if r["blind"] or any(r[n] for n in out):
                rows.append(r)
    notes.append("Context-swap check: %d detractor comments (%d per business) re-coded with their own business "
                 "context and with another business's context; code counts by business in context_swap_check.csv."
                 % (len(sample), per_business))
    return rows


def _reference_check(comments, path, kw, cl, codes):
    """Agreement with a hand-coded reference set: CSV with columns id and codes (codes separated by |)."""
    from .frame_check import kappa
    ref = {r["id"].strip(): {x.strip() for x in r["codes"].split("|") if x.strip()}
           for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
    pos = {c["id"]: i for i, c in enumerate(comments)}
    ids = [i for i in ref if i in pos]
    rows = [{"code": "_matched_reference_comments", "n_reference": len(ref), "n_matched": len(ids)}]
    for k in codes:
        r = [k in ref[i] for i in ids]
        for name, sets in (("keyword", kw), ("claude", cl)):
            p = [k in sets[pos[i]] for i in ids]
            tp = sum(a and b for a, b in zip(p, r))
            rows.append({"code": k, "coder": name, "n_reference": sum(r), "n_coder": sum(p), "n_both": tp,
                         "precision": round(tp / sum(p), 3) if sum(p) else "",
                         "recall": round(tp / sum(r), 3) if sum(r) else "", "kappa": round(kappa(p, r), 3)})
    return rows


def run(data, out, backend="lexical", model_path=None, npy=None, codes=None, names=None, k=None, months=None,
        model_in=None, llm_mode="off", llm_scope="all", llm_model_bulk=None, llm_model_reason=None, families=False, context=None, reference=None, definitions=None):
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Number of distinct clusters")
    os.makedirs(out, exist_ok=True)
    comments, trails = [], []
    for p in sorted(glob.glob(os.path.join(data, "voc_*.pkl"))):
        b = os.path.basename(p)[4:-4]
        if b.lower() == "all":
            continue
        c, tr = load_verbatims(read_pickle(p), business=b, months=months, families=families)
        comments += c; trails.append(tr)
    trail = merge_trails(trails)
    if model_in:
        # Reuse a saved model (same topics, same ids): for naming/merging topics, or for later months.
        tm = pickle.load(open(model_in, "rb"))
        tm.comment_topics = tm.assign(comments)
    else:
        tm = TopicModel(backend=backend, model_path=model_path, npy_path=npy, k=k).fit(comments)
        if backend != "transformer":
            pickle.dump(tm, open(os.path.join(out, "model.pkl"), "wb"))
    claude = None
    llm_notes = []
    if llm_mode != "off":
        from . import llm
        claude = llm.Claude(llm_mode, out)
        mb, mr = llm_model_bulk or llm.MODEL_BULK, llm_model_reason or llm.MODEL_REASON
    if names:
        tm.rename(json.load(open(names)))
    elif claude:
        try:
            nm = llm.name_topics(claude, tm.summary(), model=mr)
            json.dump(nm, open(os.path.join(out, "names_claude.json"), "w"), indent=1)
            tm.rename(nm)
            llm_notes.append("Topic names: Claude (%s), from top words and centroid exemplars; "
                             "names_claude.json can be edited and passed back with --names." % mr)
        except llm.NeedResponses as e:
            llm_notes.append("Topic names pending: %s" % e)
    themes = tm.themes()
    res = {"trail": trail, "embedding": tm.embed_info, "k": tm.k, "k_grid": tm.k_grid, "seed_ari": tm.stability,
           "topics": tm.summary()}
    _csv(os.path.join(out, "prevalence_business.csv"), stats.prevalence(comments, themes, ("business",)))
    _csv(os.path.join(out, "prevalence_business_channel.csv"), stats.prevalence(comments, themes, ("business", "channel")))
    eff = stats.effects(comments, themes)
    _csv(os.path.join(out, "comment_length.csv"), _length_table(comments))
    _csv(os.path.join(out, "effects_topics.csv"), eff)
    _csv(os.path.join(out, "lift_topics.csv"), stats.lift(comments, themes))
    if codes:
        cf = frame_check.load_codes(codes)
        cs = frame_check.apply_codes(comments, cf)
        _csv(os.path.join(out, "frame_check.csv"), frame_check.topic_vs_codes(comments, themes, cs))
        _csv(os.path.join(out, "effects_keyword_codes.csv"), stats.effects(comments, cs))
        _csv(os.path.join(out, "prevalence_keyword_codes.csv"), stats.prevalence(comments, cs, ("business",)))
        # What the frame misses: cluster only the detractor comments that no keyword code matches.
        resid = [c for c, s in zip(comments, cs) if c["group"] == "detractor" and not s and not c["no_content"]]
        res["uncoded_detractor_comments"] = len(resid)
        res["uncoded_share_of_detractor_comments"] = len(resid) / max(1, sum(1 for c in comments if c["group"] == "detractor"))
        if len(resid) >= 50:
            rm = TopicModel(backend=backend if backend != "precomputed" else "lexical", model_path=model_path,
                            ks=range(4, 17, 2)).fit(resid, balance=False)
            rt = rm.themes()
            summ = rm.summary()
            for row in summ:
                members = [c for c, s in zip(resid, rt) if row["name"] in s]
                row["comments"] = len(members)
                row["by_business"] = {b: sum(1 for c in members if c["business"] == b) for b in sorted({c["business"] for c in comments})}
            res["uncoded_topics"] = summ
        if claude:
            labels = frame_check.load_labels(codes)
            meta = frame_check.load_frame_meta(codes)
            ctx = json.load(open(context)) if context else None
            if definitions:
                defs = json.load(open(definitions))
                labels = {k: (v, defs.get(k, "")) for k, v in labels.items()}
            book = llm.build_codebook(labels, meta=meta)
            induced = None
            try:
                ind, sample = llm.induce_codes(claude, [c["text"] for c in resid], book, model=mr)
                induced = ind["new_codes"]
                res["claude_induced_codes"] = [{**x, "examples": [sample[i] for i in x["examples"][:5] if i < len(sample)]}
                                               for x in induced]
                res["claude_frame_gaps"] = [{**x, "examples": [sample[i] for i in x["examples"][:5] if i < len(sample)]}
                                            for x in ind["frame_gaps"]]
                llm_notes.append("New codes: Claude (%s) read %d of %d uncoded detractor comments and proposed %d codes."
                                 % (mr, len(sample), len(resid), len(induced)))
            except llm.NeedResponses as e:
                llm_notes.append("New-code step pending: %s" % e)
            if induced is not None:
                book = llm.build_codebook(labels, induced, meta=meta)
                json.dump(book, open(os.path.join(out, "claude_codebook.json"), "w"), indent=1)
                scope = [i for i, c in enumerate(comments) if not c["no_content"]
                         and (llm_scope == "all" or c["group"] == "detractor")]
                try:
                    exp = {}
                    sc_all = [comments[i] for i in scope]
                    th, other, unanswered = llm.code(claude, sc_all, book, model=mb, expectation=exp)   # blind
                    keep = [j for j in range(len(scope)) if th[j] is not None]
                    # Claude tables use answered comments only; the keyword sets are cut to the same base.
                    cc = [sc_all[j] for j in keep]
                    lsc = [th[j] - {"no_content"} for j in keep]
                    kwc = [cs[scope[j]] for j in keep]
                    expc = {k: exp[j] for k, j in enumerate(keep) if j in exp}
                    _csv(os.path.join(out, "expectation_by_business.csv"), _expectation_table(cc, expc, lsc))
                    _csv(os.path.join(out, "tier_by_business.csv"), _tier_table(cc, lsc, book))
                    _csv(os.path.join(out, "effects_claude_codes.csv"), stats.effects(cc, lsc))
                    _csv(os.path.join(out, "prevalence_claude_codes.csv"), stats.prevalence(cc, lsc, ("business",)))
                    _csv(os.path.join(out, "keyword_v_claude.csv"), llm.agreement(cc, kwc, lsc, sorted(labels)))
                    if reference:
                        with open(reference, newline="", encoding="utf-8") as fh:
                            cols = set(next(csv.reader(fh), []))
                        if "codes" in cols:
                            _csv(os.path.join(out, "reference_check.csv"),
                                 _reference_check(cc, reference, kwc, lsc, sorted(labels)))
                        if "tier" in cols:
                            _csv(os.path.join(out, "reference_tier_check.csv"),
                                 _reference_tier_check(cc, reference, kwc, lsc, book))
                    if ctx:
                        _csv(os.path.join(out, "context_swap_check.csv"),
                             _context_swap(claude, llm, sc_all, th, book, ctx, mb, llm_notes))
                    oc = {}
                    for i, lab in other.items():
                        oc[lab] = oc.get(lab, 0) + 1
                    res["claude_other_labels"] = sorted(oc.items(), key=lambda x: -x[1])[:40]
                    det = [k for k, c in enumerate(cc) if c["group"] == "detractor"]
                    res["claude_uncoded_share_of_detractor_comments"] = (
                        sum(1 for k in det if not (lsc[k] - {"praise", "other"})) / max(1, len(det)))
                    res["keyword_uncoded_share_same_base"] = sum(1 for k in det if not kwc[k]) / max(1, len(det))
                    res["claude_unanswered"] = unanswered
                    res["claude_answered_by"] = claude.answered_by()
                    with open(os.path.join(out, "assignments_claude.csv"), "w", newline="") as f:
                        w = csv.writer(f); w.writerow(["id", "business", "channel", "group", "codes", "expectation"])
                        for k, c in enumerate(cc):
                            w.writerow([c["id"], c["business"], c["channel"], c["group"], " | ".join(sorted(lsc[k])),
                                        expc.get(k, "")])
                    llm_notes.append(
                        "Comment coding: blind (businesses shuffled together, no business named or described in the "
                        "prompt), %d comments (%s), %d per request; requested model %s; answers recorded as from: %s; "
                        "identifiers masked before sending; %d comments unanswered and excluded from every Claude "
                        "table; keyword comparisons use the same answered base."
                        % (len(scope), llm_scope, llm.BATCH, mb, json.dumps(claude.answered_by()), unanswered))
                except llm.NeedResponses as e:
                    llm_notes.append("Comment coding pending: %s" % e)
    _csv(os.path.join(out, "validation_sample.csv"), frame_check.validation_sample(comments, themes))
    with open(os.path.join(out, "assignments.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "business", "channel", "group", "themes"])
        for c, s in zip(comments, themes):
            w.writerow([c["id"], c["business"], c["channel"], c["group"], " | ".join(sorted(s))])
    json.dump(res, open(os.path.join(out, "topics.json"), "w"), indent=1, default=str)
    sig = [r for r in eff if r["significant"]]
    with open(os.path.join(out, "METHODS.md"), "w") as f:
        f.write("# Method note: verbatim themes\n\n")
        f.write("Basis: survey comments; one comment counts once per theme. Detractor = score 0 to 6.\n\n")
        f.write("Rows: " + "; ".join("%s: %d kept, %d removed" % (s, n, r) for s, n, r in trail) + ".\n\n")
        f.write("Themes: comments split into clauses; clauses embedded (%s) and clustered by k-means; "
                "k = %d chosen by cosine silhouette among settings with seed stability (adjusted Rand) of at least 0.75; "
                "stability of the final model %.2f.\n\n" % (json.dumps(tm.embed_info), tm.k, tm.stability))
        f.write("Tests: business v other businesses adjusted for channel (Mantel-Haenszel); channel v other channels "
                "adjusted for business; within-business channel cells unadjusted. Benjamini-Hochberg within each table "
                "(each code set corrected separately; %d topic tests); "
                "significant = q < 0.05 with both groups n >= 30. Cells under 30 are suppressed; 30 to 99 flagged caution.\n\n"
                % len(eff))
        f.write("Significant topic effects: %d.\n\n" % len(sig))
        if llm_notes:
            f.write("Claude layer (%s mode):\n" % llm_mode + "".join("- %s\n" % n for n in llm_notes) + "\n")
        f.write("Run time %.1f s.\n" % (time.time() - t0))
    if claude and claude.log:
        json.dump(claude.log, open(os.path.join(out, "llm", "usage_log.json"), "w"), indent=1)
    res["llm_notes"] = llm_notes
    json.dump(res, open(os.path.join(out, "topics.json"), "w"), indent=1, default=str)
    return comments, tm, eff


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="lexical"); ap.add_argument("--model-path"); ap.add_argument("--npy")
    ap.add_argument("--codes"); ap.add_argument("--names"); ap.add_argument("--k", type=int)
    ap.add_argument("--months"); ap.add_argument("--model-in")
    ap.add_argument("--llm", default="off", choices=["off", "api", "handoff"])
    ap.add_argument("--llm-scope", default="all", choices=["all", "detractor"])
    ap.add_argument("--llm-model-bulk"); ap.add_argument("--llm-model-reason")
    ap.add_argument("--context", help="JSON {business: context text}; used only for the context-swap check "
                    "(the coding itself is blind)")
    ap.add_argument("--definitions", help="JSON {code: definition with inclusions and exclusions}")
    ap.add_argument("--reference", help="hand-coded reference CSV: columns id and codes (separated by |) and/or id and tier")
    ap.add_argument("--channel-families", action="store_true",
                    help="collapse channel spellings to PHONE, EMAIL, WALK-IN, DIGITAL, OTHER")
    a = ap.parse_args(argv)
    run(a.data, a.out, a.backend, a.model_path, a.npy, a.codes, a.names, a.k,
        [m.strip() for m in a.months.split(",")] if a.months else None, a.model_in,
        a.llm, a.llm_scope, a.llm_model_bulk, a.llm_model_reason, a.channel_families,
        a.context, a.reference, a.definitions)


if __name__ == "__main__":
    main()
