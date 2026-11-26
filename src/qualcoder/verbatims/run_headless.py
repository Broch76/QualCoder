# -*- coding: utf-8 -*-
"""End-to-end verbatim run, offline, outputs written locally.

python -m qualcoder.verbatims.run_headless --data DIR_WITH_voc_PKLS --out OUT_DIR
    [--backend lexical|transformer|precomputed] [--model-path DIR] [--npy FILE]
    [--codes CODE_FRAME.py] [--names NAMES.json] [--k K] [--months "August 2025,..."] [--model-in model.pkl]

Two-step use: run once (writes model.pkl); read topics.json and write NAMES.json ({topic id: theme name},
same name = merged); run again with --model-in OUT/model.pkl --names NAMES.json for the named tables.

Reads voc_<business>.pkl (columns: Response ID, QNet Promoter Score, Free Response, Month,
Interaction Channel). Writes CSV and JSON tables and METHODS.md. Raw comment text appears only in
topics.json (exemplars) and validation_sample.csv: keep OUT_DIR with the raw data.
"""
import argparse, csv, glob, json, os, pickle, time, warnings
import numpy as np
from .load import load_verbatims, merge_trails
from .topics import TopicModel
from . import stats, frame_check


def _csv(path, rows):
    if not rows:
        open(path, "w").close(); return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def run(data, out, backend="lexical", model_path=None, npy=None, codes=None, names=None, k=None, months=None,
        model_in=None):
    import pandas as pd
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Number of distinct clusters")
    os.makedirs(out, exist_ok=True)
    comments, trails = [], []
    for p in sorted(glob.glob(os.path.join(data, "voc_*.pkl"))):
        b = os.path.basename(p)[4:-4]
        c, tr = load_verbatims(pd.read_pickle(p), business=b, months=months)
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
    if names:
        tm.rename(json.load(open(names)))
    themes = tm.themes()
    res = {"trail": trail, "embedding": tm.embed_info, "k": tm.k, "k_grid": tm.k_grid, "seed_ari": tm.stability,
           "topics": tm.summary()}
    _csv(os.path.join(out, "prevalence_business.csv"), stats.prevalence(comments, themes, ("business",)))
    _csv(os.path.join(out, "prevalence_business_channel.csv"), stats.prevalence(comments, themes, ("business", "channel")))
    eff = stats.effects(comments, themes)
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
                "adjusted for business; within-business channel cells unadjusted. Benjamini-Hochberg across all %d tests; "
                "significant = q < 0.05 with both groups n >= 30. Cells under 30 are suppressed; 30 to 99 flagged caution.\n\n"
                % len(eff))
        f.write("Significant topic effects: %d.\n\nRun time %.1f s.\n" % (len(sig), time.time() - t0))
    return comments, tm, eff


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="lexical"); ap.add_argument("--model-path"); ap.add_argument("--npy")
    ap.add_argument("--codes"); ap.add_argument("--names"); ap.add_argument("--k", type=int)
    ap.add_argument("--months"); ap.add_argument("--model-in")
    a = ap.parse_args(argv)
    run(a.data, a.out, a.backend, a.model_path, a.npy, a.codes, a.names, a.k,
        [m.strip() for m in a.months.split(",")] if a.months else None, a.model_in)


if __name__ == "__main__":
    main()
