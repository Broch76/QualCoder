"""Tests for qualcoder.verbatims (headless survey verbatim analysis)."""
import sys
import numpy as np
import pandas as pd
import pytest
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qualcoder.verbatims import load, stats, topics, frame_check, text
from qualcoder.verbatims.embed import LexicalEmbedder


def _frame():
    return pd.DataFrame({
        "Response ID": ["a", "b", "c", "d", "e", "f"],
        "QNet Promoter Score": [0, 5, 7, 9, "x", 10],
        "Free Response": ["Took 3 weeks, still waiting", "nan", "Skipped", "good", "app broken", "Consultant was helpful"],
        "Month": ["August 2025"] * 6,
        "Interaction Channel": ["telephone", "EMAIL", "EMAIL", "COUNTER CALLS", "EMAIL", "TELEPHONIC"],
    })


def test_loader_trail_counts_and_channel_map():
    comments, trail = load.load_verbatims(_frame(), business="A")
    steps = dict((s, (n, r)) for s, n, r in trail)
    assert steps["rows read"] == (6, 0)
    assert steps["no NPS score"] == (5, 1)
    assert steps["blank or skipped comment"] == (3, 2)
    assert [c["group"] for c in comments] == ["detractor", "promoter", "promoter"]
    assert comments[0]["channel"] == "TELEPHONIC"          # TELEPHONE mapped
    assert [c["no_content"] for c in comments] == [False, True, False]   # "good" has no content


def test_stoplist_keeps_negations_and_service_words():
    toks = text.tokenize("I was not called back, still waiting")
    for w in ("not", "called", "back", "still", "waiting"):
        assert w in toks


def test_wilson_interval():
    lo, hi = stats.wilson(20, 100)
    assert 0.13 < lo < 0.14 and 0.28 < hi < 0.29


def test_bh_is_monotone_and_bounded():
    q = stats.bh([0.01, 0.04, 0.03, 0.5])
    assert np.all(q <= 1) and q[0] <= q[2] <= q[1] <= q[3] + 1e-12


def test_mantel_haenszel_removes_channel_confound():
    rng = np.random.default_rng(0)
    n = 4000
    ch = rng.choice(["EMAIL", "PHONE"], size=n)
    # business B takes far more EMAIL; the theme depends on channel only
    bus = np.where(rng.random(n) < np.where(ch == "EMAIL", 0.7, 0.2), "B", "A")
    y = (rng.random(n) < np.where(ch == "EMAIL", 0.3, 0.05)).astype(int)
    e = (bus == "B").astype(int)
    raw_or, raw_p = stats.two_prop(y[e == 1].sum(), e.sum(), y[e == 0].sum(), (1 - e).sum())
    mh_or, _, mh_p = stats.mantel_haenszel(y, e, ch)
    assert raw_p < 1e-6 and raw_or > 1.5
    assert 0.75 < mh_or < 1.3 and mh_p > 0.01


def test_effects_flags_small_cells():
    comments = [{"business": "A", "channel": "X", "group": "detractor"}] * 10 + \
               [{"business": "B", "channel": "X", "group": "detractor"}] * 200
    themes = [{"t"}] * 10 + [set()] * 200
    rows = [r for r in stats.effects(comments, themes) if r["kind"] == "business" and r["group_a"] == "A"]
    assert rows[0]["flag"] == "suppress" and not rows[0]["significant"]


def test_numpy_kmeans_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 0.05, size=(50, 3)) for m in ([1, 0, 0], [0, 1, 0], [0, 0, 1])])
    lab, C = topics.kmeans(X, 3, seed=0)
    truth = np.repeat([0, 1, 2], 50)
    assert topics.adjusted_rand(lab, truth) > 0.95


def test_lexical_embedding_is_deterministic():
    texts = ["the app keeps crashing", "claim was rejected", "app login fails", "claim not paid", "slow service",
             "claim paid short", "app is slow"] * 3
    a = LexicalEmbedder(dim=5).fit_transform(texts)
    b = LexicalEmbedder(dim=5).fit_transform(texts)
    assert np.allclose(a, b)


def test_topic_model_multilabel_and_frame_check():
    base = ["my claim was rejected", "the app keeps crashing", "they transferred me three times"]
    comments = []
    for i in range(60):
        t = base[i % 3]
        if i % 10 == 0:
            t = base[0] + ". " + base[1]
        comments.append({"id": str(i), "business": "A" if i % 2 else "B", "channel": "X", "group": "detractor",
                         "text": t, "no_content": False})
    tm = topics.TopicModel(k=3).fit(comments, balance=False)
    assert any(len(s) == 2 for s in tm.comment_topics)          # two-theme comments get two topics
    codes = {"claims": __import__("re").compile("claim")}
    cs = frame_check.apply_codes(comments, codes)
    rows = frame_check.topic_vs_codes(comments, tm.themes(), cs)
    assert sum(r["status"] == "frame misses this" for r in rows) == 2   # app and transfer topics are uncoded


# ---- Claude layer (no network: handoff files and a fake client) ----

def test_redact_masks_identifiers():
    from qualcoder.verbatims.llm import redact
    t = redact("mail me a.b@x.co.za or 082 555 1234, ID 8001015009087, member 12345678, see https://x.y/z")
    assert "@" not in t and "555" not in t and "8001015009087" not in t and "12345678" not in t and "https" not in t
    assert redact("waited 3 weeks") == "waited 3 weeks"


def _mini_comments():
    rows = [("the claim was paid short", 2), ("nobody called me back", 3), ("baie stadig", 1),
            ("excellent service", 10), ("I did not have to wait", 9)]
    return [{"id": str(i), "business": "A", "channel": "EMAIL", "month": "m", "score": s,
             "group": "detractor" if s <= 6 else "promoter", "text": t, "no_content": False}
            for i, (t, s) in enumerate(rows)]


def test_handoff_round_trip(tmp_path):
    import json, pytest
    from qualcoder.verbatims import llm
    cm = _mini_comments()
    book = llm.build_codebook({"claims": "Claims", "delay": "Took too long", "no_feedback": "No feedback"})
    assert [b["id"] for b in book][-3:] == ["praise", "other", "no_content"]
    c = llm.Claude("handoff", str(tmp_path))
    with pytest.raises(llm.NeedResponses):
        llm.code(c, cm, book)
    reqs = [json.loads(l) for l in open(tmp_path / "llm" / "code.requests.jsonl")]
    assert len(reqs) == 1 and reqs[0]["params"]["output_config"]["format"]["type"] == "json_schema"
    enum = reqs[0]["params"]["output_config"]["format"]["schema"]["properties"]["items"]["items"]["properties"]["codes"]["items"]["enum"]
    assert "claims" in enum and "praise" in enum
    ans = {"items": [{"i": 0, "codes": ["claims"], "other_label": ""}, {"i": 1, "codes": ["no_feedback"], "other_label": ""},
                     {"i": 2, "codes": ["delay"], "other_label": ""}, {"i": 3, "codes": ["praise", "bogus"], "other_label": ""},
                     {"i": 4, "codes": ["praise"], "other_label": ""}]}
    with open(tmp_path / "llm" / "code.responses.jsonl", "w") as f:
        f.write(json.dumps({"custom_id": reqs[0]["custom_id"], "output": json.dumps(ans)}) + "\n")
    th, other, missing = llm.code(c, cm, book)
    assert th[0] == {"claims"} and th[2] == {"delay"} and th[3] == {"praise"} and missing == 0
    rows = llm.agreement(cm, [{"claims"}, set(), set(), set(), {"delay"}], th, ["claims", "delay"])
    d = {r["code"]: r for r in rows}
    assert d["claims"]["n_both"] == 1 and d["delay"]["n_claude"] == 1


def test_api_batch_path_with_fake_client(tmp_path):
    import json, types
    from qualcoder.verbatims import llm
    cm = _mini_comments()
    book = llm.build_codebook({"claims": "Claims"})

    def msg(obj):
        return types.SimpleNamespace(stop_reason="end_turn", model="m",
                                     content=[types.SimpleNamespace(type="text", text=json.dumps(obj))],
                                     usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))

    class Batches:
        def create(self, requests):
            self.reqs = requests; return types.SimpleNamespace(id="b1", processing_status="in_progress")
        def retrieve(self, bid):
            return types.SimpleNamespace(id=bid, processing_status="ended")
        def results(self, bid):
            for r in self.reqs:
                text = r["params"]["messages"][0]["content"]
                if not isinstance(text, str):
                    text = text[0]["text"]
                idx = [int(l.split(":")[0]) for l in text.splitlines()[1:]]
                items = [{"i": i, "codes": ["claims"] if i == 0 else [], "other_label": ""} for i in idx]
                yield types.SimpleNamespace(custom_id=r["custom_id"],
                                            result=types.SimpleNamespace(type="succeeded", message=msg({"items": items})))

    c = llm.Claude.__new__(llm.Claude)
    c.mode, c.dir, c.poll, c.log = "api", str(tmp_path), 0, []
    c.models = {}
    c.client = types.SimpleNamespace(messages=types.SimpleNamespace(batches=Batches(), create=None))
    old = llm.BATCH
    try:
        llm.BATCH = 2
        th, _, missing = llm.code(c, cm, book)
    finally:
        llm.BATCH = old
    assert th[0] == {"claims"} and missing == 0
    # second call is served from the cache file, no client needed
    c.client = None
    llm.BATCH = 2
    try:
        th2, _, _ = llm.code(c, cm, book)
    finally:
        llm.BATCH = old
    assert th2 == th


def test_effects_direction_empty_group_and_sparse_cells():
    # Business A: 0 of 20 on EMAIL v B 1 of 200: no odds ratio shown, direction from shares, not significant.
    cm, th = [], []
    for b, ch, n, x in (("A", "EMAIL", 20, 0), ("B", "EMAIL", 200, 1), ("A", "PHONE", 50, 5), ("B", "PHONE", 50, 5)):
        for k in range(n):
            cm.append({"business": b, "channel": ch, "group": "detractor"}); th.append({"t"} if k < x else set())
    rows = stats.effects(cm, th)
    cell = [r for r in rows if r["kind"] == "cell" and r["group_a"] == "A EMAIL"][0]
    assert cell["odds_ratio"] != cell["odds_ratio"] and cell["direction"] == "lower"
    assert cell["test"] == "Fisher exact" and not cell["significant"]
    bus = [r for r in rows if r["kind"] == "business" and r["group_a"] == "A"][0]
    assert bus["direction"] in ("higher", "lower", "same") and "direction_disagree" in bus


def test_effects_direction_follows_adjusted_ratio():
    # Simpson set-up: A has the higher rate within each channel but sits mostly in the low-rate channel.
    cm, th = [], []
    for b, ch, n, x in (("A", "LOW", 180, 18), ("A", "HIGH", 20, 12), ("B", "LOW", 20, 1), ("B", "HIGH", 180, 90)):
        for k in range(n):
            cm.append({"business": b, "channel": ch, "group": "detractor"}); th.append({"t"} if k < x else set())
    r = [r for r in stats.effects(cm, th) if r["kind"] == "business" and r["group_a"] == "A"][0]
    assert r["pct_a"] < r["pct_b"] and r["odds_ratio"] > 1
    assert r["direction"] == "higher" and r["direction_disagree"]


def test_codebook_text_hides_tier():
    from qualcoder.verbatims import llm
    book = llm.build_codebook({"claims": "Claims"}, meta={"claims": {"group": "topic", "tier": "Health controls"}})
    t = llm._codebook_text(book)
    assert "topic" in t and "Health controls" not in t


def test_reference_tier_check_paired(tmp_path):
    from qualcoder.verbatims import run_headless as rh
    book = [{"id": "a", "tier": "Health controls"}, {"id": "b", "tier": "Scheme board decides"}]
    cc = [{"id": str(i)} for i in range(6)]
    kw = [{"a"}, {"a"}, {"b"}, set(), {"a"}, {"a"}]
    cl = [{"a"}, {"a"}, {"b"}, {"b"}, {"b"}, set()]
    ref = tmp_path / "r.csv"
    ref.write_text("id,tier\n0,Health controls\n1,Health controls\n2,Scheme board decides\n"
                   "3,Scheme board decides\n4,Health controls\n5,neither\n")
    rows = rh._reference_tier_check(cc, str(ref), kw, cl, book)
    main = {r["coder"]: r for r in rows if "table" not in r}
    assert main["claude"]["agree"] == 5 and main["keyword"]["agree"] == 4
    paired = [r for r in rows if r.get("table") == "paired" and r["coder"] == "claude v keyword"][0]
    assert (paired["both_right"], paired["only_claude"], paired["only_keyword"], paired["both_wrong"]) == (3, 2, 1, 0)
    assert any(r.get("table") == "confusion" for r in rows)


def test_tier_table_partitions_untiered():
    from qualcoder.verbatims import run_headless as rh
    book = [{"id": "unresolved", "tier": "Health controls"}, {"id": "x"}]
    cc = [{"business": "A", "group": "detractor"} for _ in range(5)]
    sets = [{"unresolved"}, {"x"}, {"praise"}, {"no_content"}, set()]
    rows = {r["tier"]: r["n"] for r in rh._tier_table(cc, sets, book)}
    assert rows["Health controls"] == 1 and rows["Health controls (without unresolved)"] == 0
    assert (rows["no tier code"], rows["praise only"], rows["no_content"], rows["nothing"]) == (1, 1, 1, 1)


def test_redact_spelled_out_email_keeps_ordinary_text():
    from qualcoder.verbatims.llm import redact
    assert redact("not resolved at all. Nobody phoned") == "not resolved at all. Nobody phoned"
    assert redact("No help at all. They never called back") == "No help at all. They never called back"
    assert redact("mail john at gmail dot com please") == "mail [email] please"
    assert redact("john (at) mweb (dot) co (dot) za") == "[email]"
    assert redact("R150000 owed") == "[amount] owed"
