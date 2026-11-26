"""Tests for qualcoder.verbatims (headless survey verbatim analysis)."""
import sys
import numpy as np
import pandas as pd
import pytest

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
                s = int(r["custom_id"].split("-")[1])
                items = [{"i": s + j, "codes": ["claims"] if s + j == 0 else [], "other_label": ""} for j in range(2)]
                yield types.SimpleNamespace(custom_id=r["custom_id"],
                                            result=types.SimpleNamespace(type="succeeded", message=msg({"items": items})))

    c = llm.Claude.__new__(llm.Claude)
    c.mode, c.dir, c.poll, c.log = "api", str(tmp_path), 0, []
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
