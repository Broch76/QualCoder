# -*- coding: utf-8 -*-
"""Optional Claude layer for the verbatim analysis.

Three jobs, each one the lexical pipeline does badly:
  1. name_topics:  turn each topic's words and exemplars into a short theme name, and say which topics
                   are the same theme (writes a NAMES.json the main run accepts).
  2. induce_codes: read the detractor comments that no keyword code matches and propose new codes
                   (name, definition, example ids). This replaces reading the uncoded topics by hand.
  3. code:         multi-label coding of every comment against one codebook (keyword-frame labels plus
                   induced codes). Handles paraphrase, typos, Afrikaans and isiZulu, and negation
                   ("did not have to wait" is not a delay). The result is a third theme set that goes
                   through the same statistics, and is compared with the keyword frame code by code.

Two transports, same requests:
  api      Anthropic Python SDK. Coding goes through the Message Batches API (half price, asynchronous).
           Needs ANTHROPIC_API_KEY (or another credential the SDK finds) and network to the API.
  handoff  No network or key needed here. Requests are written to OUT/llm/<job>.requests.jsonl; a
           Claude session (or any process with API access) answers them into
           OUT/llm/<job>.responses.jsonl, one {"custom_id", "output"} object per line; the next run
           reads them. Use it where the analysis machine cannot reach the API.

Before any text leaves the machine, identifiers are masked (redact): e-mail addresses, phone numbers,
13-digit ID numbers, long digit runs (member, policy and claim numbers) and URLs. Names in free text are
not masked; the datasets are pseudonymised at source.

Models: coding uses claude-haiku-5-5 (a bulk classification task); naming and induction use
claude-opus-5-5. Both can be changed with --llm-model-bulk / --llm-model-reason.
"""
import json
import os
import re
import time

MODEL_BULK = "claude-haiku-5-5"
MODEL_REASON = "claude-opus-5-5"
BATCH = 25            # comments per coding request
MAX_TOKENS = 8000

_RX = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[email]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[url]"),
    (re.compile(r"(?<!\d)\d{13}(?!\d)"), "[id]"),
    (re.compile(r"(?<![\w])(\+?27|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"), "[phone]"),
    (re.compile(r"(?<!\d)\d{6,}(?!\d)"), "[number]"),
]


def redact(text):
    t = str(text)
    for rx, rep in _RX:
        t = rx.sub(rep, t)
    return t


# ---------------------------------------------------------------- prompts and schemas

CODE_SYSTEM = """You code customer-survey comments from members of South African medical schemes.
Each comment answers "why did you give this score?" after a service interaction (call centre, walk-in
counter, e-mail, app or website). Comments may be in English, Afrikaans or isiZulu, and may contain typos.

For each comment, return every code from the codebook that the comment clearly expresses. Rules:
- Code what the member says happened, not words that merely appear. "I did not have to wait" is not a delay.
  "Nobody called me back" is no feedback, not a call-centre access problem.
- A comment can carry several codes, or none.
- Use "praise" for positive comments about the service or agent, "no_content" for blanks, "n/a", "none",
  single words or ratings with no reason, and "other" only for a substantive complaint or request that no
  code fits. When you use "other", give a 2 to 5 word English label for what it is about in "other_label".
- Read Afrikaans and isiZulu in the original language and code them like English.
Return one item per comment, using the comment's index."""

NAME_SYSTEM = """You name themes found by clustering customer-survey comments from medical-scheme members.
Each topic has its top words and the comments closest to its centre. Give each topic a short, specific
theme name in UK English (2 to 6 words, e.g. "Claim paid short", "Call-back never came"), as a member
would recognise it. Topics that describe the same theme must get exactly the same name; that is how they
are merged. Name a topic "Mixed or unclear" if its exemplars do not share a theme."""

INDUCE_SYSTEM = """You read detractor comments from medical-scheme members that an existing keyword code
frame did not match, and propose new codes for what the frame misses. Propose only themes that at least
three comments share; skip blanks, generic dissatisfaction and comments that are already covered by a code
in the existing frame but were missed by a keyword (list those separately as frame gaps). Each code gets a
snake_case id, a short label, a one-sentence definition precise enough for another coder to apply, and the
indices of the comments that carry it."""


def _obj(props, required=None):
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


def code_schema(code_ids):
    item = _obj({"i": {"type": "integer"},
                 "codes": {"type": "array", "items": {"type": "string", "enum": list(code_ids)}},
                 "other_label": {"type": "string"}}, ["i", "codes", "other_label"])
    return _obj({"items": {"type": "array", "items": item}})


NAME_SCHEMA = _obj({"topics": {"type": "array", "items": _obj({"topic": {"type": "integer"},
                                                                "name": {"type": "string"}})}})

INDUCE_SCHEMA = _obj({
    "new_codes": {"type": "array", "items": _obj({
        "id": {"type": "string"}, "label": {"type": "string"}, "definition": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "integer"}}})},
    "frame_gaps": {"type": "array", "items": _obj({
        "existing_code": {"type": "string"}, "pattern_missed": {"type": "string"},
        "examples": {"type": "array", "items": {"type": "integer"}}})},
})


def build_codebook(labels=None, induced=None):
    """labels: {code_id: label or (label, definition)} from the keyword frame; induced: list of dicts from
    induce_codes. Returns an ordered list of {"id", "label", "definition"}, with praise, other and
    no_content appended."""
    book = []
    for k, v in (labels or {}).items():
        lab, d = (v if isinstance(v, tuple) else (v, ""))
        book.append({"id": k, "label": lab, "definition": d})
    for c in induced or []:
        if c["id"] not in {b["id"] for b in book}:
            book.append({"id": c["id"], "label": c["label"], "definition": c.get("definition", "")})
    for k, lab in (("praise", "Positive about the service or agent"),
                   ("other", "Substantive point no code fits"),
                   ("no_content", "No reason given")):
        if k not in {b["id"] for b in book}:
            book.append({"id": k, "label": lab, "definition": ""})
    return book


def _codebook_text(book):
    return "\n".join("- %s: %s%s" % (b["id"], b["label"], (". " + b["definition"]) if b["definition"] else "")
                     for b in book)


def _req(custom_id, model, system, user, schema, effort):
    """custom_id carries a short hash of the request, so a cached answer is reused only for an identical
    request (a changed codebook, prompt or model is asked again)."""
    import hashlib
    h = hashlib.sha256(json.dumps([model, system, user, schema, effort], sort_keys=True).encode()).hexdigest()[:10]
    custom_id = "%s-%s" % (custom_id, h)
    return {"custom_id": custom_id, "params": {
        "model": model, "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}}}}


def coding_requests(comments, book, model=MODEL_BULK, batch=None):
    """One request per `batch` comments. custom_id = code-<first index>; indices are positions in
    `comments` so results map back without ids leaving the machine."""
    batch = batch or BATCH
    ids = [b["id"] for b in book]
    system = CODE_SYSTEM + "\n\nCodebook:\n" + _codebook_text(book)
    reqs = []
    for s in range(0, len(comments), batch):
        chunk = comments[s:s + batch]
        user = "Comments (index: text):\n" + "\n".join(
            "%d: %s" % (s + j, redact(c["text"]).replace("\n", " ")) for j, c in enumerate(chunk))
        reqs.append(_req("code-%d" % s, model, system, user, code_schema(ids), "low"))
    return reqs


def naming_request(topics, model=MODEL_REASON):
    lines = []
    for t in topics:
        lines.append("Topic %d (%d clauses). Words: %s\n  Exemplars:\n%s" % (
            int(t["topic"]), t.get("clauses", 0), ", ".join(t["words"][:12]),
            "\n".join("  - " + redact(e) for e in t["exemplars"][:6])))
    return _req("name-0", model, NAME_SYSTEM, "\n\n".join(lines), NAME_SCHEMA, "medium")


def induce_request(texts, book, model=MODEL_REASON):
    user = ("Existing code frame:\n" + _codebook_text(book) + "\n\nUncoded detractor comments (index: text):\n"
            + "\n".join("%d: %s" % (i, redact(t).replace("\n", " ")) for i, t in enumerate(texts)))
    return _req("induce-0", model, INDUCE_SYSTEM, user, INDUCE_SCHEMA, "high")


# ---------------------------------------------------------------- transports

class NeedResponses(RuntimeError):
    """Raised in handoff mode when a job's responses file is missing or incomplete."""


class Claude:
    def __init__(self, mode, out_dir, poll=30):
        if mode not in ("api", "handoff"):
            raise ValueError("mode must be api or handoff")
        self.mode, self.dir, self.poll = mode, os.path.join(out_dir, "llm"), poll
        os.makedirs(self.dir, exist_ok=True)
        self.log = []
        if mode == "api":
            import anthropic  # imported only when used
            self.client = anthropic.Anthropic()

    def run(self, job, requests, use_batch=True):
        """Returns {custom_id: parsed JSON output}. Results are cached in llm/<job>.responses.jsonl, so a
        re-run never pays twice for the same job."""
        resp_path = os.path.join(self.dir, job + ".responses.jsonl")
        with open(os.path.join(self.dir, job + ".requests.jsonl"), "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        got = self._read(resp_path)
        missing = [r for r in requests if r["custom_id"] not in got]
        if missing and self.mode == "handoff":
            raise NeedResponses("%s: %d of %d requests unanswered. Answer %s into %s and run again." % (
                job, len(missing), len(requests), job + ".requests.jsonl", resp_path))
        if missing:
            new = self._batch(missing) if (use_batch and len(missing) > 1) else self._direct(missing)
            with open(resp_path, "a", encoding="utf-8") as f:
                for cid, out in new.items():
                    f.write(json.dumps({"custom_id": cid, "output": out}, ensure_ascii=False) + "\n")
            got.update(new)
        return got

    @staticmethod
    def _read(path):
        got = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    out = d["output"]
                    got[d["custom_id"]] = json.loads(out) if isinstance(out, str) else out
        return got

    @staticmethod
    def _text(msg):
        if msg.stop_reason == "refusal":
            raise RuntimeError("request refused")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("output truncated at max_tokens")
        return json.loads(next(b.text for b in msg.content if b.type == "text"))

    def _direct(self, reqs):
        out = {}
        for r in reqs:
            msg = self.client.messages.create(**r["params"])
            out[r["custom_id"]] = self._text(msg)
            self.log.append({"custom_id": r["custom_id"], "model": msg.model,
                             "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens})
        return out

    def _batch(self, reqs):
        b = self.client.messages.batches.create(requests=reqs)
        while True:
            b = self.client.messages.batches.retrieve(b.id)
            if b.processing_status == "ended":
                break
            time.sleep(self.poll)
        out, failed = {}, []
        for res in self.client.messages.batches.results(b.id):
            if res.result.type == "succeeded":
                try:
                    out[res.custom_id] = self._text(res.result.message)
                except (RuntimeError, ValueError, StopIteration) as e:
                    failed.append((res.custom_id, str(e)))
            else:
                failed.append((res.custom_id, res.result.type))
        if failed:   # retry failures once, one at a time
            retry = [r for r in reqs if r["custom_id"] in {c for c, _ in failed}]
            out.update(self._direct(retry))
        self.log.append({"batch": b.id, "requests": len(reqs), "failed_first_pass": len(failed)})
        return out


# ---------------------------------------------------------------- jobs

def code(claude, comments, book, model=MODEL_BULK, job="code"):
    """Returns (themes, other_labels): a set of code ids per comment, and the "other" labels by index."""
    reqs = coding_requests(comments, book, model=model)
    res = claude.run(job, reqs)
    valid = {b["id"] for b in book}
    themes = [set() for _ in comments]
    other = {}
    seen = set()
    for r in reqs:
        for it in res[r["custom_id"]]["items"]:
            i = it["i"]
            if 0 <= i < len(comments):
                themes[i] = {c for c in it["codes"] if c in valid}
                seen.add(i)
                if "other" in themes[i] and it.get("other_label"):
                    other[i] = it["other_label"].strip().lower()
    unanswered = len(comments) - len(seen)
    return themes, other, unanswered


def name_topics(claude, topics, model=MODEL_REASON):
    r = naming_request(topics, model=model)
    res = claude.run("name", [r], use_batch=False)[r["custom_id"]]
    return {str(t["topic"]): t["name"].strip() for t in res["topics"]}


def induce_codes(claude, texts, book, model=MODEL_REASON, max_comments=400, seed=7):
    import random
    texts = list(texts)
    if len(texts) > max_comments:
        texts = random.Random(seed).sample(texts, max_comments)
    r = induce_request(texts, book, model=model)
    res = claude.run("induce", [r], use_batch=False)[r["custom_id"]]
    return res, texts


def agreement(comments, a_sets, b_sets, codes, group="detractor"):
    """Code-by-code agreement between two coders (e.g. keyword frame v Claude), on one NPS group.
    Precision and recall treat b as the reference."""
    from .frame_check import kappa
    idx = [i for i, c in enumerate(comments) if group is None or c["group"] == group]
    rows = []
    for k in codes:
        a = [k in a_sets[i] for i in idx]
        b = [k in b_sets[i] for i in idx]
        both = sum(x and y for x, y in zip(a, b))
        na, nb = sum(a), sum(b)
        rows.append({"code": k, "n_comments": len(idx), "n_keyword": na, "n_claude": nb, "n_both": both,
                     "keyword_precision_v_claude": round(both / na, 3) if na else "",
                     "keyword_recall_v_claude": round(both / nb, 3) if nb else "",
                     "kappa": round(kappa(a, b), 3)})
    return rows
