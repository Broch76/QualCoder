# -*- coding: utf-8 -*-
"""Tokenising and stopwords for short English service comments (with some Afrikaans and Zulu).

The stoplist is deliberately small: it keeps negations (not, no, never, nobody), time words
(still, again, back, long, weeks) and service words (call, help, problem, website, member),
because those carry the delay and repeat-contact signals in NPS comments.
"""
import re

TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
CLAUSE_RE = re.compile(r"[.!?;\n]+|\s+-\s+|,\s*(?:but|and then|however)\s+|\s+but\s+", re.IGNORECASE)

STOP_EN = set("""
a an the this that these those it its it's i i'm i've i'd me my mine myself we us our ours you your yours he him his she her
hers they them their theirs is am are was were be been being have has had having do does did doing will would shall should
can could may might must of in on at by for with about into onto from to up down out off over under than then so very too
just also only really quite as if or because while which who whom what when where how there here such some any each other
same own both all more most much many am also yet ok okay oh hi hello dear sir madam please regards kind
""".split())
STOP_AF = set("""die en van in is het nie dat ek my jy jou hy sy ons hulle om te met vir op aan as wat was sal kan word ook maar
by na of so dit 'n n daar hier""".split())
STOP_ZU = set("""futhi ukuthi kodwa noma uma lapho ngoba kanye kuyo kubo yena bona thina mina wena""".split())
# Comments with no analysable content once these words are removed go to a counted "no content" bucket.
GENERIC = set("""good great fine nice excellent bad poor average service thanks thank you none nothing na n/a no comment comments
ok okay yes all well done keep it up""".split())


def normalise(text):
    """Lower case, straighten curly quotes, collapse spaces."""
    t = str(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def tokenize(text, stopwords=None):
    stop = STOP_EN | STOP_AF | STOP_ZU if stopwords is None else stopwords
    return [t for t in TOKEN_RE.findall(normalise(text)) if t not in stop and len(t) > 1]


def split_clauses(text, min_tokens=2):
    """Split a comment into clauses (sentence ends, 'but', ', and then'). Very short pieces are
    merged into the previous clause so 'Took weeks. Terrible.' stays one unit."""
    parts = [p.strip() for p in CLAUSE_RE.split(str(text)) if p and p.strip()]
    out = []
    for p in parts:
        if out and len(tokenize(p)) < min_tokens:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [str(text).strip()]


def is_no_content(text):
    toks = TOKEN_RE.findall(normalise(text))
    return len([t for t in toks if t not in STOP_EN and t not in GENERIC]) == 0
