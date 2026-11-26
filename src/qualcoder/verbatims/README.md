# qualcoder.verbatims: survey comments by business, channel and NPS group

Headless and offline. Needs numpy, pandas and scipy; uses scikit-learn for k-means when installed.
Does not import Qt, matplotlib or networkx, and does not touch upstream QualCoder files.

## Run

```
cd src
python -m qualcoder.verbatims.run_headless --data DIR_WITH_voc_PKLS --out OUT --codes code_frame.py
# read OUT/topics.json, write names.json {"12": "Claims rejected", "30": "Claims rejected", ...}
python -m qualcoder.verbatims.run_headless --data DIR --out OUT2 --codes code_frame.py \
    --model-in OUT/model.pkl --names names.json
```

The code frame file defines `CODES = {name: regex}` or the `ALL = {name: (label, tier, regex, ...)}` layout.

## Outputs
- `prevalence_*.csv`: share of comments per theme, with Wilson 95% interval and base flag (suppress under 30, caution under 100)
- `effects_topics.csv`, `effects_keyword_codes.csv`: business v rest (adjusted for channel), channel v rest (adjusted
  for business), and channel cells within a business; Benjamini-Hochberg q across all tests
- `lift_topics.csv`: detractor v promoter
- `frame_check.csv`: per topic, the share covered by any keyword code; "frame misses this" below 50%
- `topics.json`: row-count trail, embedding backend, k grid, seed stability, topics with exemplars, and topics found
  among detractor comments that no keyword code matches (`uncoded_topics`)
- `validation_sample.csv`: 20 comments per topic for a hand check
- `METHODS.md`: method note with every base and setting

## Backends
- `lexical` (default, offline): TF-IDF unigrams and bigrams, truncated SVD. Joins comments that share words only.
  Use it to find what the keyword frame misses and to suggest codes, not to size themes.
- `transformer`: `--backend transformer --model-path LOCAL_MODEL_FOLDER` (copied in; no download is attempted).
- `precomputed`: `--backend precomputed --npy embeddings.npy`, rows aligned with the clause units.

Raw comment text appears only in `topics.json` and `validation_sample.csv`; keep the output folder with the raw data.
