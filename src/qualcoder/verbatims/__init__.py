# -*- coding: utf-8 -*-
"""Survey verbatim analysis for QualCoder, headless (no Qt, no matplotlib, no network).

Built for NPS-survey free text coded by business, channel and NPS group:
- load:        clean survey rows into analysable comments with an auditable row-count trail
- text:        English-first tokenising with a stoplist that keeps negations and service words
- embed:       embedding backends (precomputed .npy, local sentence-transformers folder, offline lexical)
- topics:      clause-level clustering, k chosen by silhouette and seed stability, assign() for new data
- stats:       prevalence with Wilson intervals, small-base flags, channel-adjusted tests, Benjamini-Hochberg
- frame_check: topics against an existing keyword code frame (what the frame misses, where it disagrees)
- run_headless: end-to-end command line run writing local outputs and a methods note

Only numpy is required; scipy is used for p-values when present, scikit-learn for k-means when present.
This package does not import qualcoder.irastats or qualcoder.topic_model, which pull in plotting libraries.
"""
