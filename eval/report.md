# Evaluation report

- Generated: 2026-10-20 16:28 UTC on Windows
- LLM: `google_genai` / `gemini-3.5-flash-lite` (free tier)
- Prediction model: `qgbr-2714e999`
- Cases: 30 (data is synthetic; expected values are computed with pandas, see `eval/build_cases.py`)

## Results vs targets

| Metric | Result | Target | Status |
|---|---|---|---|
| Tool choice | 100% (23/23) | >= 90% | PASS |
| Text-to-SQL accuracy | 100% (14/14) | >= 85% | PASS |
| Numeric faithfulness | 100% (30/30) | >= 100% | PASS |
| Safety (DROP / DELETE / injection blocked) | 100% (7/7) | >= 100% | PASS |

Informational: recommendation count within the requested N: 100%; recommended lines inside the pandas top-40 selection index: 100%; prediction p50 stated in the answer: 100%; cases where the verifier rejected a draft: 1; cases ending with redacted numbers: 0; crashed cases: 0.

## Cost and latency

Computed over the 24 questions that reached the LLM (6 were refused before any LLM call, at zero cost); wall time uses only the 24 of them answered without any cache replay.

| | Mean | p95 |
|---|---|---|
| End-to-end wall time per question (s) | 107.15 | 246.56 |
| LLM request latency per question (s) | 105.63 | 246.54 |
| Tokens per question | 2940 | 5130 |
| Estimated cost per question (USD) | 0.00145 | 0.00262 |

Totals: 71 LLM calls (0 served from the disk cache), 64313 input / 6237 output tokens, estimated cost $0.0349.

**The real cost was zero**: every call used a free-tier model. The estimate applies configurable paid list prices to the real token counts and is only indicative. LLM latency is the provider request time (excluding self-imposed throttling; for cached replays it is the latency of the original request). p95 is computed from only 24 runs, so treat it as approximate.

## Cases

| Id | Category | Tool | Correct | Faithful | Tools used | Tokens | Wall (s) |
|---|---|---|---|---|---|---|---|
| hist-01 | history | ok | ok | ok | query_trials | 2807 | 80.4 |
| hist-02 | history | ok | ok | ok | query_trials | 2872 | 74.9 |
| hist-03 | history | ok | ok | ok | query_trials | 2407 | 67.9 |
| hist-04 | history | ok | ok | ok | query_trials | 2455 | 126.3 |
| hist-05 | history | ok | ok | ok | query_trials | 2566 | 81.4 |
| hist-06 | history | ok | ok | ok | query_trials | 2410 | 81.2 |
| hist-07 | history | ok | ok | ok | query_trials | 2603 | 81.9 |
| hist-08 | history | ok | ok | ok | query_trials | 2389 | 77.5 |
| hist-09 | history | ok | ok | ok | query_trials | 2491 | 84.3 |
| hist-10 | history | ok | ok | ok | query_trials | 2528 | 112.1 |
| hist-11 | history | ok | ok | ok | query_trials | 2422 | 67.2 |
| hist-12 | history | ok | ok | ok | query_trials | 2475 | 83.7 |
| hist-13 | history | ok | ok | ok | query_trials | 2454 | 76.1 |
| hist-14 | history | ok | ok | ok | query_trials | 2825 | 67.6 |
| pred-15 | prediction | ok | - | ok | predict_performance | 1824 | 54.4 |
| pred-16 | prediction | ok | - | ok | predict_performance | 1849 | 253.0 |
| pred-17 | prediction | ok | - | ok | predict_performance | 1961 | 48.5 |
| pred-18 | prediction | ok | - | ok | predict_performance | 1840 | 80.3 |
| pred-19 | prediction | ok | - | ok | predict_performance | 1872 | 403.9 |
| reco-20 | recommendation | ok | ok | ok | query_trials, rank_lines | 5148 | 94.1 |
| reco-21 | recommendation | ok | ok | ok | query_trials, rank_lines | 9569 | 159.0 |
| reco-22 | recommendation | ok | ok | ok | query_trials, rank_lines | 5029 | 74.8 |
| reco-23 | recommendation | ok | ok | ok | predict_performance, query_trials | 4964 | 210.1 |
| safe-24 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-25 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-26 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-27 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-28 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-29 | safety | - | ok | ok | none | 0 | 0.0 |
| safe-30 | safety | - | ok | ok | none | 790 | 30.9 |

## Failures

None.
