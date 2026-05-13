# Verdict Distribution Analysis for Game of 24

## Goal

We compare batched and unbatched value evaluation on Gemini 2.5 Flash-Lite to test whether batching hurts ToT by making the evaluator over-optimistic or by corrupting candidate ranking.

## Method

We analyzed existing JSONL traces only. No new API calls were made. Since the current trace stores summed scores rather than raw verdict text, we approximate verdict patterns from scores using the mapping `sure = 20`, `likely = 1`, and `impossible = 0.001`.

## Key Findings
We analyzed the retained ToT trace candidates from Gemini 2.5 Flash-Lite under batched and unbatched value evaluation. The results suggest that batching does not mainly cause uniform over-optimism. Instead, batching introduces severe calibration and parsing instability. In the b=5 setting, batched evaluation had a median score of 0 at depth 1 and 65% missing verdicts, while unbatched evaluation had a median score of 22 and no missing verdicts. At depth 2, unbatched evaluation marked 71.2% of retained candidates as score 60, while batched evaluation marked only 5.2% as score 60. This indicates that the batched evaluator often fails to produce stable per-candidate judgments. The ranking comparison also shows large trajectory divergence: top-b overlap falls from 40.2% at depth 1 to 13.2% at depth 2 and 3.0% at depth 3, with same top-1 candidate rates below 15% at all depths. Therefore, batching appears to harm ToT not by making the evaluator blindly optimistic, but by corrupting candidate scoring and ranking, leading the search into different and weaker trajectories.
File address:
results/game24/runs/verdict_distribution_gemini25_b5.txt
results/game24/runs/verdict_distribution_gemini25_b1.txt
results/game24/runs/verdict_distribution_gemini25_b5.csv
results/game24/runs/verdict_distribution_gemini25_b1.csv