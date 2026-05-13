# Tree of Thoughts — Re-implementation & Extensions

A re-implementation of **Tree of Thoughts (ToT)** [Yao et al., NeurIPS 2023](https://arxiv.org/abs/2305.10601) with modern Gemini models, plus targeted extensions on the evaluator. Three tasks: **Game of 24** (BFS), **Mini Crosswords** (DFS + backtracking), and **Creative Writing** (plan/passage search with LLM-as-judge).

Built as a course project for CS 4782 (Cornell). Tested on **Gemini 3.1 Flash-Lite** (thinking), **Gemini 2.5 Flash-Lite** (non-thinking), and **Gemma 3 27B**.

---

## 1. Introduction

This repository is a re-implementation of *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., NeurIPS 2023). The original paper turns single-shot LLM reasoning into a search over intermediate "thoughts": the model proposes candidates, evaluates partial states via self-assessment, and explores with BFS or DFS. ToT's main contribution is showing that **explicit search + value-based pruning** can lift performance dramatically on tasks where one bad early step ruins the answer — most strikingly, raising Game of 24 accuracy from 7.3 % (IO) / 4 % (CoT) to **74 %** with ToT (b = 5) on GPT-4.

We reproduce the paper's three benchmark tasks on Gemini and extend the analysis along two axes the paper leaves unexplored:

1. **Evaluator design** — batched vs. unbatched prompting, focused candidate-level verdicts, gold-prompt re-verification, and a `(clue, constraint)` value cache for Crosswords.
2. **Selection mechanism** — vote-based ToT vs. independent scoring for Creative Writing.

---

## 2. Chosen Result

We targeted the original paper's headline numbers across all three tasks:

| Paper reference | Target | Paper (GPT-4) |
|-----------------|--------|--------------:|
| Table 2 | Game of 24, ToT (b = 5) success rate | **74 %** vs. IO 7.3 %, CoT 4 % |
| Table 3 | Mini Crosswords, ToT (DFS) — Letter / Word / Game | **78 / 60 / 20 %** |
| Figures 4–5 | Creative Writing, ToT vs. IO/CoT coherence | **ToT 7.56**, CoT 6.93, IO 6.19 |

These results collectively probe ToT under different search algorithms (BFS vs. DFS), thought granularities (equations vs. words vs. plans), and evaluator modes (value classification vs. voting), and demonstrate the framework's main claim: explicit search structure with LM-based evaluation outperforms single-path prompting on hard reasoning tasks.

---

## 3. GitHub Contents

```text
.
├── README.md, LICENSE, .gitignore
├── Tree of Thoughts.pdf            # Original paper (reference)
├── code/                           # All re-implementation source
│   ├── run.py                      # Unified CLI (24 / crosswords / text)
│   ├── analyze.py, rescore_24.py, reverify_24.py
│   ├── requirements.txt, conftest.py
│   ├── algorithms/                 # methods_24, methods_cw, methods_text
│   ├── core/model.py               # Async Gemini wrapper + rate limiter
│   ├── tasks/                      # State + I/O for each task
│   ├── prompts/                    # All prompt templates
│   ├── tests/                      # pytest unit tests
│   ├── scripts/{game24,crosswords,text}.sh + README.md
│   └── analysis/                   # Notebooks, scripts, figures, tables
├── data/                           # 24.csv, mini0505.json, text_inputs.json
├── results/                        # JSONL runs per task (+ logs, snapshots)
├── poster/Poster.pdf               # In-class poster
└── report/                         # Final report (PDF + .tex + .docx)
```


---

## 4. Re-implementation Details

**Models.** All three tasks run on Google Gemini through either AI Studio (free-tier) or Vertex AI (paid). Default: Gemini 3.1 Flash-Lite Preview (a "thinking" model, `thinking_level=MINIMAL` so the external ToT search drives reasoning, not internal CoT). Game of 24 ablations also cover Gemini 2.5 Flash-Lite (non-thinking) and Gemma 3 27B; Creative Writing uses Gemini 2.5 as generator with Gemini 3.1 as a fixed LLM-as-judge. **No fine-tuning** — all reasoning is prompt-driven at inference time.

**Datasets.**
- *Game of 24*: the original "100 hardest" subset (puzzles 901–1000) from `4nums.com`.
- *Mini Crosswords*: the paper's 20-puzzle subset (indices 0, 5, …, 95) from GooBix.
- *Creative Writing*: 100 four-sentence prompts (`data/text_inputs.json`).

**Tools.** Python 3.13, `google-genai` SDK, `asyncio` for concurrency, `tenacity` for retries, `sympy` for the Game of 24 verifier, `pytest`, `matplotlib` for analysis.

**Evaluation metrics.**
- *Game of 24*: exact symbolic verification (uses all four numbers once and evaluates to 24).
- *Mini Crosswords*: letter, word, and game accuracy averaged over up to 5 runs per config.
- *Creative Writing*: 1–10 LLM-as-judge coherence score (3 samples averaged).

**Key challenges / modifications.**
- The paper's per-state unbatched evaluator (~170 calls/puzzle for ToT b = 5) is implicit in the code, not the text. Our initial batched implementation silently degraded accuracy, motivating a systematic **batched vs. unbatched** comparison.
- For Crosswords we added a layered evaluator: `--batch {no, half, full}` × `{basic, focused, verified, cache}`, where *focused* asks for a single candidate-level verdict, *verified* re-confirms each kill via the gold prompt, and *cache* memoizes `(clue, constraint) → verdict` across siblings.
- For Game of 24, a zero-cost local-verifier tiebreaker re-picks the highest-scored *correct* depth-3 candidate from the existing trace; for Creative Writing, a `tot_score_select` variant replaces vote prompts with independent scoring.

---

## 5. Reproduction Steps

**Setup** (commands run from the repo root).

```bash
pip install -r code/requirements.txt
export GEMINI_API_KEY="your_key_here"        # or use --vertex with $GCP_PROJECT
mkdir -p data
curl -o data/24.csv       https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv
curl -o data/mini0505.json https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/crosswords/mini0505.json
pytest code/tests/                            # zero-API-cost sanity checks
```

**Quick start** — one config per task via the unified CLI:

```bash
# Game of 24 — ToT (b=5) unbatched evaluator (paper-strict)
python code/run.py --task 24 --method tot --b 5 --no_batch --n_puzzles 100

# Mini Crosswords — paper-strict unbatched ToT on the 20-puzzle paper split
python code/run.py --task crosswords --method tot --batch no --paper_split

# Mini Crosswords — our new Pareto leader: unbatched + value cache
python code/run.py --task crosswords --method tot --batch no --cache --paper_split

# Creative Writing — ToT vote + LLM-as-judge
python code/run.py --task text --method tot --n_puzzles 100 \
    --vertex --model gemini-2.5-flash-lite --thinking_level NONE
```

Bulk experiment drivers (each tee'd to `results/logs/<name>.log`):

```bash
bash code/scripts/game24.sh
bash code/scripts/crosswords.sh
bash code/scripts/text.sh
```

Each script accepts env-var overrides (`N_PUZZLES`, `MODEL`, `RPM`, `VERTEX=1`, `OVERWRITE=1`, etc.) — see [`code/scripts/README.md`](code/scripts/README.md) for the full list. All runs auto-resume; rerun the same command after a daily rate-limit hit and missing puzzles will be appended to the existing JSONL.

**Compute & cost.** No GPU required — this is API-driven. Free-tier Gemini Flash-Lite caps at 15 RPM and 1 000 RPD; expect ~7 min for IO/CoT and ~2 h for ToT (b = 5) per 100-puzzle Game of 24 run. Total Vertex spend across all our experiments was approximately **$15**.

---

## 6. Results / Insights

### Game of 24 (Table from paper: 74 % on GPT-4)

100 hardest puzzles (901–1000). Per-state unbatched ToT (b = 5) **exceeds the paper** on both Gemini models:

| Method | Gemini 3.1 | Gemini 2.5 | Gemma 3 27B | Paper (GPT-4) |
| :--- | :---: | :---: | :---: | :---: |
| **IO** | 16 % | 17 % | 14 % | 7.3 % |
| **CoT** | 84 % | 29 % | 32 % | 4.0 % |
| **ToT (b = 1) batched** | 65 % | 11 % | 9 % | — |
| **ToT (b = 5) batched** | 88 % | 15 % | 14 % | — |
| **ToT (b = 5) batched + rescore** | 96 % | 28 % | 33 % | — |
| **ToT (b = 1) unbatched** | 83 % | 53 % | — | 45 % |
| **ToT (b = 5) unbatched** | **96 %** | **87 %** | — | **74 %** |

*Insight.* Batching collapses Gemini 2.5 (87 → 15 %) but not Gemini 3.1 (96 → 88 %) — non-thinking models shift from absolute to competitive grading under multi-candidate prompts (66 % "impossible" verdicts vs. 0 % unbatched). The zero-cost rescore (trace-based local verifier tiebreaker) fully recovers Gemini 3.1 but only partially recovers Gemini 2.5, diagnosing that batching corrupts only the final selection on thinking models but the entire BFS trajectory on non-thinking ones.

### Mini Crosswords (Table 3 in paper: 78 / 60 / 20 %)

Paper split (20 puzzles, indices 0, 5, …, 95) on Gemini 3.1 Flash-Lite. Averaged over up to 5 runs.

| Configuration | Letter % | Word % | Game % (± SE) | API calls/puzzle | Steps |
| :--- | ---: | ---: | ---: | ---: | ---: |
| IO baseline | 45.4 | 18.3 | 0.0 | 10 | 1 |
| CoT baseline | 38.8 | 15.0 | 0.3 | 10 | 1 |
| ToT unbatched *(paper-strict)* | 77.4 | 59.1 | **33.0** ± 2.5 | 462 | 24.3 |
| → + best-state oracle | 80.0 | 61.3 | 33.0 | 462 | 24.3 |
| → − prune | 70.7 | 47.5 | 17.5 | 88 | 17.5 |
| → − backtrack | 39.3 | 23.5 | 5.0 | 26 | 3.6 |
| ToT full batch (basic) | 72.7 | 50.6 | 24.0 ± 2.4 | **158** | 26.9 |
| ToT half batch (basic) | 77.6 | 55.6 | 25.0 ± 1.6 | 414 | 37.2 |
| **ToT full + focused (k = 0)** | 74.8 | 54.0 | 27.5 ± 3.2 | 118 | 19.6 |
| **ToT half + focused (k = 7)** | 82.2 | 63.4 | 36.2 ± 3.8 | 421 | 29.6 |
| **ToT unbatched + cache** *(new Pareto leader)* | 77.1 | 61.5 | **40.0 ± 4.1** | **347** | 27.0 |

Paper reference (GPT-4): IO 38.7 / 14.0 / 0.0 %, CoT 40.6 / 15.6 / 1.0 %, ToT 78.0 / 60.0 / 20.0 %.

*Insights.*
- **Per-letter parity, nearly 2 × game accuracy** vs. the paper (33 vs. 20 % unbatched). Gemini's stronger word-level recall closes whole grids more often.
- **Search commits cleanly** — the `+best-state` oracle adds zero (33.0 → 33.0 %), unlike the paper's +15 pp jump. The deepest visited state *is* the final state.
- **Backtracking is the engine** — without it, accuracy collapses to 5 % game in 3.6 steps. Pruning halves the cost; removing it costs ~16 pp.
- **Focused candidate-level verdicts Pareto-dominate** per-clue scoring on both batch levels. `half_focused_k7` (36.2 %) matches unbatched at lower cost.
- **The value cache is the cheapest single accuracy win** — adding `--cache` to unbatched cuts value calls 38 %, runtime 23 %, and *adds* 7 pp game accuracy by regularizing noisy gold-prompt verdicts across siblings.
- **Verified re-confirmation regressed every focused configuration tested** — gold-prompt restoration tends to un-kill candidates the search should have pruned.

Full per-config table and plots: [`results/crosswords/analysis.md`](results/crosswords/analysis.md), [`code/analysis/crosswords/plots/`](code/analysis/crosswords/plots/).

### Creative Writing (Paper Figures 4–5)

100 four-sentence prompts. Generator Gemini 2.5 Flash-Lite, judge Gemini 3.1 Flash-Lite Preview (1–10 coherence).

| Method | Mean ± SEM | Insight |
| :--- | :---: | :--- |
| IO | 6.63 ± 0.21 | Direct one-shot baseline. |
| CoT | 6.50 ± 0.22 | Plan + passage in one prompt. |
| Standard ToT | 6.94 ± 0.21 | Best of the vote-based baselines. |
| ToT + refine | **7.14** ± 0.21 | Best of the refined-baseline group. |
| best-of-k CoT | 7.61 ± 0.20 | k = 5 samples, independent score-select. |
| best-of-k IO | 7.93 ± 0.19 | Sampling beats voting. |
| **`tot_score_select`** | **8.14** ± 0.19 | Independent scoring instead of vote prompts. |

Paper reference (GPT-4 judge): ToT 7.56, CoT 6.93, IO 6.19.

*Insight.* Vote-based ToT helps (6.94 vs. 6.63/6.50), and refinement helps a bit more. But the bigger story is that **score-based selection consistently beats vote-based planning** — `tot_score_select` is the headline at 8.14, and even `best-of-k IO` (no tree at all) reaches 7.93. For open-ended tasks, the search structure matters less than the quality of the candidate-selection mechanism.

---

## 7. Conclusion

The headline takeaway: ToT's value cleanly decomposes into a **search structure** (BFS/DFS with backtracking) and an **evaluation mechanism**, and which one dominates depends on the task.

- **Constrained tasks (Game of 24, Crosswords):** search dominates. Backtracking and pruning are essential — removing either collapses accuracy by 30+ pp. Beating GPT-4 on Game of 24 (96 % vs. 74 %) is largely a search-faithfulness win; the most important lesson was that the paper's per-state evaluator pattern is implicit in the code only.
- **Open-ended tasks (Creative Writing):** evaluation dominates. Vote-based ToT improves on IO/CoT, but independent score-selection improves much more — `tot_score_select` 8.14 vs. ToT 6.94. Tree structure helps less than reliable candidate scoring.
<!-- - **For Crosswords specifically:** focused candidate-level verdicts Pareto-dominate per-clue scoring at both batch levels, and a one-line `--cache` flag was the cheapest accuracy win in the entire study (40.0 % game / 347 calls / −23 % runtime). Verified re-confirmation, in contrast, regressed every configuration it touched. -->

Future ToT implementations should match evaluation strategy to task type rather than reuse the paper's vote-prompt template universally.

---

## 8. References

1. S. Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS 2023.
2. J. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022.
3. Z. Cheng et al., *Batch Prompting: Efficient Inference with LLM APIs*, EMNLP Industry Track 2023.
4. J. Lin et al., *BatchPrompt: Accomplish More with Less*, ICLR 2024.
5. M. Lee, P. Liang, Q. Yang, *CoAuthor: Designing a Human-AI Collaborative Writing Dataset*, CHI 2022.
6. ToT reference code — <https://github.com/princeton-nlp/tree-of-thought-llm>
7. Puzzle data — 4nums.com (Game of 24), GooBix (Mini Crosswords).
8. Google Cloud Vertex AI — Gemini 2.5 Flash-Lite, Gemini 3.1 Flash-Lite Preview.
9. SymPy — symbolic mathematics library for Python.

---

## 9. Acknowledgements

This work was completed as the final project for **CS 4782/5782: Introduction to Deep Learning (Cornell, Spring 2026)**, taught by **Profs. Killian Weinberger and Wei-Chiu Ma**. We thank the course staff for feedback during the proposal and poster sessions, and Princeton NLP for releasing the original ToT reference implementation and data splits.

**Team 78:** Minghan Gao (mg2328), Warren Hua (wsh48), Zhichun Zhang (zz547), Yihan Zhao (yz2788).