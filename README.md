# Tree of Thoughts — Game of 24, Mini Crosswords & Creative Writing

An optimized re-implementation of the **Tree of Thoughts (ToT)** framework [Yao et al. (NeurIPS 2023)](https://arxiv.org/abs/2305.10601) for solving complex reasoning tasks. This project currently supports **Game of 24** using BFS search, **Mini Crosswords** using DFS with backtracking, and **Creative Writing** using IO, CoT, best-of-k, ToT-style planning/voting, scoring, and refinement.

The implementation is tested on **Gemini 3.1 Flash-Lite** (thinking), **Gemini 2.5 Flash-Lite** (no thinking), and **Gemma 3 27B**.

---

## 🚀 Key Features

- **Multi-task architecture**: Decouples domain logic in `tasks/` from search algorithms in `algorithms/`.
- **Asynchronous concurrency**: Uses `asyncio` to handle many LLM requests while respecting rate limits.
- **Cost optimization**: Supports **batched evaluation** to reduce value-evaluation API calls by packing multiple candidates into one prompt.
- **Resumable experiments**: Output JSONL files are append-only; rerunning the same command skips puzzles that are already completed.
- **Anti-hallucination guards**: Uses strict parsing and validation, including regex-based validation for Crosswords and exact symbolic verification for Game of 24.
- **Creative writing extension**: Adds open-ended text generation experiments comparing IO, CoT, best-of-k, ToT-style plan selection, score selection, and refinement.

---

## ⚙️ Setup

1. **Install dependencies**:

   ```bash
   pip install -r code/requirements.txt
   ```

2. **Set API key**:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Download data**:

   Download `24.csv` and `mini0505.json` from the original repository and place them in `data/`. The creative writing task uses `data/text_inputs.json`.

   ```bash
   mkdir -p data
   curl -o data/24.csv https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv
   curl -o data/mini0505.json https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/crosswords/mini0505.json
   ```

---

## Read This Before Launching a Long Run

The free-tier Gemini API caps Flash-Lite at **15 requests/minute** and **1000 requests/day**. The value evaluator can be batched, so call counts are much lower than a naive ToT implementation.

| Method | Calls / puzzle | At 15 RPM | 100 puzzles | Free-tier puzzles/day |
|--------|---------------:|----------:|------------:|----------------------:|
| IO | 1 | 4 s | 7 min | 1000 |
| CoT | 1 | 4 s | 7 min | 1000 |
| ToT (b=1) | ~12 | 50 s | 80 min | ~83 |
| ToT (b=5) | ~20 | 80 s | 2.2 h | ~50 |

The script auto-resumes across days. Pick a stable `--out` path, rerun the same command, and the driver will append to the JSONL while skipping puzzles that are already done. Use `--overwrite` to restart from scratch.

If you want to skip the daily-cap workflow, paid Tier 1 raises Flash-Lite rate limits substantially and removes the daily cap. Pass `--rpm 0` to disable the local limiter.

---

## 🧩 Task 1: Game of 24 (BFS)

Game of 24 is solved using Breadth-First Search (BFS). We test three models via Google's Vertex AI (a cloud API for Gemini models, billed per token): Gemini 3.1 Flash-Lite (a thinking model with internal chain-of-thought), Gemini 2.5 Flash-Lite (same family, no thinking), and Gemma 3 27B (open-weight, run free via Google AI Studio). No models are fine-tuned — all reasoning is driven by ToT's prompts at inference time. The model proposes arithmetic operations, the search evaluates and keeps the top states under a beam width b, and a symbolic verifier checks whether the final equation uses all four numbers exactly once and evaluates to 24.

### Evaluator Architectures
To optimize API costs while maintaining accuracy, this implementation supports two value evaluator modes for the Game of 24:

- **Unbatched** (`--no_batch`): The strict paper implementation. Evaluates each candidate state in a separate API call. Highly accurate but consumes significantly more API quota.

- **Batched** (`Default`): Packs all candidate states at the current BFS depth into a single prompt. This dramatically reduces API calls (e.g., from ~170 to ~20 calls per puzzle for b=5) and is the recommended default for fast, cost-effective experiments.

### Usage

```bash
# 0. Sanity-check the exact-math verifier locally (Zero API cost)
pytest code/tests/

# 1. IO and CoT Baselines
python code/run.py --task 24 --method io --n_puzzles 100
python code/run.py --task 24 --method cot --n_puzzles 100

# 2. ToT: Batched Evaluator (Cost-optimized, recommended default)
python code/run.py --task 24 --method tot --b 1 --n_puzzles 100
python code/run.py --task 24 --method tot --b 5 --n_puzzles 100

# 3. ToT: Unbatched Evaluator (Strict paper replication)
python code/run.py --task 24 --method tot --b 1 --no_batch --n_puzzles 100
python code/run.py --task 24 --method tot --b 5 --no_batch --n_puzzles 100

# 4. Post-processing: Rescore traces (Recover failed searches using trajectory history)
python code/rescore_24.py results/game24/runs/gem31_tot_b5_batched.jsonl

# 5. Summarize and format results
python code/analyze.py results/game24/runs/*.jsonl
```

Each run writes a JSONL file to `results/game24/runs/<name>.jsonl`, one JSON object per puzzle:

```text
{idx, inputs, equation, verified, elapsed_s, trace}
```

Unbatched runs use a `_unbatched` suffix to avoid collisions.

### Results: Game of 24

Tested on the original "100 hardest" subset, puzzles 901–1000.

| Method | Gemini 3.1 Flash-Lite | Gemini 2.5 Flash-Lite | Gemma 3 27B | Paper (GPT-4) |
| :--- | :---: | :---: | :---: | :---: |
| **IO** | 16% | 17% | 14% | 7.3% |
| **CoT** | 84% | 29% | 32% | 4.0% |
| **ToT (b=1) batched** | 65% | 11% | 9% | — |
| **ToT (b=5) batched** | 88% | 15% | 14% | — |
| **ToT (b=5) batched + rescore** | 96% | 28% | 33% | — |
| **ToT (b=1) unbatched** | 83% | 53% | — | 45% |
| **ToT (b=5) unbatched** | **96%** | **87%** | — | **74%** |

With per-state unbatched evaluation, ToT exceeds the paper's GPT-4 benchmark on both Gemini models: **96%** on Gemini 3.1 and **87%** on Gemini 2.5, compared with **74%** in the paper. Total Vertex cost across all Gemini experiments was about **$8.03**.

---

## 🔠 Task 2: Mini Crosswords (DFS)

Mini Crosswords are solved using Depth-First Search (DFS) with backtracking. The solver incrementally fills a 5x5 grid, validates candidate words against intersecting grid constraints using regex, and filters out malformed or hallucinated outputs before they can corrupt the search state.

Unlike Game of 24, Crosswords require spatial 2D reasoning. Standard autoregressive generation (IO/CoT) struggles significantly here due to the "anchoring" effect—once a model hallucinates an incorrect horizontal word, it cannot easily backtrack, which cascades errors into the intersecting vertical words.

### Evaluator Architectures
The crossword value evaluator is configured along two orthogonal axes: **call shape** (`--batch`) and **composable features** (`--focused`, `--verified`, `--cache`). All evaluators iterate clues in **declaration order** (`h1..h5, v1..v5`); only `--focused --top_k N` selects which clues participate, but the prompt still renders them in declaration order.

**Call shape (`--batch {no, half, full}`):**

- `no` — Paper-strict. One API call per testable clue, per sibling. Most accurate, most expensive. No cache.
- `half` — One API call per sibling, all of its testable clues batched in.
- `full` (default) — One API call per DFS step: all siblings × all testable clues batched in. Cheapest.

**Composable features (apply on top of `half` / `full`; rejected with `no`):**

- `--focused` — Single candidate-level verdict (`viable` / `not_viable`) per sibling instead of per-clue verdicts. Without `--top_k`, every testable clue is included. With `--top_k N` (>0), the N most-constrained clues are selected and rendered in declaration order. Falls back to the basic variant if every sibling at a step is judged not_viable.
- `--verified` — Re-confirm every kill against the unbatched `value_prompt`; only confirmed impossibles prune. Surviving kills are gold-prompt-confirmed.
- `--cache` — Opt-in per-puzzle `(clue, constraint) → verdict` cache. Only verified stage 2 writes it (the gold prompt is authoritative); focused reads it to short-circuit candidates whose clue is already cached as `impossible`. Useful when DFS revisits the same partially-filled board across backtracks.

The flags compose: `--focused --verified --cache` runs focused with cache short-circuit, verifies each kill via the gold prompt, and stores those gold verdicts back into the cache for future short-circuits.

The selected combination becomes the run's evaluator name (and JSONL filename suffix):

| Evaluator                       | `--batch` | `--focused` | `--verified` | `--cache` |
|---------------------------------|-----------|:-----------:|:------------:|:---------:|
| `unbatched`                     | `no`      |             |              |           |
| `full_basic` (default)          | `full`    |             |              |           |
| `full_focused`                  | `full`    | ✓           |              |           |
| `full_verified`                 | `full`    |             | ✓            |           |
| `full_cache`                    | `full`    |             |              | ✓         |
| `full_focused_verified`         | `full`    | ✓           | ✓            |           |
| `full_focused_verified_cache`   | `full`    | ✓           | ✓            | ✓         |
| `half_basic`                    | `half`    |             |              |           |
| `half_focused`                  | `half`    | ✓           |              |           |
| `half_verified`                 | `half`    |             | ✓            |           |
| `half_focused_verified`         | `half`    | ✓           | ✓            |           |
| `half_focused_verified_cache`   | `half`    | ✓           | ✓            | ✓         |

(other subsets follow the same `<level>_<flags>` naming.)

### Usage
Use the `--paper_split` flag to run experiments strictly on the 20-puzzle testing subset used in the original NeurIPS paper (indices 0, 5, 10...95).

```bash
# IO and CoT Baselines (Automatically averages over 10 samples per puzzle)
python code/run.py --task crosswords --method io  --n_puzzles 100 --paper_split
python code/run.py --task crosswords --method cot --n_puzzles 100 --paper_split

# ToT: Paper-strict per-clue evaluator (slowest, most accurate)
python code/run.py --task crosswords --method tot --batch no --max_steps 100 --paper_split

# ToT: Full-batch focused (top-3) + verified + cache
python code/run.py --task crosswords --method tot --batch full \
    --focused --verified --cache --top_k 3 --paper_split

# ToT: Half-batch verified (one call per sibling, kills reconfirmed)
python code/run.py --task crosswords --method tot --batch half --verified --paper_split

# Ablations: Disable Pruning or Backtracking
python code/run.py --task crosswords --method tot --batch full --no_prune --paper_split
python code/run.py --task crosswords --method tot --batch full --no_backtrack --paper_split

# Analyze Results: Generate performance tables (Letter, Word, Game Accuracy & Steps).
# The "+ best state" oracle metric is reconstructed from each run's DFS trace at analysis time
python code/analyze.py results/crosswords/runs/*.jsonl
```

### Results: Mini Crosswords
Tested on the original 20-puzzle subset (indices 0, 5, 10...95) using `gemini-3.1-flash-lite-preview`. Baseline scores (IO/CoT) represent the average across 10 independent samples per puzzle.

*Note: The original paper utilized an early 2023 version of GPT-4. Our reproduction compares these results against Gemini 3.1 Flash-Lite to evaluate the effectiveness of Tree of Thoughts on modern fast/lite reasoning models.*


<table>
  <thead>
    <tr>
      <th rowspan="2">Avg. Success Rate (%)<br>of 5 Runs<br>of 20 Puzzle Games</th>
      <th colspan="2" style="border-left: 2px solid #999;">Letter Acc.</th>
      <th colspan="2" style="border-left: 2px solid #999;">Word Acc.</th>
      <th colspan="2" style="border-left: 2px solid #999;">Game Acc.</th>
      <th rowspan="2" style="border-left: 2px solid #999;">Avg. Steps<br>(Ours)</th>
      <th rowspan="2" style="border-left: 2px solid #999;">Avg. API Calls<br>(Ours)</th>
    </tr>
    <tr>
      <th style="border-left: 2px solid #999;">Paper<br>(GPT-4)</th>
      <th>Ours<br>(Gemini 3.1)</th>
      <th style="border-left: 2px solid #999;">Paper</th>
      <th>Ours</th>
      <th style="border-left: 2px solid #999;">Paper</th>
      <th>Ours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>IO</strong></td>
      <td style="border-left: 2px solid #999;">38.7%</td>
      <td><strong>45.9%<strong></td>
      <td style="border-left: 2px solid #999;">14.0%</td>
      <td><strong>18.5%<strong></td>
      <td style="border-left: 2px solid #999;">0.0%</td>
      <td><strong>0.0%<strong></td>
      <td style="border-left: 2px solid #999;">1.0</td>
      <td style="border-left: 2px solid #999;">1.0</td>
    </tr>
    <tr>
      <td><strong>CoT</strong></td>
      <td style="border-left: 2px solid #999;">40.6%</td>
      <td><strong>38.8%<strong></td>
      <td style="border-left: 2px solid #999;">15.6%</td>
      <td><strong>15.2%<strong></td>
      <td style="border-left: 2px solid #999;">1.0%</td>
      <td><strong>1.0%<strong></td>
      <td style="border-left: 2px solid #999;">1.0</td>
      <td style="border-left: 2px solid #999;">1.0</td>
    </tr>
    <tr>
      <td><strong>ToT (unbatched)</strong></td>
      <td style="border-left: 2px solid #999;">78.0%</td>
      <td><strong>78.8%</strong></td>
      <td style="border-left: 2px solid #999;">60.0%</td>
      <td><strong>61.8%</strong></td>
      <td style="border-left: 2px solid #999;">20.0%</td>
      <td><strong>37.2%</strong></td>
      <td style="border-left: 2px solid #999;">27.8</td>
      <td style="border-left: 2px solid #999;">499.7</td>
    </tr>
<tr style="background-color: #f2f2f2;">
  <td><em>→ + best state</em></td>
  <td style="border-left: 2px solid #999;">82.4%</td>
  <td>81.2%</td>
  <td style="border-left: 2px solid #999;">67.5%</td>
  <td>64.0%</td>
  <td style="border-left: 2px solid #999;">35.0%</td>
  <td>37.1%</td>
  <td style="border-left: 2px solid #999;">19.9</td>
  <td style="border-left: 2px solid #999;">225.68</td>
</tr>

<tr style="background-color: #f2f2f2;">
  <td><em>→ - prune</em></td>
  <td style="border-left: 2px solid #999;">65.4%</td>
  <td>69.13%</td>
  <td style="border-left: 2px solid #999;">41.5%</td>
  <td>46.5%</td>
  <td style="border-left: 2px solid #999;">5.0%</td>
  <td>20.0%</td>
  <td style="border-left: 2px solid #999;">18.6</td>
  <td style="border-left: 2px solid #999;">103.25</td>
</tr>

<tr style="background-color: #f2f2f2;">
  <td><em>→ - backtrack</em></td>
  <td style="border-left: 2px solid #999;">54.6%</td>
  <td>37.5%</td>
  <td style="border-left: 2px solid #999;">20.0%</td>
  <td>22.2%</td>
  <td style="border-left: 2px solid #999;">5.0%</td>
  <td>3.3%</td>
  <td style="border-left: 2px solid #999;">3.5</td>
  <td style="border-left: 2px solid #999;">26.0</td>
</tr>
  </tbody>
</table>

### Cache Ablation: Unbatched vs. Unbatched + Cache

<table>
  <thead>
    <tr>
      <th style=" border-right:2px solid #999;">Config</th>
      <th style="background-color:#f2f2f2;">Game ± SE</th>
      <th>Letter</th>
      <th style="border-right:2px solid #999;">Word</th>
      <th style="background-color:#f2f2f2;">Calls</th>
      <th>Propose</th>
      <th style="border-right:2px solid #999;">Value</th>
      <th>Steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="background-color:#e6e6e6; border-right:2px solid #999;">Unbatched (paper)</td>
      <td style="background-color:#f2f2f2; text-align:right;">35.0 ± 1.8</td>
      <td style="text-align:right;">76.6</td>
      <td style="text-align:right; border-right:2px solid #999;">59.1</td>
      <td style="background-color:#f2f2f2; text-align:right;">448</td>
      <td style="text-align:right;">118</td>
      <td style="text-align:right; border-right:2px solid #999;">329</td>
      <td style="text-align:right;">23.7</td>
    </tr>
    <tr>
      <td style="background-color:#e6e6e6; border-right:2px solid #999;"><strong>Unbatched + cache</strong></td>
      <td style="background-color:#f2f2f2; text-align:right;"><strong>40.0 ± 4.1</strong></td>
      <td style="text-align:right;">77.1</td>
      <td style="text-align:right; border-right:2px solid #999;">61.5</td>
      <td style="background-color:#f2f2f2; text-align:right;"><strong>347</strong></td>
      <td style="text-align:right;">135</td>
      <td style="text-align:right; border-right:2px solid #999;"><strong>212</strong></td>
      <td style="text-align:right;">27.0</td>
    </tr>
  </tbody>
</table>




---

## ✍️ Task 3: Creative Writing / Text Generation

The creative writing task extends the Tree of Thoughts framework from constrained puzzle-solving to open-ended generation. Instead of searching over arithmetic states or crossword grids, this task searches over writing plans and candidate passages.

Each input contains four required ending sentences. The model must generate a coherent four-paragraph passage where paragraph `i` ends with sentence `i`. Since there is no deterministic verifier, outputs are evaluated using an LLM-as-judge coherence score from 1 to 10.

The generator model is **Gemini 2.5 Flash-Lite**, and the fixed judge model is **Gemini 3.1 Flash-Lite Preview**. Because the original paper used GPT-4 as judge, the absolute scores should not be directly compared to the paper’s values. The meaningful comparison is the relative ordering among methods under the same judge and input set.

### Methods

This task includes several generation and selection strategies:

- **IO**: Directly generates one passage from the input prompt.
- **CoT**: Generates one plan and passage in a single response.
- **ToT**: Generates 5 plans, votes for the best plan, generates 5 passages from that plan, and votes for the best passage.
- **Refinement**: Applies a post-generation revision step before scoring.
- **best_IO**: Generates 5 direct passages, scores each independently, and selects the highest-scored passage.
- **best_CoT**: Generates 5 complete plan-passage CoT outputs, scores each passage independently, and selects the highest-scored passage.
- **plan_only_cot**: Generates one explicit plan, then makes a second call to write one passage from that plan.
- **plan_vote**: Generates 5 plans, votes for the best plan, then writes one passage.
- **tot_score_select**: Uses independent coherence scoring instead of vote prompts to select candidates.

### Usage

Creative writing experiments are run through the same `run.py` entry point as the other tasks, with `--task text`.

```bash
# Main baselines
caffeinate -dimsu python code/run.py --task text --method io --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_io_gem25_n100.jsonl

caffeinate -dimsu python code/run.py --task text --method cot --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_cot_gem25_n100.jsonl

caffeinate -dimsu python code/run.py --task text --method tot --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_tot_gem25_n100.jsonl

# Selection and sampling extensions
caffeinate -dimsu python code/run.py --task text --method best_io --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_best_io_gem25_n100.jsonl

caffeinate -dimsu python code/run.py --task text --method best_cot --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_best_cot_gem25_n100.jsonl

caffeinate -dimsu python code/run.py --task text --method plan_vote --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_plan_vote_gem25_n100.jsonl

caffeinate -dimsu python code/run.py --task text --method tot_score_select --n_puzzles 100 \
  --vertex --rpm 15 --model gemini-2.5-flash-lite --thinking_level NONE \
  --out results/text/runs/text_tot_score_select_gem25_n100.jsonl
```

The default judge model is `gemini-3.1-flash-lite-preview` (override with `--judge_model`). When `--out` is omitted, runs land at `results/text/runs/text_<method>_<model>.jsonl`.

Raw creative writing outputs are stored in:

```text
results/text/runs/
```

Refined outputs are stored in:

```text
results/text/refined/
```

Creative writing figures and tables are stored in:

```text
analysis/text/figures/
analysis/text/tables/
```

### Results: Creative Writing

Tested on 100 creative writing inputs. Each input contains four required ending sentences, and the model must produce a coherent four-paragraph passage where paragraph `i` ends with sentence `i`.

Outputs are evaluated using an LLM-as-judge coherence score from 1 to 10. The generator model is **Gemini 2.5 Flash-Lite**, and the fixed judge model is **Gemini 3.1 Flash-Lite Preview**. Because the original paper used GPT-4 as the judge, absolute scores should not be directly compared to the paper’s reported values. The more meaningful comparison is the relative ordering among methods under the same judge and input set.

#### Main Baselines and Refinement

| Method | N | Mean coherence | Std. | SEM | Interpretation |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **IO** | 100 | 6.63 | 2.15 | 0.21 | Direct generation baseline. |
| **CoT** | 100 | 6.50 | 2.19 | 0.22 | Single-plan baseline; slightly below IO in this run. |
| **ToT** | 100 | 6.94 | 2.13 | 0.21 | Best among the main non-refined baselines. |
| **IO + Refine** | 100 | 6.64 | 2.14 | 0.21 | Very small gain over IO. |
| **CoT + Refine** | 100 | 6.82 | 2.16 | 0.22 | Refinement helps CoT noticeably. |
| **ToT + Refine** | 100 | **7.14** | 2.10 | 0.21 | Highest among main baselines + refinement. |

Standard ToT improves over both IO and CoT, and refinement further improves ToT. Among the main baseline/refinement methods, **ToT + Refine** achieves the highest mean coherence score.

#### Selection and Sampling Extensions

| Method | N | Mean coherence | Std. | Median | SEM | Interpretation |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **IO** | 100 | 6.63 | 2.15 | 6.67 | 0.21 | Direct one-shot generation. |
| **CoT** | 100 | 6.50 | 2.19 | 6.67 | 0.22 | One plan + one passage. |
| **plan_only_cot** | 100 | 6.57 | 2.35 | 7.00 | 0.23 | Separates planning and writing, but uses no selection. |
| **plan_vote** | 100 | 6.62 | 2.14 | 6.67 | 0.21 | Selects among plans, then writes one passage. |
| **ToT** | 100 | 6.94 | 2.13 | 7.00 | 0.21 | Standard vote-based ToT. |
| **best_CoT** | 100 | 7.61 | 2.00 | 8.00 | 0.20 | Samples multiple full CoT chains and score-selects. |
| **best_IO** | 100 | 7.93 | 1.86 | 8.67 | 0.19 | Samples multiple direct passages and score-selects. |
| **tot_score_select** | 100 | **8.14** | 1.85 | 8.67 | 0.19 | Highest mean score; uses independent score-based selection. |

The extension results show that the largest gain comes from **candidate scoring and selection**. Standard vote-based ToT improves over IO and CoT, but **best_IO**, **best_CoT**, and especially **tot_score_select** outperform standard ToT. This suggests that, for open-ended creative writing, sampling multiple candidates and selecting well is a very strong baseline.

#### Key Takeaways

- **ToT helps over direct baselines**: Standard ToT reaches 6.94 mean coherence, outperforming IO at 6.63 and CoT at 6.50.
- **Refinement helps most when the base method already has structure**: CoT improves from 6.50 to 6.82 after refinement, and ToT improves from 6.94 to 7.14.
- **Selection matters more than the tree alone**: best_IO, best_CoT, and tot_score_select all outperform standard ToT.
- **Best overall method**: `tot_score_select` achieves the highest mean coherence score, 8.14.
- **Main interpretation**: For Creative Writing, ToT should be understood as evaluator-guided search. The tree structure helps, but the quality of the scoring or selection mechanism is central to performance.

### Creative Writing vs. Puzzle Tasks

Unlike Game of 24 and Mini Crosswords, Creative Writing has no exact symbolic verifier or grid-based correctness metric. Therefore, the evaluation is less about binary correctness and more about relative quality under a fixed judge.

| Task | Search unit | Evaluation | Main lesson |
| :--- | :--- | :--- | :--- |
| **Game of 24** | Arithmetic states | Exact symbolic verification | ToT can strongly improve reasoning, but evaluator design and batching matter. |
| **Mini Crosswords** | Partial grid states | Letter, word, and game accuracy | ToT helps with constraint-heavy search when pruning and backtracking are reliable. |
| **Creative Writing** | Plans and passages | LLM-as-judge coherence score | ToT helps, but candidate scoring/selection drives the largest gains. |

Overall, the Creative Writing extension supports the broader project conclusion: ToT is not just “generating more thoughts.” Its success depends on whether the system can reliably evaluate, select, and refine intermediate candidates.

---

## 🏗️ Project Structure

```text
├── README.md
├── LICENSE
├── .gitignore
├── Tree of Thoughts.pdf        # Original paper (Yao et al., NeurIPS 2023)
├── code/                       # All re-implementation source
│   ├── run.py                  # Unified CLI entry point (24 / crosswords / text)
│   ├── analyze.py              # Performance analysis tool
│   ├── rescore_24.py           # Trace-based post-processing for Game of 24
│   ├── reverify_24.py          # Re-runs verification on stored equations
│   ├── requirements.txt
│   ├── conftest.py             # Makes `pytest code/tests/` work from repo root
│   ├── algorithms/
│   │   ├── methods_24.py       # BFS logic: IO, CoT, and ToT for Game of 24
│   │   ├── methods_cw.py       # DFS backtracking logic for Crosswords
│   │   └── methods_text.py     # Creative writing generation and selection methods
│   ├── core/
│   │   └── model.py            # Async Gemini wrapper, rate limiter, and budgeting
│   ├── tasks/
│   │   ├── game24.py           # State logic and exact verification for Game of 24
│   │   ├── crosswords.py       # Grid logic, constraints, and hallucination filters
│   │   └── text.py             # Creative writing input loader
│   ├── prompts/
│   │   ├── prompts_24.py       # Prompts for Game of 24
│   │   ├── prompts_cw.py       # Prompts for Crosswords, including batched value prompts
│   │   └── prompts_text.py     # Prompts for creative writing and judging
│   ├── tests/
│   │   ├── test_game24.py      # Unit tests for 24-point verifier
│   │   └── test_crossword.py   # Unit tests for Crossword constraints
│   ├── scripts/
│   │   └── crosswords.sh
│   └── analysis/               # Notebooks and analysis scripts for the writeup
│       ├── crosswords/         # Crossword reproduction notebooks
│       ├── game24/             # Game of 24 analysis
│       └── text/               # Creative writing figures / tables
├── data/
│   ├── 24.csv                  # Game of 24 data
│   ├── mini0505.json           # Mini Crosswords data
│   ├── mini0505_0_100_5.json
│   └── text_inputs.json        # Creative writing inputs
├── results/
│   ├── game24/
│   │   ├── runs/               # Game of 24 experiment outputs
│   │   └── reverified/         # Game of 24 reverified / rescored outputs
│   ├── crosswords/
│   │   ├── runs/               # Crossword experiment outputs
│   │   └── snapshots/          # Historical Crossword snapshots
│   ├── text/
│   │   ├── runs/               # Raw creative writing outputs
│   │   └── refined/            # Refined creative writing outputs
│   └── logs/                   # Long-run log files
├── poster/
│   └── Poster.pdf              # In-class presentation poster
└── report/                     # Final report PDF (to be added)
```

All commands assume you run from the repo root, e.g. `python code/run.py ...` or `pytest code/tests/`.

---

## Implementation Notes

- **Verifier correctness matters.** Run `pytest code/tests/` before trusting any result. The classic `8 / (3 - 8/3) = 24` case should pass because it requires exact rational arithmetic.
- **Value scoring follows the paper.** The value prompt sums the three vote scores rather than multiplying them.
- **Batched value evaluation is a deliberate deviation from the paper.** The original ToT implementation makes one API call per `(candidate, vote)`. This implementation can pack all candidates from one BFS depth into a single prompt and run `n_votes=3` batched evaluations in parallel. This greatly reduces cost but may introduce calibration drift.
- **Two generator prompts are used for Game of 24.** Depths 0–1 use a proposal prompt that applies one binary operation at a time. Depth 2 switches to a CoT-style finishing prompt to produce the final equation.
- **Thinking level.** Gemini 3.x defaults to `--thinking_level MINIMAL` so that the external Tree-of-Thought search structure, rather than the model's internal reasoning, drives the comparison.
- **Rate limits.** `AsyncRateLimiter` / `GeminiWrapper` enforces the 15 RPM free-tier limit and works correctly under `asyncio.gather`.
- **Budgeting.** Batched evaluation can reduce Game of 24 costs substantially, from the original paper-style evaluation to a much smaller number of API calls per puzzle.
- **Creative writing evaluation is judge-based.** Unlike Game of 24 and Mini Crosswords, creative writing does not have exact correctness labels, so IO, CoT, best-of-k, ToT, score selection, and refinement results should be compared using the same judge model and evaluation rubric.

---

## Model-Specific Notes

### Gemini 3.1 Flash-Lite

Gemini 3.1 Flash-Lite is a thinking model. The default setting is `--thinking_level MINIMAL`, which keeps behavior closer to the original ToT setup by letting the explicit search structure do most of the reasoning. To experiment with the model's natural mode, pass `--thinking_level NONE`, but expect much higher token usage.

### Gemini 2.5 Flash-Lite

Gemini 2.5 Flash-Lite has no internal chain-of-thought. Use:

```bash
python code/run.py --model gemini-2.5-flash-lite --thinking_level NONE
```

For meaningful ToT results on this model, prefer `--no_batch` because batched value evaluation significantly reduces accuracy.

### Gemma 3 27B

Gemma 3 27B is run through AI Studio with `--thinking_level NONE`. Only batched results were completed. Like Gemini 2.5, it shows substantial batching sensitivity.

---

## 7. Conclusion

This project reproduces the **Tree of Thoughts** framework on three tasks — Game of 24, Mini Crosswords, and Creative Writing — using modern fast/lite Gemini models in place of GPT-4. Across all three tasks, ToT improves over IO and CoT baselines when paired with a reliable evaluator: on Game of 24, paper-strict unbatched ToT(b=5) reaches **96%** on Gemini 3.1 and **87%** on Gemini 2.5, exceeding the **74%** reported in the original paper; on Mini Crosswords, ToT with DFS, pruning, and backtracking lifts game accuracy from **0–1%** (IO/CoT) to **37.2%**, beating the paper's **20%**; on Creative Writing, `tot_score_select` achieves the highest mean coherence (**8.14**) among all tested methods.

A consistent theme across tasks is that **evaluator design matters more than the tree alone.** Batched value prompts dramatically reduce API cost, but on non-thinking models they can collapse ToT accuracy below CoT (see 7.1). For Crosswords, the largest gains came from composable evaluator features (focused / verified / cache) rather than from changing the search itself. For Creative Writing, replacing vote-based selection with independent scoring (`tot_score_select`) outperformed standard ToT — again pointing to the evaluator as the load-bearing component.

The total Vertex cost for all Gemini Game-of-24 experiments was about **\$8.03**, well within the free-tier and Tier-1 budgets used during development. The full reproduction was completed on free-tier and entry-paid Gemini APIs, demonstrating that ToT-style reasoning is reproducible on commodity model endpoints without GPT-4 access.

### 7.1 Batching Caveat

Batching all BFS candidates into a single value prompt is useful for cost and rate-limit reasons, but it can silently damage ToT accuracy on non-thinking models. On Gemini 2.5, batched ToT(b=5) scored only **15%**, worse than CoT at **29%**. Switching to the paper-style per-state evaluator with `--no_batch` (Game of 24) or `--batch no` (Crosswords) recovered performance to **87%**.

Thinking models are more resistant. Gemini 3.1 loses fewer points from batching, and trace-based rescoring can recover some final-answer mistakes. However, rescoring cannot repair a corrupted search trajectory.

**If you are running on a non-thinking model, use `--batch no` for Crosswords (and `--no_batch` for Game of 24).** The extra API calls are usually worth it. For Crosswords on a thinking model, `--batch full --focused --verified` keeps the cheap call shape while adding a confirmation pass on every kill.

---

## 8. References

[1] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS) 36. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)

[2] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS) 35.

[3] Cheng, J., Liu, X., Zheng, K., Ke, P., Wang, H., Dong, Y., Tang, J., & Huang, M. (2023). *Black-Box Prompt Optimization: Aligning Large Language Models without Model Training.* Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

[4] Lin, B. Y., Deng, Y., Chandu, K., Brahman, F., Ravichander, A., Pyatkin, V., Dziri, N., Bras, R. L., & Choi, Y. (2024). *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback.* International Conference on Learning Representations (ICLR).

[5] Princeton NLP. *Tree of Thoughts — Official Implementation.* GitHub repository. [https://github.com/princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm)

[6] 4nums.com. *24 Puzzle Database.* [https://www.4nums.com/](https://www.4nums.com/)

[7] Google Cloud. *Vertex AI — Generative AI on Google Cloud.* [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)

[8] Meurer, A., Smith, C. P., Paprocki, M., Čertík, O., Kirpichev, S. B., Rocklin, M., et al. (2017). *SymPy: Symbolic Computing in Python.* PeerJ Computer Science 3:e103. [https://www.sympy.org/](https://www.sympy.org/)

---

## 9. Acknowledgements

This project was completed as part of CS 4782: Introduction to Deep Learning at Cornell University, Spring 2026.