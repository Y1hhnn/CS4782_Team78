# Experiment Scripts

Convenience drivers that wrap `python code/run.py` with sensible defaults and stream a per-experiment log to `results/logs/`. One script per task.

| Script | Task | Default experiments |
|--------|------|---------------------|
| [`game24.sh`](game24.sh) | Game of 24 (BFS) | `io`, `cot`, `tot` at `b=1` / `b=5` (batched). Unbatched ToT lines are commented in by default. |
| [`crosswords.sh`](crosswords.sh) | Mini Crosswords (DFS) | `io`, `cot`, plus various `tot` evaluator combinations (batch / focused / verified / cache). |
| [`text.sh`](text.sh) | Creative Writing | `io`, `cot`, `tot`, `best_io`, `best_cot`, `plan_only_cot`, `plan_vote`, `tot_score_select`. |

## Run from the repo root

All scripts auto-`cd` to the repo root so that `data/` and `results/` resolve correctly. You can invoke them from anywhere:

```bash
bash code/scripts/game24.sh
bash code/scripts/crosswords.sh
bash code/scripts/text.sh
```

## Override via environment variables

Common knobs (apply to all three scripts unless noted):

| Variable | Default | Effect |
|----------|---------|--------|
| `N_PUZZLES` | `100` | Number of puzzles / inputs. |
| `MODEL` | `gemini-3.1-flash-lite-preview` (game24, crosswords); `gemini-2.5-flash-lite` (text) | Generator model. |
| `THINKING_LEVEL` | `MINIMAL` (game24, crosswords); `NONE` (text) | `MINIMAL` / `LOW` / `MEDIUM` / `HIGH` / `NONE`. |
| `RPM` | `15` | Per-minute rate limit. `0` disables. |
| `CONCURRENCY` | `5` | Max concurrent puzzle tasks. |
| `VERTEX` | `0` | Set `=1` to route through Vertex AI. |
| `OVERWRITE` | `0` | Set `=1` to truncate existing JSONL outputs before running. |

Crosswords-only:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_STEPS` | `100` | DFS expansion budget per puzzle. |
| `PAPER_SPLIT` | `1` | Set `=0` to disable the 20-puzzle paper subset. |

Text-only:

| Variable | Default | Effect |
|----------|---------|--------|
| `JUDGE_MODEL` | `gemini-3.1-flash-lite-preview` | LLM-as-judge model. |
| `JUDGE_THINKING_LEVEL` | `MINIMAL` | Thinking level for the judge. |
| `RUN_TAG` | `gem25_n${N_PUZZLES}` | Suffix appended to output filenames so distinct configurations don't collide. |

## Examples

```bash
# Smoke test: 5 puzzles each, fresh output
N_PUZZLES=5 OVERWRITE=1 bash code/scripts/game24.sh

# Full run on Vertex with raised throughput
VERTEX=1 RPM=60 N_PUZZLES=100 bash code/scripts/text.sh

# Crosswords without the paper subset (full 100 puzzles)
PAPER_SPLIT=0 N_PUZZLES=100 OVERWRITE=1 bash code/scripts/crosswords.sh

# Override the model for everything
MODEL=gemini-2.5-flash-lite THINKING_LEVEL=NONE bash code/scripts/game24.sh
```

## Output locations

- JSONL results: `results/<task>/runs/<name>.jsonl`
  (e.g. `results/game24/runs/gem31_tot_b5_batched.jsonl`, `results/text/runs/text_tot_gem25_n100.jsonl`)
- Per-experiment logs: `results/logs/<name>.log`

## Customizing which experiments run

Each script lists every experiment as a `run_exp …` line. Comment / uncomment lines to pick the subset you want. Search for `run_exp` in the file to find the experiment list.
