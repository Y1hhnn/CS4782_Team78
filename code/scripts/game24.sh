#!/usr/bin/env bash
set -euo pipefail

# Game of 24 driver.
#
# Usage examples (run from anywhere — the script cd's to the repo root):
#   bash code/scripts/game24.sh
#   N_PUZZLES=20 OVERWRITE=1 bash code/scripts/game24.sh
#   MODEL=gemini-2.5-flash-lite THINKING_LEVEL=NONE VERTEX=1 RPM=60 \
#       bash code/scripts/game24.sh
#
# By default it runs IO + CoT baselines, then ToT(b=1) and ToT(b=5) in both
# batched and unbatched value-evaluator modes. Edit / comment lines below to
# pick which experiments to run.

# Anchor to the repo root so relative paths (data/, results/) resolve consistently
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="24"
N_PUZZLES="${N_PUZZLES:-100}"
MODEL="${MODEL:-gemini-3.1-flash-lite-preview}"
THINKING_LEVEL="${THINKING_LEVEL:-MINIMAL}"
RPM="${RPM:-15}"
CONCURRENCY="${CONCURRENCY:-5}"

COMMON_ARGS=(
  --task "$TASK"
  --n_puzzles "$N_PUZZLES"
  --model "$MODEL"
  --thinking_level "$THINKING_LEVEL"
  --rpm "$RPM"
  --concurrency "$CONCURRENCY"
)

if [[ "${VERTEX:-0}" == "1" ]]; then
  COMMON_ARGS+=(--vertex)
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

mkdir -p results/logs

run_exp () {
  local name="$1"
  shift

  echo
  echo "============================================================"
  echo "Running: $name"
  echo "Command: python code/run.py ${COMMON_ARGS[*]} $*"
  echo "============================================================"

  python code/run.py "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "results/logs/${name}.log"
}

# ---------------------------------------------------------
# Baselines (1 LLM call per puzzle)
# ---------------------------------------------------------
run_exp "g24_io"  --method io
run_exp "g24_cot" --method cot

# ---------------------------------------------------------
# ToT batched (default; one call scores all candidates at a depth)
# ---------------------------------------------------------
run_exp "g24_tot_b1_batched" --method tot --b 1
run_exp "g24_tot_b5_batched" --method tot --b 5

# ---------------------------------------------------------
# ToT unbatched (paper-strict; one call per candidate state)
# More accurate, ~5x more API calls.
# ---------------------------------------------------------
# run_exp "g24_tot_b1_unbatched" --method tot --b 1 --no_batch
# run_exp "g24_tot_b5_unbatched" --method tot --b 5 --no_batch

echo
echo "All Game of 24 experiments finished."
echo "Results saved under results/game24/runs/"
echo "Logs saved under results/logs/"
echo
echo "Optional post-processing:"
echo "  python code/rescore_24.py  results/game24/runs/*.jsonl   # trace-based rescore"
echo "  python code/reverify_24.py results/game24/runs/<f>.jsonl results/game24/reverified/<f>.jsonl"
echo "  python code/analyze.py     results/game24/runs/*.jsonl"
