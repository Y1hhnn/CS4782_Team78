#!/usr/bin/env bash
set -euo pipefail

# Usage examples (run from anywhere — the script cd's to the repo root):
#   bash code/scripts/crosswords.sh
#   PAPER_SPLIT=0 N_PUZZLES=100 MAX_STEPS=100 OVERWRITE=1 bash code/scripts/crosswords.sh
#   MODEL=gemini-3.1-flash-lite-preview RPM=15 CONCURRENCY=5 bash code/scripts/crosswords.sh

# Anchor to the repo root so relative paths (data/, results/) resolve consistently
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="crosswords"
N_PUZZLES="${N_PUZZLES:-100}"
MAX_STEPS="${MAX_STEPS:-100}"
MODEL="${MODEL:-gemini-3.1-flash-lite-preview}"
THINKING_LEVEL="${THINKING_LEVEL:-MINIMAL}"
RPM="${RPM:-15}"
CONCURRENCY="${CONCURRENCY:-5}"

COMMON_ARGS=(
  --task "$TASK"
  --n_puzzles "$N_PUZZLES"
  --max_steps "$MAX_STEPS"
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

# Added support for the newly implemented paper split logic
if [[ "${PAPER_SPLIT:-1}" == "1" ]]; then
  COMMON_ARGS+=(--paper_split)
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
# Baselines
# ---------------------------------------------------------
run_exp "cw_io"  --method io
run_exp "cw_cot" --method cot

# ---------------------------------------------------------
# ToT Level 1 -- full batch (1 LLM call per DFS step)
# Cheapest call shape. Default = full_basic.
# Tip: pass --top_k 3 to focused runs to filter to top-3 clues, or --cache
# to enable the per-puzzle short-circuit cache (verified writes, focused reads).
# ---------------------------------------------------------
run_exp "cw_tot_full_basic"             --method tot --batch full
run_exp "cw_tot_full_focused"           --method tot --batch full --focused
run_exp "cw_tot_full_verified"          --method tot --batch full --verified
run_exp "cw_tot_full_focused_verified"  --method tot --batch full --focused --verified

run_exp "cw_tot_full_focused_topk3"     --method tot --batch full --focused --top_k 3
run_exp "cw_tot_full_focused_topk3_verified"     --method tot --batch full --focused --top_k 3 --verified
run_exp "cw_tot_full_focused_topk5"                 --method tot --batch full --focused --top_k 5
run_exp "cw_tot_full_focused_topk7"                 --method tot --batch full --focused --top_k 7

# # ---------------------------------------------------------
# ToT Level 2 -- half batch (1 LLM call per sibling)
# Off by default; uncomment to enable.
# ---------------------------------------------------------
run_exp "cw_tot_half_basic"             --method tot --batch half

run_exp "cw_tot_half_focused"           --method tot --batch half --focused
run_exp "cw_tot_half_verified"          --method tot --batch half --verified
run_exp "cw_tot_half_focused_verified"  --method tot --batch half --focused --verified

run_exp "cw_tot_half_focused_topk3"     --method tot --batch half --focused --top_k 3
run_exp "cw_tot_half_focused_topk5"                 --method tot --batch half --focused --top_k 5
run_exp "cw_tot_half_focused_topk7"                 --method tot --batch half --focused --top_k 7
run_exp "cw_tot_half_focused_topk7_verified"                 --method tot --batch half --focused --top_k 7 --verified

# ---------------------------------------------------------
# ToT Level 3 -- unbatched (paper-strict; 1 call per testable clue)
# Includes search ablations for analysis.
# --batch no is incompatible with --focused / --verified / --cache.
# ---------------------------------------------------------
run_exp "cw_tot_unbatched_normal"       --method tot --batch no
run_exp "cw_tot_unbatched_noprune"      --method tot --batch no --no_prune
run_exp "cw_tot_unbatched_nobacktrack"  --method tot --batch no --no_backtrack
run_exp "cw_tot_unbatched_cache"  --method tot --batch no --cache

echo
echo "All Crosswords experiments finished."
echo "Results saved under results/crosswords/runs/"
echo "Logs saved under results/logs/"
