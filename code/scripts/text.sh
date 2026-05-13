#!/usr/bin/env bash
set -euo pipefail

# Creative Writing driver.
#
# Usage examples (run from anywhere — the script cd's to the repo root):
#   bash code/scripts/text.sh
#   N_PUZZLES=20 OVERWRITE=1 bash code/scripts/text.sh
#   RUN_TAG=gem25_n100 VERTEX=1 RPM=60 bash code/scripts/text.sh
#   MODEL=gemini-2.5-flash JUDGE_MODEL=gemini-3.1-flash-lite-preview \
#       bash code/scripts/text.sh
#
# The generator model defaults to gemini-2.5-flash-lite (no thinking) and the
# judge model to gemini-3.1-flash-lite-preview (MINIMAL thinking), matching the
# numbers reported in the README. Edit / comment lines below to pick which
# methods to run.

# Anchor to the repo root so relative paths (data/, results/) resolve consistently
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TASK="text"
N_PUZZLES="${N_PUZZLES:-100}"
MODEL="${MODEL:-gemini-2.5-flash-lite}"
THINKING_LEVEL="${THINKING_LEVEL:-NONE}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
JUDGE_THINKING_LEVEL="${JUDGE_THINKING_LEVEL:-MINIMAL}"
RPM="${RPM:-15}"
CONCURRENCY="${CONCURRENCY:-5}"

# Tag appended to output filenames so distinct configurations don't collide.
# Set RUN_TAG=anything to override (e.g., RUN_TAG=gem25_n50).
RUN_TAG="${RUN_TAG:-gem25_n${N_PUZZLES}}"

COMMON_ARGS=(
  --task "$TASK"
  --n_puzzles "$N_PUZZLES"
  --model "$MODEL"
  --thinking_level "$THINKING_LEVEL"
  --judge_model "$JUDGE_MODEL"
  --judge_thinking_level "$JUDGE_THINKING_LEVEL"
  --rpm "$RPM"
  --concurrency "$CONCURRENCY"
)

if [[ "${VERTEX:-0}" == "1" ]]; then
  COMMON_ARGS+=(--vertex)
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

mkdir -p results/logs results/text/runs

run_exp () {
  local method="$1"
  local name="text_${method}_${RUN_TAG}"
  local out_path="results/text/runs/${name}.jsonl"

  echo
  echo "============================================================"
  echo "Running: $name"
  echo "Command: python code/run.py ${COMMON_ARGS[*]} --method ${method} --out ${out_path}"
  echo "============================================================"

  python code/run.py "${COMMON_ARGS[@]}" --method "$method" --out "$out_path" \
      2>&1 | tee "results/logs/${name}.log"
}

# ---------------------------------------------------------
# Main baselines
# ---------------------------------------------------------
run_exp io
run_exp cot
run_exp tot

# ---------------------------------------------------------
# Selection and sampling extensions
# ---------------------------------------------------------
run_exp best_io
run_exp best_cot
run_exp plan_only_cot
run_exp plan_vote
run_exp tot_score_select

echo
echo "All Creative Writing experiments finished."
echo "Results saved under results/text/runs/"
echo "Logs saved under results/logs/"
