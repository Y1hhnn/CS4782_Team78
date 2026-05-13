"""Convert Game-of-24 JSONL run files into the CSV layout used by the R
analysis scripts (`code/analysis/game24/poster_plots.R`).

Emits two aggregate CSVs to `results/game24/csv/`:

  all_runs.csv  -- one row per (source_file, puzzle); includes per-depth
                   score aggregates and the top-5 child scores extracted
                   from each puzzle's trace.
  summary.csv   -- one row per source_file; n_puzzles, n_solved,
                   success_rate, mean/total elapsed_s.

Filename convention parsed for run metadata:

  <method>_b<beam>[_bs<size> | _<model_tag>_unbatched | _<model_tag>][.rescored].jsonl

Method is the first underscore-separated token (`io`, `cot`, `tot`).
Default model is gemini-3.1-flash-lite; `_gemini25*` => gemini-2.5-flash-lite,
`_gemini31*` => gemini-3.1-flash-lite, `_gemma3-27B` => gemma-3-27b.

Usage:
    python code/to_csv.py results/game24/runs/*.jsonl
    python code/to_csv.py --out-dir results/game24/csv results/game24/runs/*.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


# Header used for both all_runs.csv and the per-row dicts.
BASE_FIELDS = [
    "source_file", "method", "beam", "batch_mode", "batch_size", "model",
    "idx", "inputs", "equation", "verified", "elapsed_s",
]
DEPTH_FIELDS = []
for d in range(3):
    DEPTH_FIELDS += [
        f"depth{d}_max_score", f"depth{d}_mean_score",
        f"depth{d}_min_score", f"depth{d}_n_candidates",
    ]
    DEPTH_FIELDS += [f"depth{d}_score_{k}" for k in range(1, 6)]
ALL_RUNS_HEADER = BASE_FIELDS + DEPTH_FIELDS

SUMMARY_HEADER = [
    "source_file", "method", "beam", "batch_mode", "batch_size", "model",
    "n_puzzles", "n_solved", "success_rate",
    "mean_elapsed_s", "total_elapsed_s",
]


# ----------------------------------------------------------------------
# Filename parsing
# ----------------------------------------------------------------------

_BEAM_RE = re.compile(r"_b(\d+)")
_BS_RE = re.compile(r"_bs(\d+)")

_MODEL_TAGS = [
    ("gemini25", "gemini-2.5-flash-lite"),
    ("gemini2flashlite", "gemini-2.5-flash-lite"),
    ("gemini31", "gemini-3.1-flash-lite"),
    ("gemma3-27B", "gemma-3-27b"),
    ("gemma", "gemma-3-27b"),
    ("20flash", "gemini-2.0-flash"),
]


def parse_filename(name: str) -> dict:
    """Extract method/beam/batch_mode/batch_size/model from a JSONL filename."""
    stem = name.replace(".rescored.jsonl", "").replace(".jsonl", "")
    tokens = stem.split("_")
    method = tokens[0] if tokens else ""

    m = _BEAM_RE.search(stem)
    beam = int(m.group(1)) if m else (1 if method in {"io", "cot"} else 5)

    bs_match = _BS_RE.search(stem)
    if bs_match:
        batch_size = int(bs_match.group(1))
        batch_mode = f"bs{batch_size}"
    elif "unbatched" in stem:
        batch_size = 1
        batch_mode = "unbatched"
    else:
        batch_size = ""
        batch_mode = "batched"

    model = "gemini-3.1-flash-lite"
    for tag, full in _MODEL_TAGS:
        if tag in stem:
            model = full
            break

    return {
        "method": method,
        "beam": beam,
        "batch_mode": batch_mode,
        "batch_size": batch_size,
        "model": model,
    }


# ----------------------------------------------------------------------
# Trace aggregation
# ----------------------------------------------------------------------

def depth_features(trace, max_depth: int = 3, top_k: int = 5) -> dict:
    """Extract per-depth score aggregates from a ToT trace.

    `trace[d]` is a list of `[score, last_step]` pairs for the top-b states
    at depth d. For each depth, we record max/mean/min/count and the top-k
    individual scores (already sorted descending in the JSONL).
    """
    out = {}
    if not trace:
        return out
    for d in range(max_depth):
        if d >= len(trace) or not trace[d]:
            continue
        scores = [pair[0] for pair in trace[d] if pair]
        if not scores:
            continue
        out[f"depth{d}_max_score"] = max(scores)
        out[f"depth{d}_mean_score"] = round(statistics.mean(scores), 4)
        out[f"depth{d}_min_score"] = min(scores)
        out[f"depth{d}_n_candidates"] = len(scores)
        for k in range(1, top_k + 1):
            if k <= len(scores):
                out[f"depth{d}_score_{k}"] = scores[k - 1]
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _format_inputs(raw) -> str:
    """JSONL stores inputs as a list/tuple; CSVs render them as space-separated."""
    if isinstance(raw, (list, tuple)):
        return " ".join(str(int(x)) if float(x).is_integer() else str(x) for x in raw)
    return str(raw)


def process_file(path: Path) -> tuple[list[dict], dict]:
    """Return (per-puzzle rows, summary row) for one JSONL."""
    meta = parse_filename(path.name)
    rows = []
    elapsed = []
    n_solved = 0

    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            row = {f: "" for f in ALL_RUNS_HEADER}
            row.update({"source_file": path.name, **meta})
            row["idx"] = d.get("idx", "")
            row["inputs"] = _format_inputs(d.get("inputs", ""))
            row["equation"] = d.get("equation", "")
            row["verified"] = d.get("verified", "")
            row["elapsed_s"] = d.get("elapsed_s", "")
            row.update(depth_features(d.get("trace")))
            rows.append(row)

            if d.get("verified"):
                n_solved += 1
            try:
                elapsed.append(float(d.get("elapsed_s", 0) or 0))
            except (TypeError, ValueError):
                pass

    n = len(rows)
    summary = {
        "source_file": path.name,
        **meta,
        "n_puzzles": n,
        "n_solved": n_solved,
        "success_rate": round(100 * n_solved / n, 2) if n else 0.0,
        "mean_elapsed_s": round(sum(elapsed) / len(elapsed), 2) if elapsed else 0.0,
        "total_elapsed_s": sum(elapsed),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+", type=Path, help="Game-of-24 run JSONLs")
    ap.add_argument(
        "--out-dir", type=Path, default=Path("results/game24/csv"),
        help="Where to write all_runs.csv and summary.csv",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for path in sorted(args.jsonl):
        rows, summary = process_file(path)
        all_rows.extend(rows)
        summaries.append(summary)

    all_path = args.out_dir / "all_runs.csv"
    with all_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_RUNS_HEADER)
        w.writeheader()
        w.writerows(all_rows)

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        w.writerows(summaries)

    print(f"Wrote {len(all_rows)} rows to {all_path}")
    print(f"Wrote {len(summaries)} files to {summary_path}")


if __name__ == "__main__":
    main()
