"""Analyze the value-evaluator verdict distributions in ToT traces.

Compares batched vs. un-batched runs to answer: does batching make the
evaluator uniformly optimistic ("sure" to everything), or does it
scramble rankings while preserving the overall distribution?

Usage:
    python analyze_verdicts.py results/runs/reverified/tot_b5.jsonl \
                               results/runs/reverified/tot_b5_unbatched.jsonl

Pass any number of JSONL files. For each file, the script extracts per-depth
score distributions and prints summary statistics. When two files are given,
it also prints a head-to-head comparison on overlapping puzzles.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# These are the score values assigned by the value prompt:
#   sure=20, likely=1, impossible=0.001
# With n_votes=3 and summed scores, the possible per-state totals are
# combinations of {20, 1, 0.001} * 3 votes. Common values:
#   60.0   = 3× sure
#   41.0   = 2× sure + 1× likely
#   40.0   = 2× sure + 1× impossible  (or 2×sure if rounding)
#   22.0   = 1× sure + 2× likely
#   21.0   = 1× sure + 1× likely + 1× impossible
#   20.0   = 1× sure + 2× impossible
#    3.0   = 3× likely
#    2.0   = 2× likely + 1× impossible
#    1.0   = 1× likely + 2× impossible
#    0.003 = 3× impossible

SCORE_TO_LABEL = {
    60.0: "3×sure",
    41.0: "2×sure+1×likely",
    40.0: "2×sure+1×imp",
    22.0: "1×sure+2×likely",
    21.0: "1×sure+1×likely+1×imp",
    20.0: "1×sure+2×imp",
    3.0:  "3×likely",
    2.0:  "2×likely+1×imp",
    1.0:  "1×likely+2×imp",
    0.003: "3×imp",
}


def classify_score(score: float) -> str:
    """Map a summed score back to a human-readable vote breakdown."""
    # Find nearest known score (handle float imprecision)
    best_label = "unknown"
    best_dist = 0.5
    for known, label in SCORE_TO_LABEL.items():
        d = abs(score - known)
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label


def extract_depth_scores(row: dict) -> list[list[float]]:
    """Extract the score values from a trace at each BFS depth.

    trace[d] is a list of (score, step_text) pairs for the top-b candidates
    at depth d+1.
    """
    trace = row.get("trace") or []
    result = []
    for depth_entries in trace:
        scores = []
        for entry in depth_entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    scores.append(float(entry[0]))
                except (ValueError, TypeError):
                    pass
        result.append(scores)
    return result


def analyze_file(path: Path) -> dict:
    """Analyze one JSONL file. Returns structured stats."""
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        return {"path": str(path), "n": 0}

    n = len(rows)
    n_solved = sum(r.get("verified", False) for r in rows)

    # Per-depth score distributions
    depth_scores = {0: [], 1: [], 2: []}
    for row in rows:
        per_depth = extract_depth_scores(row)
        for d, scores in enumerate(per_depth):
            if d in depth_scores:
                depth_scores[d].extend(scores)

    # Score classification distributions per depth
    depth_verdicts = {}
    for d, scores in depth_scores.items():
        counter = Counter(classify_score(s) for s in scores)
        total = sum(counter.values())
        depth_verdicts[d] = {
            "total_candidates": total,
            "distribution": {k: (v, f"{100*v/total:.1f}%") for k, v in
                           counter.most_common()},
        }

    # Overall discrimination: how spread are the scores at each depth?
    depth_stats = {}
    for d, scores in depth_scores.items():
        if scores:
            import statistics
            depth_stats[d] = {
                "n": len(scores),
                "mean": round(statistics.mean(scores), 2),
                "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                "pct_max_score": round(100 * sum(1 for s in scores if s > 59) / len(scores), 1),
                "pct_min_score": round(100 * sum(1 for s in scores if s < 0.01) / len(scores), 1),
            }

    return {
        "path": str(path),
        "n": n,
        "solved": n_solved,
        "pct": round(100 * n_solved / n, 1) if n else 0,
        "depth_verdicts": depth_verdicts,
        "depth_stats": depth_stats,
    }


def print_analysis(analysis: dict) -> None:
    """Pretty-print one file's analysis."""
    print(f"\n{'='*70}")
    print(f"  {analysis['path']}")
    print(f"  {analysis['solved']}/{analysis['n']} = {analysis['pct']}% solved")
    print(f"{'='*70}")

    for d in range(3):
        stats = analysis.get("depth_stats", {}).get(d)
        verdicts = analysis.get("depth_verdicts", {}).get(d)
        if not stats:
            continue

        print(f"\n  Depth {d}: {stats['n']} candidates evaluated")
        print(f"    mean score: {stats['mean']:.1f}  std: {stats['std']:.1f}")
        print(f"    % at max (3×sure, ≥60):  {stats['pct_max_score']}%")
        print(f"    % at min (3×imp, ≤0.01): {stats['pct_min_score']}%")

        if verdicts:
            print(f"    verdict breakdown:")
            for label, (count, pct) in verdicts["distribution"].items():
                bar = "█" * max(1, int(float(pct.rstrip('%')) / 3))
                print(f"      {label:<25} {count:>5} ({pct:>6})  {bar}")


def print_comparison(a1: dict, a2: dict) -> None:
    """Print head-to-head comparison of two analyses."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"  A: {a1['path']}")
    print(f"  B: {a2['path']}")
    print(f"{'='*70}")

    print(f"\n  {'':25} {'A':>12} {'B':>12} {'Δ':>12}")
    print(f"  {'Solved':25} {a1['pct']:>11}% {a2['pct']:>11}% {a2['pct']-a1['pct']:>+11.1f}%")

    for d in range(3):
        sa = a1.get("depth_stats", {}).get(d, {})
        sb = a2.get("depth_stats", {}).get(d, {})
        if not sa or not sb:
            continue

        print(f"\n  Depth {d}:")
        print(f"  {'  mean score':25} {sa['mean']:>12.1f} {sb['mean']:>12.1f} {sb['mean']-sa['mean']:>+12.1f}")
        print(f"  {'  std score':25} {sa['std']:>12.1f} {sb['std']:>12.1f} {sb['std']-sa['std']:>+12.1f}")
        print(f"  {'  % at 3×sure (≥60)':25} {sa['pct_max_score']:>11.1f}% {sb['pct_max_score']:>11.1f}% {sb['pct_max_score']-sa['pct_max_score']:>+11.1f}%")
        print(f"  {'  % at 3×imp (≤0.01)':25} {sa['pct_min_score']:>11.1f}% {sb['pct_min_score']:>11.1f}% {sb['pct_min_score']-sa['pct_min_score']:>+11.1f}%")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    analyses = []

    for path in paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        analysis = analyze_file(path)
        analyses.append(analysis)
        print_analysis(analysis)

    # If exactly two files given, print head-to-head comparison
    if len(analyses) == 2:
        print_comparison(analyses[0], analyses[1])


if __name__ == "__main__":
    main()
