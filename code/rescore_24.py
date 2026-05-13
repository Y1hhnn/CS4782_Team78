"""Rescore existing ToT JSONL by preferring depth-3 candidates that actually verify.

When the value evaluator can't reliably distinguish correct equations from
near-misses (a known weakness of smaller models), the highest-*scored* final
candidate is often wrong even when a correct candidate is in the top-b.

This script walks each row's trace, finds the highest-scored final candidate
whose equation passes game24.verify(), and uses that as the new answer.
Costs zero API calls -- it just re-picks the winner from the existing trace.

Usage:
    python rescore.py results/runs/tot_b5.jsonl [more.jsonl ...]

Writes <original>.rescored.jsonl alongside each input.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tasks.game24 import verify


def rescore_row(row: dict) -> dict:
    """Return a (possibly updated) row with equation/verified re-picked from trace.

    If any depth-3 candidate verifies, the highest-scored such candidate is used.
    Otherwise the row is returned unchanged.
    """
    inputs = tuple(row["inputs"])
    trace = row.get("trace") or []
    if len(trace) < 3:
        return row

    # trace[2] is the depth-3 / final-candidate frontier, sorted by score desc.
    # Each entry is [score, equation_string] (tuple in Python, list in JSON).
    for entry in trace[2]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        eq = entry[1]
        if eq and verify(eq, inputs):
            # Found a verifying candidate -- pick it
            if eq == row.get("equation") and row.get("verified"):
                return row                          # already what we had
            return {**row, "equation": eq, "verified": True}
    return row


def summarize(in_path: Path) -> None:
    rows = [json.loads(l) for l in open(in_path)]
    if not rows:
        print(f"{in_path.name}: empty")
        return

    new_rows = [rescore_row(r) for r in rows]

    before = sum(r["verified"] for r in rows)
    after = sum(r["verified"] for r in new_rows)
    n = len(rows)

    flipped = [(r["idx"], tuple(r["inputs"]), r["equation"], r2["equation"])
               for r, r2 in zip(rows, new_rows)
               if r2["verified"] and not r["verified"]]

    print(f"\n{in_path.name}:")
    print(f"  Before: {before}/{n} = {100*before/n:.1f}%")
    print(f"  After:  {after}/{n} = {100*after/n:.1f}%   (+{after - before})")

    if flipped:
        print(f"  Flipped {len(flipped)} (✗ -> ✓):")
        for idx, inputs, old_eq, new_eq in flipped:
            print(f"    [{idx:3d}] {inputs}")
            print(f"          was:  {old_eq}")
            print(f"          now:  {new_eq}")

    out_path = in_path.with_suffix(".rescored.jsonl")
    with open(out_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote:  {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path_str in sys.argv[1:]:
        summarize(Path(path_str))


if __name__ == "__main__":
    main()
