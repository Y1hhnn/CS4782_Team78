#!/usr/bin/env python3
"""
Analyze Game of 24 ToT value-score distributions for batched vs unbatched runs.

This uses existing JSONL trace data only. No API calls.

Example (run from the repo root):
python code/analysis/game24/verdict_distribution_24.py \
  --batched results/game24/runs/gem31_tot_b5_batched.jsonl \
  --unbatched results/game24/runs/gem31_tot_b5_unbatched.jsonl \
  --out_csv results/game24/runs/verdict_distribution_b5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


LABEL_SCORE = {
    "sure": 20.0,
    "likely": 1.0,
    "impossible": 0.001,
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_score(score: float, n_votes: int = 3) -> tuple[int, int, int, int, str]:
    """
    Decode a summed score back into approximate vote counts.

    Returns:
      sure_count, likely_count, impossible_count, missing_count, pattern

    If score is exactly 60, this is 3 sure votes.
    If score is 21.001, this is 1 sure + 1 likely + 1 impossible.
    If score is 0, this usually means missing/unparsed output, not impossible.
    """
    if abs(score) < 1e-12:
        return 0, 0, 0, n_votes, "missing"

    best = None
    for s in range(n_votes + 1):
        for l in range(n_votes - s + 1):
            i = n_votes - s - l
            val = 20.0 * s + 1.0 * l + 0.001 * i
            err = abs(score - val)
            cand = (err, s, l, i, val)
            if best is None or cand < best:
                best = cand

    err, s, l, i, val = best
    missing = 0

    if err > 1e-6:
        pattern = f"unknown_score={score:g}"
    else:
        parts = []
        if s:
            parts.append(f"{s}S")
        if l:
            parts.append(f"{l}L")
        if i:
            parts.append(f"{i}I")
        pattern = "+".join(parts) if parts else "missing"

    return s, l, i, missing, pattern


def flatten_trace(rows: list[dict], run_name: str, n_votes: int = 3) -> list[dict]:
    """
    Convert each row's trace into one record per retained candidate.
    Current project trace format:
      trace[depth] = [[score, last_step], [score, last_step], ...]
    """
    records = []

    for row in rows:
        idx = row.get("idx")
        verified = bool(row.get("verified", False))
        trace = row.get("trace") or []

        for depth_i, depth_items in enumerate(trace, start=1):
            if not isinstance(depth_items, list):
                continue

            for rank, item in enumerate(depth_items, start=1):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue

                score = float(item[0])
                step = str(item[1])
                s, l, i, m, pattern = decode_score(score, n_votes=n_votes)

                records.append({
                    "run": run_name,
                    "idx": idx,
                    "depth": depth_i,
                    "rank": rank,
                    "score": score,
                    "step": step,
                    "verified": verified,
                    "sure_votes": s,
                    "likely_votes": l,
                    "impossible_votes": i,
                    "missing_votes": m,
                    "pattern": pattern,
                })

    return records


def summarize(records: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")

    by_key = defaultdict(list)
    for r in records:
        by_key[(r["run"], r["depth"])].append(r)

    for (run, depth), group in sorted(by_key.items()):
        scores = [r["score"] for r in group]
        total_votes = sum(
            r["sure_votes"] + r["likely_votes"] + r["impossible_votes"] + r["missing_votes"]
            for r in group
        )

        sure = sum(r["sure_votes"] for r in group)
        likely = sum(r["likely_votes"] for r in group)
        impossible = sum(r["impossible_votes"] for r in group)
        missing = sum(r["missing_votes"] for r in group)

        pattern_counts = Counter(r["pattern"] for r in group)

        print(f"\n{run} | depth {depth}")
        print(f"  retained candidates: {len(group)}")
        print(f"  score mean / median: {mean(scores):.3f} / {median(scores):.3f}")
        print(f"  score min / max:     {min(scores):.3f} / {max(scores):.3f}")
        print(f"  % score==60:         {100 * sum(abs(x - 60.0) < 1e-9 for x in scores) / len(scores):.1f}%")
        print(f"  % score>=40:         {100 * sum(x >= 40.0 for x in scores) / len(scores):.1f}%")
        print(f"  % score>=20:         {100 * sum(x >= 20.0 for x in scores) / len(scores):.1f}%")

        if total_votes:
            print("  decoded vote distribution:")
            print(f"    sure:       {100 * sure / total_votes:.1f}%")
            print(f"    likely:     {100 * likely / total_votes:.1f}%")
            print(f"    impossible: {100 * impossible / total_votes:.1f}%")
            print(f"    missing:    {100 * missing / total_votes:.1f}%")

        print("  top score patterns:")
        for pat, cnt in pattern_counts.most_common(8):
            print(f"    {pat:<18} {cnt:>5} ({100 * cnt / len(group):.1f}%)")


def compare_overlap(batched_records: list[dict], unbatched_records: list[dict]) -> None:
    """
    Compare ranking overlap for candidates with the same puzzle idx, same depth,
    and same step string.

    This is limited because traces only contain top-b states. Also, if batched
    and unbatched runs explored different trajectories, there may be little overlap.
    """
    b_map = defaultdict(dict)
    u_map = defaultdict(dict)

    for r in batched_records:
        b_map[(r["idx"], r["depth"])][r["step"]] = r

    for r in unbatched_records:
        u_map[(r["idx"], r["depth"])][r["step"]] = r

    shared_keys = sorted(set(b_map) & set(u_map))

    print("\n=== Ranking / overlap comparison on shared puzzle-depth pairs ===")

    by_depth = defaultdict(list)

    for key in shared_keys:
        b_steps = b_map[key]
        u_steps = u_map[key]
        common_steps = set(b_steps) & set(u_steps)

        if not b_steps or not u_steps:
            continue

        b_top = min(b_steps.values(), key=lambda r: r["rank"])["step"]
        u_top = min(u_steps.values(), key=lambda r: r["rank"])["step"]

        rank_deltas = [
            abs(b_steps[s]["rank"] - u_steps[s]["rank"])
            for s in common_steps
        ]

        depth = key[1]
        by_depth[depth].append({
            "overlap": len(common_steps),
            "b_count": len(b_steps),
            "u_count": len(u_steps),
            "same_top1": int(b_top == u_top),
            "rank_delta_mean": mean(rank_deltas) if rank_deltas else None,
            "score_delta_mean": mean(
                abs(b_steps[s]["score"] - u_steps[s]["score"])
                for s in common_steps
            ) if common_steps else None,
        })

    for depth, rows in sorted(by_depth.items()):
        avg_overlap = mean(r["overlap"] for r in rows)
        avg_overlap_rate = mean(
            r["overlap"] / max(r["b_count"], r["u_count"])
            for r in rows
        )
        same_top1_rate = mean(r["same_top1"] for r in rows)

        rank_deltas = [r["rank_delta_mean"] for r in rows if r["rank_delta_mean"] is not None]
        score_deltas = [r["score_delta_mean"] for r in rows if r["score_delta_mean"] is not None]

        print(f"\ndepth {depth}")
        print(f"  puzzle-depth pairs:       {len(rows)}")
        print(f"  avg top-b overlap count:  {avg_overlap:.2f}")
        print(f"  avg top-b overlap rate:   {100 * avg_overlap_rate:.1f}%")
        print(f"  same top-1 candidate:     {100 * same_top1_rate:.1f}%")

        if rank_deltas:
            print(f"  mean rank delta, overlap: {mean(rank_deltas):.2f}")
        if score_deltas:
            print(f"  mean score delta, overlap:{mean(score_deltas):.3f}")


def write_csv(records: list[dict], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "idx", "depth", "rank", "score", "pattern",
        "sure_votes", "likely_votes", "impossible_votes", "missing_votes",
        "verified", "step",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fields})

    print(f"\nWrote CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batched", required=True, help="Batched ToT JSONL")
    parser.add_argument("--unbatched", required=True, help="Unbatched ToT JSONL")
    parser.add_argument("--n_votes", type=int, default=3)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    batched_rows = load_jsonl(args.batched)
    unbatched_rows = load_jsonl(args.unbatched)

    # Compare only same puzzle ids.
    b_by_idx = {r.get("idx"): r for r in batched_rows}
    u_by_idx = {r.get("idx"): r for r in unbatched_rows}
    shared = sorted(set(b_by_idx) & set(u_by_idx))

    batched_shared = [b_by_idx[i] for i in shared]
    unbatched_shared = [u_by_idx[i] for i in shared]

    print(f"Loaded batched rows:   {len(batched_rows)}")
    print(f"Loaded unbatched rows: {len(unbatched_rows)}")
    print(f"Shared puzzle ids:     {len(shared)}")

    batched_records = flatten_trace(batched_shared, "batched", n_votes=args.n_votes)
    unbatched_records = flatten_trace(unbatched_shared, "unbatched", n_votes=args.n_votes)

    all_records = batched_records + unbatched_records

    summarize(all_records, "Verdict / score distribution from retained trace candidates")
    compare_overlap(batched_records, unbatched_records)

    if args.out_csv:
        write_csv(all_records, args.out_csv)


if __name__ == "__main__":
    main()