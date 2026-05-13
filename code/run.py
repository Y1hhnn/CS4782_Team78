"""
CLI Driver for Tree of Thoughts (ToT) Experiments.

This script is the unified entry point for running different problem-solving
methods (IO, CoT, ToT and Creative-Writing variants) across multiple tasks
(Game of 24, Mini Crosswords, Creative Writing). It handles data loading,
asynchronous model calls, rate limiting, and result logging.

Crosswords evaluator selection (--batch / --focused / --verified / --cache):
    --batch {no, half, full}     Call shape per DFS step. Default: full.
                                   no   = one call per testable clue per sibling
                                   half = one call per sibling (clues batched)
                                   full = one call total (siblings + clues batched)
    --focused                    Candidate-level verdict instead of per-clue.
                                   Compose with full/half. Use --top_k to pick
                                   the K most-constrained clues; otherwise all
                                   testable clues are included (declaration order).
    --verified                   Re-confirm every kill via the unbatched
                                   value_prompt (compose with full/half).
    --cache                      Opt-in (clue,constraint)->verdict cache.
                                   Verified stage 2 writes; focused reads for
                                   short-circuit. With --batch no, the
                                   unbatched gold prompt itself populates the
                                   cache (run name 'unbatched_cache'), giving
                                   paper-strict accuracy at much lower cost.
    --top_k INT                  Top-K clues for --focused (0 = all, default).

Creative Writing methods (`--task text`):
    io, cot, tot                 Direct generation, plan+passage, ToT vote.
    best_io, best_cot            Sample k candidates, independently score, pick best.
    plan_only_cot                Two-call CoT: one plan, one passage.
    plan_vote                    Vote on plans, then write one passage.
    tot_score_select             ToT with independent scoring instead of voting.

Usage Examples:
    # Game of 24 with ToT (BFS)
    python run.py --task 24 --method tot --b 5 --n_puzzles 100

    # Crosswords: paper-strict per-clue evaluator
    python run.py --task crosswords --method tot --batch no --paper_split

    # Crosswords: full batch with focused (top-3) + verified + cache
    python run.py --task crosswords --method tot --batch full \\
        --focused --verified --cache --top_k 3 --paper_split

    # Creative Writing: ToT-style plan/passage voting
    python run.py --task text --method tot --n_puzzles 100 --vertex \\
        --model gemini-2.5-flash-lite --thinking_level NONE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import sys
from pathlib import Path

# Task-specific logic and data loaders
from tasks.game24 import load_puzzles as load_24, verify as verify_24
from tasks.crosswords import load_crosswords, check_correctness as verify_cw
from tasks.text import load_text_inputs

# Algorithm implementations
from algorithms.methods_24 import io_solve as io_24, cot_solve as cot_24, tot_solve as tot_solve_24
from algorithms.methods_cw import EvalConfig, dfs_solve_cw, io_solve_cw, cot_solve_cw
from algorithms.methods_text import (
    io_solve_text,
    cot_solve_text,
    tot_solve_text,
    score_passage,
    best_of_k_io_solve_text,
    best_of_k_cot_solve_text,
    tot_score_select_solve_text,
    plan_only_cot_solve_text,
    plan_vote_solve_text,
)

# Core utilities
from core.model import GeminiWrapper, estimate_calls_per_puzzle


# Methods that each task accepts. The unified --method choices are the union;
# we re-validate after parse so users get a clear error per task.
_METHODS_PER_TASK = {
    "24": {"io", "cot", "tot"},
    "crosswords": {"io", "cot", "tot"},
    "text": {
        "io", "cot", "tot",
        "best_io", "best_cot",
        "tot_score_select",
        "plan_only_cot", "plan_vote",
    },
}


async def run_one(args: argparse.Namespace,
                  puzzle_data: any,
                  model: GeminiWrapper,
                  judge_model: GeminiWrapper | None = None) -> dict:
    """
    Execute a single puzzle run based on the specified task and method.
    Returns a dictionary containing the results and meta-information for JSONL logging.
    """
    t0 = time.time()
    prune = not args.no_prune
    backtrack = not args.no_backtrack

    # ==========================================
    # Task 1: Game of 24
    # ==========================================
    if args.task == "24":
        if args.method == "io":
            result_eq = await io_24(puzzle_data, model)
            meta = {"expansions": 1}
        elif args.method == "cot":
            result_eq = await cot_24(puzzle_data, model)
            meta = {"expansions": 1}
        elif args.method == "tot":
            result_eq, trace = await tot_solve_24(
                puzzle_data,
                model,
                b=args.b,
                n_votes=args.n_votes,
                batch=not args.no_batch
            )
            meta = {"expansions": 3, "trace": trace}

        is_success = verify_24(result_eq, puzzle_data)
        return {
            "inputs": puzzle_data,
            "equation": result_eq,
            "verified": is_success,
            "elapsed_s": round(time.time() - t0, 2),
            "steps": meta.get("expansions", 1),
            **meta,
        }

    # ==========================================
    # Task 2: Mini Crosswords
    # ==========================================
    if args.task == "crosswords":
        if args.method == "io":
            # IO and CoT baselines: multi-sample with averaged metrics
            result_state, _, meta = await io_solve_cw(puzzle_data, model, n=10)
        elif args.method == "cot":
            result_state, _, meta = await cot_solve_cw(puzzle_data, model, n=10)
        elif args.method == "tot":
            # ToT DFS Search with ablation parameters.
            # Note: search is answer-blind. The "+best state" oracle metric is
            # reconstructed at analysis time from the trace, not chosen here.
            eval_config = EvalConfig(
                batch=args.batch,
                focused=args.focused,
                verified=args.verified,
                cache=args.cache,
                top_k=args.top_k,
            )
            result_state, _, meta = await dfs_solve_cw(
                initial_state=puzzle_data,
                model=model,
                max_steps=args.max_steps,
                prune=prune,
                backtrack=backtrack,
                eval_config=eval_config,
            )

        # Note: 'verified' strictly applies to the single 'result_state' returned above.
        # For IO/CoT baselines, the true performance is in meta['avg_game_acc'].
        is_success = verify_cw(result_state)

        return {
            "grid": result_state.grid,
            "ground_truth": result_state.ground_truth,
            "rendered": result_state.render(),
            "verified": is_success,
            "elapsed_s": round(time.time() - t0, 2),
            "steps": meta.get("expansions", 1),
            **meta,
        }

    # ==========================================
    # Task 3: Creative Writing
    # ==========================================
    if args.task == "text":
        assert judge_model is not None, "text task requires a judge_model"
        sentences = puzzle_data["sentences"]
        extra: dict = {}

        if args.method == "io":
            passage = await io_solve_text(sentences, model)
        elif args.method == "cot":
            passage = await cot_solve_text(sentences, model)
        elif args.method == "tot":
            passage = await tot_solve_text(
                sentences, model,
                k_plans=args.k_plans, k_passages=args.k_passages, n_votes=args.n_votes,
            )
        elif args.method == "best_io":
            passage, candidate_scores = await best_of_k_io_solve_text(
                sentences, model, judge_model,
                k=args.k, n_score_samples=args.n_score_samples,
            )
            extra["candidate_scores"] = candidate_scores
        elif args.method == "best_cot":
            passage, candidate_scores = await best_of_k_cot_solve_text(
                sentences, model, judge_model,
                k=args.k, n_score_samples=args.n_score_samples,
            )
            extra["candidate_scores"] = candidate_scores
        elif args.method == "tot_score_select":
            passage, info = await tot_score_select_solve_text(
                sentences, model, judge_model,
                k_plans=args.k_plans, k_passages=args.k_passages,
                n_score_samples=args.n_score_samples,
            )
            extra.update(info)
        elif args.method == "plan_only_cot":
            passage, plan = await plan_only_cot_solve_text(sentences, model)
            extra["plan"] = plan
        elif args.method == "plan_vote":
            passage, info = await plan_vote_solve_text(
                sentences, model,
                k_plans=args.k_plans, n_votes=args.n_votes,
            )
            extra.update(info)
        else:
            raise ValueError(f"unknown text method: {args.method}")

        score = await score_passage(passage, judge_model, n_samples=args.judge_samples)

        return {
            "id": puzzle_data["id"],
            "sentences": sentences,
            "method": args.method,
            "model": args.model,
            "judge_model": args.judge_model,
            "passage": passage,
            "score": score,
            "elapsed_s": round(time.time() - t0, 2),
            **extra,
        }

    return {}


def _default_out_path(args: argparse.Namespace) -> Path:
    """Auto-generate output path under results/<task>/runs/<name>.jsonl."""
    if args.task == "24":
        batch_flag = "batched" if not args.no_batch else "unbatched"
        name = f"gem31_{args.method}_b{args.b}_{batch_flag}"
        return Path(f"results/game24/runs/{name}.jsonl")

    if args.task == "crosswords":
        if args.method in ("io", "cot"):
            name = f"cw_gem31_{args.method}"
        else:
            eval_flag = EvalConfig(
                batch=args.batch,
                focused=args.focused,
                verified=args.verified,
                cache=args.cache,
                top_k=args.top_k,
            ).name()
            prune_flag = "noprune" if args.no_prune else "prune"
            backtrack_flag = "nobacktrack" if args.no_backtrack else "backtrack"
            name = f"cw_gem31_tot_{eval_flag}_s{args.max_steps}_{prune_flag}_{backtrack_flag}"
        if args.paper_split:
            name += "_papersplit"
        return Path(f"results/crosswords/runs/{name}.jsonl")

    if args.task == "text":
        safe_model = args.model.replace("/", "_")
        return Path(f"results/text/runs/text_{args.method}_{safe_model}.jsonl")

    raise ValueError(f"Unknown task: {args.task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tree of Thoughts experiments.")

    # Task & Method selection
    parser.add_argument("--task", type=str, required=True,
                        choices=["24", "crosswords", "text"], help="Task to run.")
    parser.add_argument("--method", type=str, required=True,
                        choices=sorted(set().union(*_METHODS_PER_TASK.values())),
                        help="Solver method. Valid set depends on --task.")

    # Model configuration
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview", help="Model string.")
    parser.add_argument("--thinking_level", type=str, default="MINIMAL", help="Thinking level (NONE, MINIMAL, etc.).")
    parser.add_argument("--rpm", type=int, default=15, help="Rate limit: requests per minute (0 disables).")
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Google AI Studio.")
    parser.add_argument("--location", type=str, default=None,
                        help="Vertex region (auto-picked if omitted).")

    # Creative Writing: judge model (LLM-as-judge coherence scorer)
    parser.add_argument("--judge_model", type=str, default="gemini-3.1-flash-lite-preview",
                        help="Text task: judge model for coherence scoring.")
    parser.add_argument("--judge_thinking_level", type=str, default="MINIMAL",
                        choices=["MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"],
                        help="Text task: thinking level for the judge model.")
    parser.add_argument("--judge_samples", type=int, default=3,
                        help="Text task: number of judge samples to average per passage.")

    # Execution & Concurrency
    parser.add_argument("--n_puzzles", type=int, default=100, help="Number of puzzles/inputs to solve.")
    parser.add_argument("--paper_split", action="store_true", help="Crosswords: 20-puzzle subset (indices 0, 5, 10...95).")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent puzzle-solving tasks.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output JSONL path (default: results/<task>/runs/<auto>.jsonl).")

    # ToT Search Parameters (Game of 24)
    parser.add_argument("--b", type=int, default=5, help="Beam size for BFS (Game of 24).")

    # ToT Search Parameters (Crosswords)
    parser.add_argument("--max_steps", type=int, default=100, help="Max expansions for DFS (Crosswords).")

    # Ablation / Evaluator Controls
    parser.add_argument("--no_prune", action="store_true", help="Disable pruning (accept all candidate states).")
    parser.add_argument("--no_backtrack", action="store_true", help="Disable backtracking (greedy search).")

    # Crosswords value evaluator (level + composable features). 'no' is the
    # paper-strict per-clue evaluator and is incompatible with --focused /
    # --verified / --cache.
    parser.add_argument("--batch", type=str, default="full", choices=["no", "half", "full"],
                        help="Crosswords value evaluator level: no (per-clue), half (per-sibling), full (one call).")
    parser.add_argument("--focused", action="store_true",
                        help="Crosswords: candidate-level verdict (declaration order; --top_k filters).")
    parser.add_argument("--verified", action="store_true",
                        help="Crosswords: re-confirm every kill via unbatched value_prompt.")
    parser.add_argument("--cache", action="store_true",
                        help="Crosswords: opt-in (clue, constraint)->verdict cache. Only verified stage 2 writes; focused reads for short-circuit.")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Crosswords --focused: top-K most-constrained clues to include (0 = all testable, default).")

    # Game of 24 ToT
    parser.add_argument("--no_batch", action="store_true",
                        help="Game of 24: disable batched value evaluation (use one call per state).")
    parser.add_argument("--n_votes", type=int, default=3,
                        help="Number of votes (Game of 24 ToT evaluator; also reused for text ToT vote prompts).")

    # Creative Writing search parameters
    parser.add_argument("--k", type=int, default=5,
                        help="Text task: number of samples for best_io / best_cot.")
    parser.add_argument("--k_plans", type=int, default=5,
                        help="Text task: number of candidate plans (tot / tot_score_select / plan_vote).")
    parser.add_argument("--k_passages", type=int, default=5,
                        help="Text task: number of candidate passages (tot / tot_score_select).")
    parser.add_argument("--n_score_samples", type=int, default=3,
                        help="Text task: judge samples per candidate during scoring-based selection.")

    args = parser.parse_args()

    # Method/task compatibility
    allowed = _METHODS_PER_TASK[args.task]
    if args.method not in allowed:
        parser.error(
            f"--method {args.method} is not valid for --task {args.task}. "
            f"Valid methods: {sorted(allowed)}"
        )

    # Crosswords evaluator validation
    if args.task == "crosswords" and args.method == "tot":
        if args.batch == "no" and (args.focused or args.verified):
            parser.error("--batch no is incompatible with --focused / --verified.")

    return args


def _build_model(args, *, model_name: str, thinking: str | None) -> GeminiWrapper:
    rpm = args.rpm if args.rpm > 0 else None
    return GeminiWrapper(
        model=model_name,
        thinking_level=thinking,
        rpm=rpm,
        vertex=args.vertex,
        location=args.location,
    )


async def main():
    args = parse_args()

    out_path = Path(args.out) if args.out else _default_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and out_path.exists():
        out_path.unlink()

    # Resume: detect already-completed rows. Text uses 'id', others use 'idx'.
    id_key = "id" if args.task == "text" else "idx"
    completed: set = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                try:
                    completed.add(json.loads(line).get(id_key))
                except json.JSONDecodeError:
                    pass

    # Load task data and map to strict IDs to maintain correct indices when subsetting
    indexed_puzzles: list[tuple[int, any]] = []
    if args.task == "24":
        start_idx = 900
        raw_puzzles = load_24(start=start_idx, end=start_idx + args.n_puzzles)
        indexed_puzzles = list(enumerate(raw_puzzles, start=start_idx))
    elif args.task == "crosswords":
        # Load slightly more just in case to cover up to index 95
        raw_puzzles = load_crosswords(end=max(156, args.n_puzzles))
        if args.paper_split:
            test_indices = list(range(0, 100, 5))  # 0, 5, 10 ... 95
            indexed_puzzles = [(i, raw_puzzles[i]) for i in test_indices]
        else:
            indexed_puzzles = list(enumerate(raw_puzzles[:args.n_puzzles]))
    elif args.task == "text":
        raw_inputs = load_text_inputs(end=args.n_puzzles)
        indexed_puzzles = [(row["id"], row) for row in raw_inputs]
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Filter out already completed puzzles
    todo = [(idx, p) for idx, p in indexed_puzzles if idx not in completed]

    if not todo:
        print("All selected puzzles are already completed. Use --overwrite to rerun.")
        return

    # Initialize Model Wrapper(s)
    thinking = None if args.thinking_level == "NONE" else args.thinking_level
    model = _build_model(args, model_name=args.model, thinking=thinking)

    judge_model: GeminiWrapper | None = None
    if args.task == "text":
        judge_thinking = None if args.judge_thinking_level == "NONE" else args.judge_thinking_level
        judge_model = _build_model(args, model_name=args.judge_model, thinking=judge_thinking)

    # Print summary and budget estimation
    print(f"Task: {args.task} | Method: {args.method} | Model: {args.model}")
    if args.task == "text":
        print(f"Judge: {args.judge_model} (thinking={args.judge_thinking_level})")
    print(f"Total: {len(indexed_puzzles)} | Done: {len(completed)} | To Run: {len(todo)}")
    if args.task != "text":
        cpp = estimate_calls_per_puzzle(
            args.method,
            task=args.task,
            b=args.b,
            n_votes=args.n_votes,
            avg_depth=args.max_steps // 5,
        )
        print(f"Budget Estimate: ~{cpp} calls/puzzle at {args.rpm} RPM")
    print(f"Output: {out_path}\n")

    # Limit maximum concurrent requests to avoid triggering rate limit spikes
    sem = asyncio.Semaphore(args.concurrency)

    async def run_with_sem(idx: int, data: any) -> dict:
        async with sem:
            result = await run_one(args, data, model, judge_model)
            result[id_key] = idx
            return result

    # Launch concurrent tasks
    tasks = [run_with_sem(idx, p) for idx, p in todo]

    # Stream results to file as they complete
    with open(out_path, "a") as fp:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            fp.write(json.dumps(res) + "\n")
            fp.flush()

            # Formatting terminal output per task
            if args.task == "text":
                sid = res.get("id", "?")
                score = res.get("score", 0.0)
                print(f"[{sid:3d}] score={score:.2f} | Time: {res.get('elapsed_s', '?')}s")
            elif args.task == "crosswords" and args.method in ["io", "cot"]:
                acc_str = f"{res.get('avg_game_acc', 0.0) * 100:.0f}%"
                print(f"[{res.get('idx', '?'):3d}] ~{acc_str} | Steps: {res.get('steps', '?')} | Time: {res.get('elapsed_s', '?')}s")
            else:
                status = "✓" if res.get("verified", False) else "✗"
                print(f"[{res.get('idx', '?'):3d}] {status} | Steps: {res.get('steps', '?')} | Time: {res.get('elapsed_s', '?')}s")

    # Text-task end-of-run summary (mean score, tokens, cost)
    if args.task == "text":
        all_rows = [json.loads(l) for l in open(out_path)]
        valid_scores = [r["score"] for r in all_rows if r.get("score", 0) > 0]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        total_prompt = model.usage.prompt_tokens + (judge_model.usage.prompt_tokens if judge_model else 0)
        total_completion = model.usage.completion_tokens + (judge_model.usage.completion_tokens if judge_model else 0)
        total_cost = model.usage.cost_usd(args.model) + (judge_model.usage.cost_usd(args.judge_model) if judge_model else 0.0)

        print()
        print(f"Cumulative scored rows: {len(valid_scores)}")
        print(f"Mean coherence score: {mean_score:.2f}")
        print(f"Tokens this run: {total_prompt:,} prompt + {total_completion:,} completion")
        print(f"Cost this run: ${total_cost:.4f}")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Workaround for Windows environments (if applicable)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
