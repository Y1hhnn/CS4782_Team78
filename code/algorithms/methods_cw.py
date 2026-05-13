"""DFS search algorithm for Mini Crosswords with three-level batched evaluation.

Search is answer-blind: the trace records every visited state so analysis
tooling can recover oracle metrics (e.g. "+best state") after the fact.

Evaluator design (level x optional features, selected via run.py CLI):

    Level (call shape per DFS step, --batch):
        no    - one call per testable clue per sibling (paper baseline)
        half  - one call per sibling, all of its testable clues batched in
        full  - one call total, all siblings x all testable clues batched in

    Optional features (compose on top of 'half' or 'full'):
        --focused   Single candidate-level verdict ("viable" / "not_viable")
                    per sibling instead of per-clue verdicts. Clues are
                    rendered in declaration order. With --top_k N (>0), the
                    N most-constrained testable clues are selected; without
                    it, every testable clue is included. If every sibling
                    is judged not_viable, falls back to the basic variant
                    of the same level for that step (a sanity check against
                    over-pruning).
        --verified  Re-confirm every kill via the unbatched value_prompt
                    before letting it prune the search. Surviving kills
                    are gold-prompt-confirmed.
        --cache     Opt-in per-puzzle (clue_text, constraint) -> verdict
                    cache. Only verified stage 2 writes it (gold prompt is
                    authoritative); focused (and any other reader) short-
                    circuits a candidate when one of its testable clues is
                    cached as 'impossible'. Useful when DFS revisits the
                    same partially-filled board across backtracks.

    'no' is paper-strict and incompatible with --focused / --verified.
    'no' may be combined with --cache: the gold-prompt verdicts written by
    Level 3 are then memoized within the puzzle, so DFS siblings that share
    a (clue, constraint) skip the redundant API call -- a large cost
    reduction at the price of sibling-order independence (the run name
    becomes 'unbatched_cache' to keep results separable).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from tasks.crosswords import BoardState, filter_hallucinations

try:
    from prompts.prompts_cw import (
        BATCHED_VALUE_PROMPT,
        CANDIDATE_VERDICT_PROMPT,
        cot_prompt,
        propose_prompt,
        standard_prompt,
        value_prompt,
    )
except ImportError as e:
    raise ImportError("Check prompts/prompts_cw.py") from e


# --------------------------------------------------------------------
# Eval configuration
# --------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    """Selects the value evaluator. See module docstring for semantics.

    batch:    'no' | 'half' | 'full'      call shape per DFS step
    focused:  candidate-level verdict, optional top-K filtering
    verified: 2-stage; re-confirm kills via the unbatched gold prompt
    cache:    opt-in; only verified-stage-2 writes the cache, focused
              (and any other reader) short-circuits on cached impossibles
    top_k:    0 = include every testable clue in focused prompts;
              >0 = pick that many most-constrained, render in declaration
              order. Ignored when --focused is off.
    """
    batch: str = "full"      # 'no' | 'half' | 'full'
    focused: bool = False
    verified: bool = False
    cache: bool = False
    top_k: int = 0

    def name(self) -> str:
        if self.batch == "no":
            return "unbatched_cache" if self.cache else "unbatched"
        flags = []
        if self.focused:
            flags.append("focused")
            if self.top_k > 0:
                flags.append(f"k{self.top_k}")
        if self.verified:
            flags.append("verified")
        if self.cache:
            flags.append("cache")
        suffix = "_".join(flags) if flags else "basic"
        return f"{self.batch}_{suffix}"


@dataclass
class EvalStats:
    """Mutable per-puzzle telemetry. The DFS owns one of these and threads it
    through the evaluator stack; each LLM call site bumps the relevant
    counter so the meta dict can report a budget breakdown.

    Counts are in *underlying* generate-call units (n samples count as n).
    Cache hits do not count.
    """
    llm_calls_propose: int = 0
    llm_calls_value: int = 0      # stage-1 (basic / focused) value calls
    llm_calls_verified: int = 0   # stage-2 verified re-confirmation calls
    fallback_count: int = 0       # focused steps that fell back to basic
    verified_attempts: int = 0    # siblings that ran stage 2
    verified_confirmed_kills: int = 0
    verified_restored: int = 0

    @property
    def llm_calls_total(self) -> int:
        return (self.llm_calls_propose
                + self.llm_calls_value
                + self.llm_calls_verified)

    def to_dict(self) -> dict:
        return {
            "llm_calls_propose": self.llm_calls_propose,
            "llm_calls_value": self.llm_calls_value,
            "llm_calls_verified": self.llm_calls_verified,
            "llm_calls_total": self.llm_calls_total,
            "fallback_count": self.fallback_count,
            "verified_attempts": self.verified_attempts,
            "verified_confirmed_kills": self.verified_confirmed_kills,
            "verified_restored": self.verified_restored,
        }


# --------------------------------------------------------------------
# Helper: Parse 5x5 Grid (IO/CoT baselines)
# --------------------------------------------------------------------

def _parse_grid_from_text(text: str) -> tuple[tuple[str, ...], ...]:
    block = text.split("Output:\n")[-1]
    lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
    lines = lines[-5:]

    grid = []
    for line in lines:
        tokens = line.split()[:5]
        word = "".join(tokens).upper()
        if len(word) != 5 or not word.isalpha():
            return tuple(tuple(["_"] * 5) for _ in range(5))
        grid.append(tuple(word))

    if len(grid) != 5:
        return tuple(tuple(["_"] * 5) for _ in range(5))
    return tuple(grid)


def _state_letter_accuracy(state: BoardState) -> float:
    """Oracle score used only for logging and analysis, not for search control."""
    if not state.ground_truth:
        return 0.0
    flat = [c for row in state.grid for c in row]
    truth = list(state.ground_truth)
    correct = sum(1 for o, t in zip(flat, truth) if o != "_" and o == t)
    return correct / 25.0


def _state_word_accuracy(state: BoardState) -> float:
    if not state.ground_truth:
        return 0.0
    current_words = []
    for i in range(5):
        current_words.append("".join(state.grid[i]))
    for i in range(5):
        current_words.append("".join([state.grid[r][i] for r in range(5)]))
    gt_letters = list(state.ground_truth)
    gt_words = []
    for i in range(5):
        gt_words.append("".join(gt_letters[i*5:(i+1)*5]))
    for i in range(5):
        gt_words.append("".join(gt_letters[i::5]))
    correct = sum(1 for c, g in zip(current_words, gt_words) if c == g)
    return correct / 10.0


# --------------------------------------------------------------------
# IO & CoT Baselines (Standard Paper Version: No Oracle)
# --------------------------------------------------------------------

async def _solve_baseline_with_averages(state: BoardState, model, prompt_template, n=10):
    """IO/CoT baseline: sample n outputs and average metrics, as in paper."""
    clues_text = "\n".join([f"{cid}. {text}" for cid, text in state.inputs])
    prompt = prompt_template.format(input=clues_text)
    outputs = await model.generate(prompt, temperature=0.7, n=n)
    results = []
    for out_text in outputs:
        grid = _parse_grid_from_text(out_text)
        s = state.apply_full_grid(grid)
        results.append({
            "letter_acc": _state_letter_accuracy(s),
            "word_acc": _state_word_accuracy(s),
            "game_acc": 1.0 if [c for r in s.grid for c in r] == list(s.ground_truth) else 0.0,
            "state": s,
        })
    meta = {
        "avg_letter_acc": sum(r["letter_acc"] for r in results) / n,
        "avg_word_acc": sum(r["word_acc"] for r in results) / n,
        "avg_game_acc": sum(r["game_acc"] for r in results) / n,
        "samples": n,
    }
    return results[0]["state"], False, meta


async def io_solve_cw(state: BoardState, model, n=10):
    return await _solve_baseline_with_averages(state, model, standard_prompt, n=n)


async def cot_solve_cw(state: BoardState, model, n=10):
    return await _solve_baseline_with_averages(state, model, cot_prompt, n=n)


# --------------------------------------------------------------------
# ToT Proposal Generator
# --------------------------------------------------------------------

async def generate_proposals(
    state: BoardState, model, stats: EvalStats | None = None,
) -> list[tuple[str, str]]:
    board_status = state.render()
    unfilled_info = [
        f"{cid}. {dict(state.inputs)[cid]} | constraint: {state.get_constraint_for(cid)}"
        for cid in state.unfilled_clues
    ]
    prompt = propose_prompt.format(
        input=f"Current board:\n{board_status}\n\nUnfilled:\n" + "\n".join(unfilled_info)
    )
    if stats is not None:
        stats.llm_calls_propose += 5  # n below
    outputs = await model.generate(prompt, temperature=0.7, n=5)

    weights = {"certain": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}
    move_scores = defaultdict(float)
    pattern = r"([hv][1-5])\.\s*([a-zA-Z]{5})\s*\((certain|high|medium|low)\)"

    for out in outputs:
        matches = re.findall(pattern, out.lower())
        for cid, word, conf in matches:
            word = word.upper()
            if cid in state.unfilled_clues and filter_hallucinations(
                [word], state.get_constraint_for(cid)
            ):
                move_scores[(cid, word)] += weights.get(conf, 0.0)

    return [move for move, _ in sorted(move_scores.items(), key=lambda x: -x[1])]


# --------------------------------------------------------------------
# Shared parsing / clue helpers
# --------------------------------------------------------------------

_VERDICT_LINE = re.compile(
    r"^\s*state\s+(\d+)\s*:\s*(sure|maybe|impossible)\s*\.?\s*$",
    re.IGNORECASE,
)
_CANDIDATE_LINE = re.compile(
    r"^\s*candidate\s+(\d+)\s*:\s*(viable|not_viable|not viable)\s*\.?\s*$",
    re.IGNORECASE,
)


def _testable_clues(state: BoardState) -> list[tuple[str, str, str]]:
    """Return [(clue_id, clue_text, constraint), ...] for clues with < 4 blanks,
    in **declaration order** (h1..h5, v1..v5). Paper-faithful ordering.
    """
    inputs_dict = dict(state.inputs)
    items = []
    for cid in state.unfilled_clues:
        constraint = state.get_constraint_for(cid)
        if constraint.count("_") >= 4:
            continue
        items.append((cid, inputs_dict[cid], constraint))
    return items


def _focused_clues(state: BoardState, top_k: int) -> list[tuple[str, str]]:
    """Clues a focused evaluator should ask about.

    Selection: if top_k <= 0, include every testable clue. Otherwise pick the
    top-K most-constrained (fewest blanks) -- but render them in **declaration
    order** in the prompt so position effects don't differ across DFS steps.
    """
    items = _testable_clues(state)
    if 0 < top_k < len(items):
        ranked = sorted(items, key=lambda x: x[2].count("_"))[:top_k]
        keep = set(ranked)
        items = [c for c in items if c in keep]
    return [(ct, c) for _, ct, c in items]


def _parse_unbatched_verdict(text: str) -> str:
    """The unbatched value_prompt convention: verdict on the last non-empty line."""
    if not text:
        return "maybe"
    last = text.strip().lower().splitlines()[-1].strip().rstrip(".")
    for v in ("impossible", "sure", "maybe"):
        if last == v or last.endswith(" " + v):
            return v
    return "maybe"


def _cache_short_circuit(state: BoardState, cache) -> tuple[str, str] | None:
    """Return the first (clue_text, constraint) cached as 'impossible', or None.
    A None cache (when --cache is off) always returns None: no short-circuit.
    """
    if cache is None:
        return None
    for _, ct, c in _testable_clues(state):
        if cache.get((ct, c)) == "impossible":
            return (ct, c)
    return None


# --------------------------------------------------------------------
# Level 3: Unbatched (one call per testable clue per sibling)
# --------------------------------------------------------------------

async def _unbatched_one_clue(
    clue_text: str, constraint: str, model, cache,
    stats: EvalStats | None = None, bucket: str = "value",
) -> str:
    """Run value_prompt once for (clue_text, constraint). Returns the verdict.
    Reads/writes cache when provided so verified stage 2 is O(1) on repeats.
    `bucket` selects which stats counter to bump on a real (non-cached) call:
    'value' (Level-3 paper-strict) or 'verified' (verified stage 2).
    """
    if cache is not None:
        cached = cache.get((clue_text, constraint))
        if cached is not None:
            return cached
    prompt = value_prompt.format(input=f"{clue_text}: {constraint}")
    if stats is not None:
        if bucket == "verified":
            stats.llm_calls_verified += 1
        else:
            stats.llm_calls_value += 1
    res = await model.generate(prompt, temperature=0.0, n=1)
    out_text = res[0] if res and res[0] else ""
    verdict = _parse_unbatched_verdict(out_text)
    if cache is not None:
        cache[(clue_text, constraint)] = verdict
    return verdict


async def _unbatched_evaluate_sibling(
    state: BoardState, model, cache, stats: EvalStats | None = None,
) -> dict:
    """Per-clue unbatched evaluation. Short-circuits on the first impossible
    (paper-style). Returns {'viable': bool, 'killed_by': [(ct, c), ...]}.
    """
    for _, ct, c in _testable_clues(state):
        verdict = await _unbatched_one_clue(ct, c, model, cache, stats=stats, bucket="value")
        if verdict == "impossible":
            return {"viable": False, "killed_by": [(ct, c)]}
    return {"viable": True, "killed_by": []}


# --------------------------------------------------------------------
# Level 1: Full batch (one call across all siblings x all testable clues)
# --------------------------------------------------------------------

async def _full_basic(
    sibling_states: list[BoardState], model, cache,
    stats: EvalStats | None = None,
) -> list[dict]:
    """One LLM call across all siblings x all testable clues, flat per-clue
    verdicts. Does not write the cache: only gold-prompt (unbatched) verdicts
    seed the cache, so focused short-circuits and verified stage 2 only ever
    trust confirmed verdicts.
    """
    del cache  # cache is only written by _unbatched_one_clue
    blocks: list[str] = []
    sibling_of_block: list[int] = []
    clue_of_block: list[tuple[str, str]] = []
    for s_idx, state in enumerate(sibling_states):
        for _, ct, c in _testable_clues(state):
            blocks.append(f"State {len(blocks)+1}: {ct}\nConstraint: {c}")
            sibling_of_block.append(s_idx)
            clue_of_block.append((ct, c))

    if not blocks:
        return [{"viable": True, "killed_by": []} for _ in sibling_states]

    prompt = BATCHED_VALUE_PROMPT.format(states="\n\n".join(blocks))
    if stats is not None:
        stats.llm_calls_value += 1
    res = await model.generate(prompt, temperature=0.0, n=1)
    out_text = res[0] if res and res[0] else ""

    verdicts = ["maybe"] * len(blocks)
    for line in out_text.splitlines():
        m = _VERDICT_LINE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(blocks):
                verdicts[idx] = m.group(2).lower()

    out: list[dict] = [{"viable": True, "killed_by": []} for _ in sibling_states]
    for idx, v in enumerate(verdicts):
        if v == "impossible":
            s_idx = sibling_of_block[idx]
            out[s_idx]["viable"] = False
            out[s_idx]["killed_by"].append(clue_of_block[idx])
    return out


async def _full_focused(
    sibling_states: list[BoardState], model, top_k: int, cache,
    stats: EvalStats | None = None,
) -> list[dict] | None:
    """One LLM call: candidate-level verdict on each sibling's top-K clues.
    Cache short-circuits any sibling whose clue is already known impossible.
    Returns None if every sibling is judged not_viable (signals a fallback
    to _full_basic for this DFS step)."""
    pre_killed: list[tuple[str, str] | None] = [None] * len(sibling_states)
    asked_idx: list[int] = []
    asked_top_k: list[list[tuple[str, str]]] = []

    for s_idx, state in enumerate(sibling_states):
        cached_kill = _cache_short_circuit(state, cache)
        if cached_kill is not None:
            pre_killed[s_idx] = cached_kill
            continue
        top = _focused_clues(state, top_k)
        if not top:
            continue  # vacuously viable -- no testable clues
        asked_idx.append(s_idx)
        asked_top_k.append(top)

    results: list[dict | None] = [None] * len(sibling_states)
    for s_idx in range(len(sibling_states)):
        if pre_killed[s_idx] is not None:
            results[s_idx] = {"viable": False, "killed_by": [pre_killed[s_idx]]}
        elif s_idx not in asked_idx:
            results[s_idx] = {"viable": True, "killed_by": []}

    if asked_idx:
        prompt_blocks = []
        for new_n, top in enumerate(asked_top_k, start=1):
            lines = [f"Candidate {new_n}:"]
            for ct, c in top:
                lines.append(f"- {ct}: {c}")
            prompt_blocks.append("\n".join(lines))
        prompt = CANDIDATE_VERDICT_PROMPT.format(candidates="\n\n".join(prompt_blocks))
        if stats is not None:
            stats.llm_calls_value += 1
        res = await model.generate(prompt, temperature=0.0, n=1)
        out_text = res[0] if res and res[0] else ""

        # Default 'viable' on parse fail so a malformed reply does not prune.
        verdicts = ["viable"] * len(asked_idx)
        for line in out_text.splitlines():
            m = _CANDIDATE_LINE.match(line)
            if m:
                n_idx = int(m.group(1)) - 1
                if 0 <= n_idx < len(asked_idx):
                    verdicts[n_idx] = m.group(2).lower().replace(" ", "_")

        for new_n, s_idx in enumerate(asked_idx):
            if verdicts[new_n] == "not_viable":
                results[s_idx] = {"viable": False, "killed_by": list(asked_top_k[new_n])}
            else:
                results[s_idx] = {"viable": True, "killed_by": []}

    final_results = [r for r in results if r is not None]
    if final_results and not any(r["viable"] for r in final_results):
        return None
    return final_results


# --------------------------------------------------------------------
# Level 2: Half batch (one call per sibling)
# --------------------------------------------------------------------

async def _half_basic(
    sibling_states: list[BoardState], model, cache,
    stats: EvalStats | None = None,
) -> list[dict]:
    """One LLM call per sibling, flat per-clue verdicts. Does not write the
    cache (see _full_basic for why)."""
    del cache

    async def _one(state: BoardState) -> dict:
        clues = [(ct, c) for _, ct, c in _testable_clues(state)]
        if not clues:
            return {"viable": True, "killed_by": []}
        blocks = [f"State {i+1}: {ct}\nConstraint: {c}" for i, (ct, c) in enumerate(clues)]
        prompt = BATCHED_VALUE_PROMPT.format(states="\n\n".join(blocks))
        if stats is not None:
            stats.llm_calls_value += 1
        res = await model.generate(prompt, temperature=0.0, n=1)
        out_text = res[0] if res and res[0] else ""

        verdicts = ["maybe"] * len(clues)
        for line in out_text.splitlines():
            m = _VERDICT_LINE.match(line)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(clues):
                    verdicts[idx] = m.group(2).lower()

        impossibles = [k for v, k in zip(verdicts, clues) if v == "impossible"]
        return {"viable": not impossibles, "killed_by": impossibles}

    return list(await asyncio.gather(*[_one(s) for s in sibling_states]))


async def _half_focused(
    sibling_states: list[BoardState], model, top_k: int, cache,
    stats: EvalStats | None = None,
) -> list[dict] | None:
    """One LLM call per sibling: candidate-level verdict on its top-K clues.
    Returns None if every sibling judged not_viable (fallback signal)."""

    async def _one(state: BoardState) -> dict:
        cached_kill = _cache_short_circuit(state, cache)
        if cached_kill is not None:
            return {"viable": False, "killed_by": [cached_kill]}
        top = _focused_clues(state, top_k)
        if not top:
            return {"viable": True, "killed_by": []}
        lines = ["Candidate 1:"]
        for ct, c in top:
            lines.append(f"- {ct}: {c}")
        prompt = CANDIDATE_VERDICT_PROMPT.format(candidates="\n".join(lines))
        if stats is not None:
            stats.llm_calls_value += 1
        res = await model.generate(prompt, temperature=0.0, n=1)
        out_text = res[0] if res and res[0] else ""

        verdict = "viable"
        for line in out_text.splitlines():
            m = _CANDIDATE_LINE.match(line)
            if m and int(m.group(1)) == 1:
                verdict = m.group(2).lower().replace(" ", "_")
                break

        if verdict == "not_viable":
            return {"viable": False, "killed_by": list(top)}
        return {"viable": True, "killed_by": []}

    results = list(await asyncio.gather(*[_one(s) for s in sibling_states]))
    if results and not any(r["viable"] for r in results):
        return None
    return results


# --------------------------------------------------------------------
# Top-level dispatcher
# --------------------------------------------------------------------

async def evaluate_siblings(
    sibling_states: list[BoardState], model, config: EvalConfig, cache,
    stats: EvalStats | None = None,
) -> tuple[list[dict], bool]:
    """Evaluate every sibling and return ``(results, fallback_used)``.

    Each result dict has:
        viable: bool                       final viability after stage 2
        killed_by: list[(ct, c)]           clues that triggered the kill (stage 1)
        verified: bool | None              stage-2 outcome:
                                             None  -> stage 2 didn't run
                                             True  -> confirmed dead
                                             False -> stage 2 disagreed (restored)
        restored_by_verification: bool     stage-1 killed but stage-2 saved it

    ``fallback_used`` is True when --focused had to retry the step under
    --basic because every candidate came back not_viable.
    """
    if not sibling_states:
        return [], False

    fallback_used = False
    if config.batch == "no":
        # Level 3 is paper-strict by default: every clue gets an independent
        # gold-prompt call. With --cache, gold verdicts are memoized within
        # the puzzle so siblings sharing a (clue, constraint) deduplicate to
        # a single API call -- this trades sibling-order independence for a
        # large cost reduction (cf. EvalConfig.cache; 'unbatched_cache' name).
        cache_arg = cache if config.cache else None
        results = list(await asyncio.gather(*[
            _unbatched_evaluate_sibling(s, model, cache_arg, stats) for s in sibling_states
        ]))
    elif config.batch == "half":
        if config.focused:
            stage1 = await _half_focused(sibling_states, model, config.top_k, cache, stats)
            if stage1 is None:
                fallback_used = True
                if stats is not None:
                    stats.fallback_count += 1
                stage1 = await _half_basic(sibling_states, model, cache, stats)
            results = stage1
        else:
            results = await _half_basic(sibling_states, model, cache, stats)
    elif config.batch == "full":
        if config.focused:
            stage1 = await _full_focused(sibling_states, model, config.top_k, cache, stats)
            if stage1 is None:
                fallback_used = True
                if stats is not None:
                    stats.fallback_count += 1
                stage1 = await _full_basic(sibling_states, model, cache, stats)
            results = stage1
        else:
            results = await _full_basic(sibling_states, model, cache, stats)
    else:
        raise ValueError(f"Unknown batch level: {config.batch!r}")

    # Default verification fields (stage 2 may overwrite below).
    for r in results:
        r.setdefault("verified", None)
        r.setdefault("restored_by_verification", False)

    if not config.verified:
        return results, fallback_used

    # Stage 2: re-confirm every kill against the unbatched value_prompt.
    confirm_tasks = []
    confirm_owner: list[int] = []
    for s_idx, r in enumerate(results):
        if r["viable"]:
            continue
        for ct, c in r["killed_by"]:
            confirm_tasks.append(
                _unbatched_one_clue(ct, c, model, cache, stats=stats, bucket="verified")
            )
            confirm_owner.append(s_idx)

    n_attempts = sum(1 for r in results if not r["viable"])
    if stats is not None:
        stats.verified_attempts += n_attempts

    if not confirm_tasks:
        return results, fallback_used

    confirms = await asyncio.gather(*confirm_tasks)
    confirmed_dead: set[int] = {
        s_idx for s_idx, v in zip(confirm_owner, confirms) if v == "impossible"
    }

    # Verified semantics: only confirmed impossibles prune. Stage-1 kills
    # without a confirming stage-2 verdict are restored.
    for s_idx, r in enumerate(results):
        if r["viable"]:
            continue
        if s_idx in confirmed_dead:
            r["verified"] = True
            if stats is not None:
                stats.verified_confirmed_kills += 1
        else:
            r["verified"] = False
            r["restored_by_verification"] = True
            r["viable"] = True
            if stats is not None:
                stats.verified_restored += 1

    return results, fallback_used


# --------------------------------------------------------------------
# Main DFS Solver
# --------------------------------------------------------------------

async def dfs_solve_cw(
    initial_state: BoardState,
    model,
    max_steps: int = 100,
    prune: bool = True,
    backtrack: bool = True,
    eval_config: EvalConfig = EvalConfig(),
) -> tuple[BoardState, bool, dict]:
    """Answer-blind DFS for ToT Crosswords with pluggable batched evaluator.

    Returns: (deepest_explored_state, is_success, meta).
    The trace records every visited state via `filled` so analysis tooling can
    reconstruct oracle metrics ("+best state") without the search ever seeing
    the ground truth.
    """
    stack = [initial_state]
    deepest_state = initial_state
    expansions = 0
    trace: list[dict] = []
    # Per-puzzle cache, only when --cache is set. None disables short-circuit
    # everywhere -- evaluators check `if cache is not None` before reading.
    cache: dict | None = {} if eval_config.cache else None
    stats = EvalStats()
    t_start = time.time()

    def _build_meta() -> dict:
        # Use 'eval_' prefix on the evaluator-config fields so they don't
        # shadow the puzzle-level 'verified' / 'steps' keys when the meta
        # dict is spread into the JSONL row by run.py.
        return {
            "expansions": expansions,
            "evaluator": eval_config.name(),
            "eval_batch": eval_config.batch,
            "eval_focused": eval_config.focused,
            "eval_verified": eval_config.verified,
            "eval_cache": eval_config.cache,
            "eval_top_k": eval_config.top_k,
            "prune": prune,
            "backtrack": backtrack,
            "runtime_sec": round(time.time() - t_start, 3),
            **stats.to_dict(),
            "trace": trace,
        }

    while stack and expansions < max_steps:
        curr = stack.pop()
        if len(curr.filled_clues) >= len(deepest_state.filled_clues):
            deepest_state = curr

        if not curr.unfilled_clues:
            # Record the terminal state so analysis-time replay (+best state)
            # can see the fully-solved grid.
            trace.append({
                "step": expansions + 1,
                "depth": len(curr.filled_clues),
                "filled": [list(item) for item in curr.filled_clues],
                "moves": [],
                "num_moves": 0,
                "num_viable": 0,
                "viable_rate": 0.0,
                "fallback_used": False,
                "terminal": True,
            })
            return curr, True, _build_meta()

        expansions += 1
        moves = await generate_proposals(curr, model, stats)
        record = {
            "step": expansions,
            "depth": len(curr.filled_clues),
            "filled": [list(item) for item in curr.filled_clues],
            "moves": [],
            "num_moves": 0,
            "num_viable": 0,
            "viable_rate": 0.0,
            "fallback_used": False,
        }

        if not moves:
            trace.append(record)
            continue
        if not backtrack:
            moves = moves[:1]

        next_states = [curr.apply_word(cid, w) for cid, w in moves]

        if prune:
            eval_results, fallback_used = await evaluate_siblings(
                next_states, model, eval_config, cache, stats
            )
        else:
            eval_results = [
                {"viable": True, "killed_by": [],
                 "verified": None, "restored_by_verification": False}
                for _ in next_states
            ]
            fallback_used = False

        num_viable = sum(1 for r in eval_results if r["viable"])
        record["num_moves"] = len(moves)
        record["num_viable"] = num_viable
        record["viable_rate"] = num_viable / len(moves) if moves else 0.0
        record["fallback_used"] = fallback_used
        record["moves"] = [
            {
                "cid": cid,
                "word": word,
                "viable": bool(r["viable"]),
                "killed_by": [list(k) for k in r["killed_by"]],
                "verified": r["verified"],
                "restored_by_verification": r["restored_by_verification"],
            }
            for (cid, word), r in zip(moves, eval_results)
        ]
        trace.append(record)

        for ns, r in reversed(list(zip(next_states, eval_results))):
            if r["viable"]:
                stack.append(ns)

    return deepest_state, False, _build_meta()
