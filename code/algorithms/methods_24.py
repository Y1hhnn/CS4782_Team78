"""The three solver strategies: IO, CoT, ToT.

Each is an async function `<method>_solve(inputs, model, ...)` that returns
a final equation string (or "" on failure). Pass that string to
`game24.verify(equation, inputs)` to check correctness.

Cost optimization: the value evaluator uses BATCHED prompts -- one API call
scores all candidates at a BFS depth, instead of one call per candidate.
This drops ToT(b=5) from ~170 to ~20 calls per puzzle. See README.
"""

from __future__ import annotations

import asyncio
import random
import re

from tasks.game24 import State, fmt_nums, parse_propose
from prompts.prompts_24 import (
    BATCH_VALUE_LAST_PROMPT,
    BATCH_VALUE_PROMPT,
    COT_FINISH_PROMPT,
    COT_PROMPT,
    IO_PROMPT,
    PROPOSE_PROMPT,
    VALUE_LAST_PROMPT,
    VALUE_PROMPT,
)


# --------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------

# Find the line "Answer: <equation>"; if not present, fall back to the
# last line that contains an `=` sign (CoT models sometimes drop the prefix).
_ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+)$")


def _extract_answer(text: str) -> str:
    for line in text.strip().split("\n"):
        m = _ANSWER_LINE.match(line)
        if m:
            return m.group(1).strip()
    for line in reversed(text.strip().split("\n")):
        if "=" in line:
            return line.strip()
    return text.strip()


# --------------------------------------------------------------------
# IO baseline -- one shot, no reasoning
# --------------------------------------------------------------------

async def io_solve(inputs: tuple, model) -> str:
    prompt = IO_PROMPT.format(input=fmt_nums(inputs))
    out = (await model.generate(prompt, temperature=0.7, n=1))[0]
    return _extract_answer(out)


# --------------------------------------------------------------------
# CoT baseline -- think step by step
# --------------------------------------------------------------------

async def cot_solve(inputs: tuple, model) -> str:
    prompt = COT_PROMPT.format(input=fmt_nums(inputs))
    out = (await model.generate(prompt, temperature=0.7, n=1))[0]
    return _extract_answer(out)


# --------------------------------------------------------------------
# ToT  --  the actual contribution of the paper
# --------------------------------------------------------------------

# Maps the value-prompt's verdicts to numeric scores.
LABEL_SCORE = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}

# Parse a single line of batched value output: "State 3: sure" or "State 3: sure."
_BATCH_LINE = re.compile(r"^\s*State\s+(\d+)\s*:\s*(\w+)", re.IGNORECASE)


async def _propose(state: State, model) -> list[State]:
    """Generate next-state candidates from `state`.

    Depth 0 and 1 use the `propose` prompt (one binary op at a time).
    Depth 2 uses the `cot_finish` prompt to produce a full equation in
    one shot, since at this point only two numbers remain.
    """
    if state.depth < 2:
        prompt = PROPOSE_PROMPT.format(input=fmt_nums(state.numbers))
        out = (await model.generate(prompt, temperature=0.7, n=1))[0]
        return parse_propose(out, state)

    # depth == 2: ask for the final equation
    prompt = COT_FINISH_PROMPT.format(
        input=fmt_nums(state.inputs),
        steps="\n".join(state.steps),
    )
    out = (await model.generate(prompt, temperature=0.7, n=1))[0]
    answer = _extract_answer(out)
    return [State(
        numbers=(24.0,),                       # marker: terminal
        steps=state.steps + (answer,),         # last step IS the equation
        inputs=state.inputs,
    )]


def _format_states_for_value(states: list[State]) -> str:
    """Format intermediate states as 'State N: <numbers>' lines."""
    return "\n".join(
        f"State {i+1}: {fmt_nums(s.numbers)}"
        for i, s in enumerate(states)
    )


def _format_states_for_value_last(states: list[State]) -> str:
    """Format terminal states as 'State N:\\n  Input: ...\\n  Answer: ...' blocks."""
    blocks = []
    for i, s in enumerate(states):
        blocks.append(
            f"State {i+1}:\n"
            f"  Input: {fmt_nums(s.inputs)}\n"
            f"  Answer: {s.steps[-1] if s.steps else ''}"
        )
    return "\n".join(blocks)


def _parse_batch_value(text: str, n_states: int) -> list[float]:
    """Parse 'State N: <verdict>' lines into a per-state score list of length n_states.

    Missing or unparseable verdicts get score 0 (treated as "no signal").
    """
    scores = [0.0] * n_states
    for line in text.strip().split("\n"):
        m = _BATCH_LINE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1                    # 1-indexed in prompt
        if not (0 <= idx < n_states):
            continue
        label = m.group(2).lower().rstrip(".")
        scores[idx] = LABEL_SCORE.get(label, 0.0)
    return scores


async def _evaluate_batch(states: list[State], model,
                          n_votes: int = 3) -> list[float]:
    """Score every state in `states` with one batched value prompt per vote.

    Returns a list of summed scores aligned with `states`. We shuffle the
    state order independently for each vote to reduce position bias.
    """
    if not states:
        return []

    is_terminal = states[0].is_terminal
    template = BATCH_VALUE_LAST_PROMPT if is_terminal else BATCH_VALUE_PROMPT
    formatter = (_format_states_for_value_last if is_terminal
                 else _format_states_for_value)

    # Build a separate prompt per vote, each with a different random order
    prompts = []
    perms = []
    for _ in range(n_votes):
        perm = list(range(len(states)))
        random.shuffle(perm)
        perms.append(perm)
        shuffled = [states[i] for i in perm]
        prompts.append(template.format(states=formatter(shuffled)))

    # Fire all votes in parallel (rate limiter handles RPM cap)
    outputs = await asyncio.gather(
        *[model.generate(p, temperature=1.0, n=1) for p in prompts]
    )

    # Sum scores per original state index across all votes
    total = [0.0] * len(states)
    for perm, out_list in zip(perms, outputs):
        out = out_list[0] if out_list else ""
        if not out.strip():
            continue
        per_shuffled = _parse_batch_value(out, len(states))
        # Map from shuffled position back to original
        for shuffled_idx, score in enumerate(per_shuffled):
            original_idx = perm[shuffled_idx]
            total[original_idx] += score
    return total


# --------------------------------------------------------------------
# Un-batched evaluator (paper's original, one call per (state, vote))
# --------------------------------------------------------------------

async def _value_one(state: State, model, n_votes: int = 3) -> float:
    """Score a single state via `n_votes` independent calls (paper's method)."""
    if state.is_terminal:
        prompt = VALUE_LAST_PROMPT.format(
            input=fmt_nums(state.inputs),
            answer=state.steps[-1] if state.steps else "",
        )
    else:
        prompt = VALUE_PROMPT.format(input=fmt_nums(state.numbers))

    outputs = await model.generate(prompt, temperature=1.0, n=n_votes)
    score = 0.0
    for out in outputs:
        if not out.strip():
            continue
        last_token = out.strip().split()[-1].lower().rstrip(".")
        score += LABEL_SCORE.get(last_token, 0.0)
    return score


async def _evaluate_unbatched(states: list[State], model,
                              n_votes: int = 3) -> list[float]:
    """Score every state with one API call per (state, vote) pair.

    This matches the paper's reference implementation exactly. Costs
    ~n_states * n_votes calls per BFS depth, vs n_votes for the batched
    version. Use only as an ablation control -- on free-tier rate limits
    a single ToT(b=5) puzzle takes ~170 calls instead of ~20.
    """
    return list(await asyncio.gather(
        *[_value_one(s, model, n_votes=n_votes) for s in states]
    ))


async def _evaluate_chunked(states: list[State], model,
                            batch_size: int, n_votes: int = 3) -> list[float]:
    """Score states in chunks of `batch_size`, using the batched evaluator on
    each chunk independently.

    This interpolates between fully-batched (batch_size=len(states)) and
    fully-un-batched (batch_size=1). Used for the batch-size sweep ablation:
    does accuracy degrade gradually or cliff-like as batch_size grows?
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size == 1:
        return await _evaluate_unbatched(states, model, n_votes=n_votes)

    chunks = [states[i:i + batch_size]
              for i in range(0, len(states), batch_size)]
    all_scores: list[float] = []
    for chunk in chunks:
        chunk_scores = await _evaluate_batch(chunk, model, n_votes=n_votes)
        all_scores.extend(chunk_scores)
    return all_scores


async def tot_solve(inputs: tuple, model,
                    b: int = 5, n_votes: int = 3,
                    batch: bool = True,
                    batch_size: int | None = None) -> tuple[str, list]:
    """Tree-of-Thoughts BFS for Game of 24.

    Args:
      inputs:   the 4 input numbers, e.g. (4, 5, 6, 10).
      model:    a GeminiWrapper.
      b:        beam width -- keep top `b` states after each expansion.
      n_votes:  number of value-prompt samples per state (or per BFS depth
                in batched mode).
      batch:    if True (default), score all candidates at one BFS depth in
                a single API call per vote. If False, use the paper's
                reference one-call-per-state method (~9x more API calls).
      batch_size: if set (integer > 0), overrides `batch` and evaluates
                candidates in chunks of this size. 1 = un-batched (paper's
                method). None = use full-batch or un-batched per `batch` flag.

    Returns:
      (final_equation, trace). `trace[d]` is the top-`b` (score, last_step)
      pairs at depth `d+1`, useful for debugging and ablation.
    """
    initial = State(
        numbers=tuple(map(float, inputs)),
        steps=(),
        inputs=tuple(map(float, inputs)),
    )
    frontier = [initial]
    trace = []

    if batch_size is not None:
        async def evaluate(states, model, n_votes=3):
            return await _evaluate_chunked(states, model, batch_size, n_votes)
    elif batch:
        evaluate = _evaluate_batch
    else:
        evaluate = _evaluate_unbatched

    for _ in range(3):  # Game of 24 has depth 3
        # 1) Expand: every frontier state proposes successors (parallel)
        expansions = await asyncio.gather(*[_propose(s, model) for s in frontier])
        candidates = [c for sub in expansions for c in sub]

        if not candidates:
            return "", trace

        # 2) Evaluate (batched OR un-batched depending on `batch` flag)
        scores = await evaluate(candidates, model, n_votes=n_votes)

        # 3) Prune: keep top-b
        scored = sorted(zip(scores, candidates), key=lambda x: -x[0])
        frontier = [c for _, c in scored[:b]]
        trace.append([(s, c.steps[-1] if c.steps else "") for s, c in scored[:b]])

    final_eq = frontier[0].steps[-1] if frontier and frontier[0].steps else ""
    return final_eq, trace
