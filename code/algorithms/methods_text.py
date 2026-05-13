"""The three solver strategies for the Creative Writing task: IO, CoT, and ToT.

Each async function `<method>_solve_text(sentences, model, ...)` returns a
4-paragraph passage string. Unlike Game of 24, this task has no deterministic
verifier; generated passages are evaluated later by an LLM-as-judge coherence
score on a 1–10 scale.

Creative Writing ToT is shallower than Game of 24:
  1. generate k candidate writing plans
  2. vote for the best plan
  3. generate k candidate passages from that plan
  4. vote for the best passage
"""

from __future__ import annotations

import asyncio
import re

from prompts.prompts_text import (
    COT_PROMPT,
    SAMPLE_PROMPT,
    SCORE_PROMPT,
    STANDARD_PROMPT,
    VOTE_PROMPT,
)


_BEST_CHOICE_RE = re.compile(r"best choice is\s*(\d+)", re.IGNORECASE)
_SCORE_RE = re.compile(r"coherency score is\s*(\d+)", re.IGNORECASE)


def _format_sentences(sentences: list[str]) -> str:
    """Format 4 sentence constraints in the same compact style as the paper."""
    return " ".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))


def extract_plan(text: str) -> str:
    """Extract the Plan section from a CoT-style response."""
    if not text:
        return ""
    if "Plan:" in text and "Passage:" in text:
        return text.split("Plan:", 1)[1].split("Passage:", 1)[0].strip()
    if "Plan:" in text:
        return text.split("Plan:", 1)[1].strip()
    return text.strip()


def extract_passage(text: str) -> str:
    """Extract the Passage section from a CoT-style response."""
    if not text:
        return ""
    if "Passage:" in text:
        return text.split("Passage:", 1)[1].strip()
    return text.strip()


# --------------------------------------------------------------------
# IO baseline -- one-shot passage generation
# --------------------------------------------------------------------

async def io_solve_text(sentences: list[str], model) -> str:
    """IO baseline: directly ask the model to write the 4-paragraph passage."""
    prompt = STANDARD_PROMPT.format(input=_format_sentences(sentences))
    out = await model.generate(prompt, temperature=0.7, n=1, max_tokens=1600)
    return out[0].strip() if out else ""


# --------------------------------------------------------------------
# CoT baseline -- make a plan, then write the passage
# --------------------------------------------------------------------

async def cot_solve_text(sentences: list[str], model) -> str:
    """CoT baseline: ask for a plan, then parse and return only the passage."""
    prompt = COT_PROMPT.format(input=_format_sentences(sentences))
    out = await model.generate(prompt, temperature=0.7, n=1, max_tokens=2200)
    text = out[0] if out else ""
    return extract_passage(text)

# --------------------------------------------------------------------
# ToT -- generate/vote on plans, then generate/vote on passages
# --------------------------------------------------------------------

async def vote(instruction: str, candidates: list[str], model, n_votes: int = 5) -> str:
    """Vote among candidates and return the candidate with the most parsed votes."""
    candidates = [c.strip() for c in candidates if c and c.strip()]
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    candidates_text = "\n\n".join(
        f"Choice {i + 1}:\n{candidate}" for i, candidate in enumerate(candidates)
    )
    vote_prompt = f"{VOTE_PROMPT}\n\n{instruction}\n\n{candidates_text}\n"

    results = await asyncio.gather(*[
        model.generate(vote_prompt, temperature=1.0, n=1, max_tokens=1200)
        for _ in range(n_votes)
    ])

    votes = [0] * len(candidates)
    for result in results:
        if not result:
            continue
        text = result[0]
        matches = _BEST_CHOICE_RE.findall(text)
        if not matches:
            continue
        idx = int(matches[-1]) - 1
        if 0 <= idx < len(candidates):
            votes[idx] += 1

    best_idx = max(range(len(candidates)), key=lambda i: votes[i])
    return candidates[best_idx]


async def tot_solve_text(
    sentences: list[str],
    model,
    k_plans: int = 5,
    k_passages: int = 5,
    n_votes: int = 5,
) -> str:
    """ToT for Creative Writing: sample plans, vote, sample passages, vote."""
    formatted = _format_sentences(sentences)
    instruction = STANDARD_PROMPT.format(input=formatted)

    plan_prompt = SAMPLE_PROMPT.format(input=formatted)
    plan_results = await asyncio.gather(*[
        model.generate(plan_prompt, temperature=1.0, n=1, max_tokens=1400)
        for _ in range(k_plans)
    ])
    plans = [extract_plan(result[0]) if result else "" for result in plan_results]
    best_plan = await vote(instruction, plans, model, n_votes=n_votes)

    write_prompt = f"{instruction}\n\nPlan:\n{best_plan}\n\nPassage:\n"
    passage_results = await asyncio.gather(*[
        model.generate(write_prompt, temperature=1.0, n=1, max_tokens=1800)
        for _ in range(k_passages)
    ])
    passages = [extract_passage(result[0]) if result else "" for result in passage_results]
    best_passage = await vote(instruction, passages, model, n_votes=n_votes)
    return best_passage


# --------------------------------------------------------------------
# LLM-as-judge coherence scorer
# --------------------------------------------------------------------

async def score_passage(passage: str, judge_model, n_samples: int = 3) -> float:
    """LLM-as-judge coherence score: average n parsed scores from 1 to 10."""
    if not passage.strip():
        return 0.0

    prompt = f"{SCORE_PROMPT}\n\n{passage}\n"
    results = await asyncio.gather(*[
        judge_model.generate(prompt, temperature=1.0, n=1, max_tokens=800)
        for _ in range(n_samples)
    ])

    scores = []
    for result in results:
        if not result:
            continue
        matches = _SCORE_RE.findall(result[0])
        if not matches:
            continue
        score = int(matches[-1])
        if 1 <= score <= 10:
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0

# --------------------------------------------------------------------
# Extension baselines for Creative Writing
# --------------------------------------------------------------------

async def score_candidates_independently(
    candidates: list[str],
    judge_model,
    n_samples: int = 3,
) -> tuple[str, list[float]]:
    """Score each candidate independently and return the highest-scored one.

    This is different from vote(), which compares all candidates in one prompt.
    Here, each candidate gets its own LLM-as-judge coherence score.
    """
    if not candidates:
        return "", []

    scores = []
    for candidate in candidates:
        score = await score_passage(candidate, judge_model, n_samples=n_samples)
        scores.append(score)

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], scores


# --------------------------------------------------------------------
# Best-of-k IO -- sample k direct passages, independently score, pick best
# --------------------------------------------------------------------

async def best_of_k_io_solve_text(
    sentences: list[str],
    model,
    judge_model,
    k: int = 5,
    n_score_samples: int = 3,
) -> tuple[str, list[float]]:
    """Generate k direct passages with IO prompting, score each, pick best."""
    instruction = STANDARD_PROMPT.format(input=" ".join(sentences))

    results = await asyncio.gather(*[
        model.generate(instruction, temperature=1.0, n=1, max_tokens=1600)
        for _ in range(k)
    ])

    candidates = [r[0].strip() if r else "" for r in results]
    best_passage, scores = await score_candidates_independently(
        candidates,
        judge_model,
        n_samples=n_score_samples,
    )

    return best_passage, scores


# --------------------------------------------------------------------
# Best-of-k CoT -- sample k plan+passage outputs, score passages, pick best
# --------------------------------------------------------------------

async def best_of_k_cot_solve_text(
    sentences: list[str],
    model,
    judge_model,
    k: int = 5,
    n_score_samples: int = 3,
) -> tuple[str, list[float]]:
    """Generate k CoT plan+passage outputs, score each passage, pick best."""
    prompt = COT_PROMPT.format(input=" ".join(sentences))

    results = await asyncio.gather(*[
        model.generate(prompt, temperature=1.0, n=1, max_tokens=2000)
        for _ in range(k)
    ])

    candidates = []
    for r in results:
        text = r[0] if r else ""
        candidates.append(extract_passage(text))

    best_passage, scores = await score_candidates_independently(
        candidates,
        judge_model,
        n_samples=n_score_samples,
    )

    return best_passage, scores


# --------------------------------------------------------------------
# Independent-score ToT -- generate plans/passages, score instead of vote
# --------------------------------------------------------------------

async def tot_score_select_solve_text(
    sentences: list[str],
    model,
    judge_model,
    k_plans: int = 5,
    k_passages: int = 5,
    n_score_samples: int = 3,
) -> tuple[str, dict]:
    """ToT variant that replaces vote prompts with independent scoring.

    Normal ToT:
      generate 5 plans -> vote best plan -> generate 5 passages -> vote best passage

    This variant:
      generate 5 plans -> score plan-derived passages indirectly? 
      choose best plan by asking the model to write one passage per plan and scoring it
      then generate 5 passages from that plan
      independently score the 5 passages and pick highest

    This tests whether comparing candidates in one vote prompt is better/worse
    than scoring candidates independently.
    """
    instruction = STANDARD_PROMPT.format(input=" ".join(sentences))
    plan_prompt = COT_PROMPT.format(input=" ".join(sentences))

    plan_results = await asyncio.gather(*[
        model.generate(plan_prompt, temperature=1.0, n=1, max_tokens=1200)
        for _ in range(k_plans)
    ])

    plans = [extract_plan(r[0]) if r else "" for r in plan_results]

    plan_passages = []
    for plan in plans:
        write_prompt = f"{instruction}\n\nPlan:\n{plan}\n\nPassage:\n"
        out = await model.generate(write_prompt, temperature=0.7, n=1, max_tokens=1600)
        plan_passages.append(out[0].strip() if out else "")

    best_plan_passage, plan_scores = await score_candidates_independently(
        plan_passages,
        judge_model,
        n_samples=n_score_samples,
    )

    best_plan_idx = max(range(len(plan_scores)), key=lambda i: plan_scores[i])
    best_plan = plans[best_plan_idx]

    write_prompt = f"{instruction}\n\nPlan:\n{best_plan}\n\nPassage:\n"

    passage_results = await asyncio.gather(*[
        model.generate(write_prompt, temperature=1.0, n=1, max_tokens=1600)
        for _ in range(k_passages)
    ])

    passages = [r[0].strip() if r else "" for r in passage_results]

    best_passage, passage_scores = await score_candidates_independently(
        passages,
        judge_model,
        n_samples=n_score_samples,
    )

    info = {
        "plans": plans,
        "plan_scores": plan_scores,
        "selected_plan_index": best_plan_idx,
        "passage_scores": passage_scores,
    }

    return best_passage, info


# --------------------------------------------------------------------
# Plan-only CoT -- generate only one plan, then write from that plan
# --------------------------------------------------------------------

async def plan_only_cot_solve_text(
    sentences: list[str],
    model,
) -> tuple[str, str]:
    """CoT-style method that explicitly separates plan generation and writing.

    This differs from cot_solve_text because it makes two calls:
      1. generate one plan
      2. generate one passage from that plan

    This lets us compare one-plan CoT against ToT's five-plan vote process.
    """
    instruction = STANDARD_PROMPT.format(input=" ".join(sentences))
    plan_prompt = COT_PROMPT.format(input=" ".join(sentences))

    plan_out = await model.generate(plan_prompt, temperature=0.7, n=1, max_tokens=1200)
    plan_text = plan_out[0] if plan_out else ""
    plan = extract_plan(plan_text)

    write_prompt = f"{instruction}\n\nPlan:\n{plan}\n\nPassage:\n"
    passage_out = await model.generate(write_prompt, temperature=0.7, n=1, max_tokens=1600)
    passage = passage_out[0].strip() if passage_out else ""

    return passage, plan


# --------------------------------------------------------------------
# Plan-vote ToT -- vote on plans only, then write one passage
# --------------------------------------------------------------------

async def plan_vote_solve_text(
    sentences: list[str],
    model,
    k_plans: int = 5,
    n_votes: int = 5,
) -> tuple[str, dict]:
    """Generate k plans, vote for best plan, then write one passage.

    This isolates the effect of plan voting alone.

    Full ToT:
      vote plans + vote passages

    Plan-vote only:
      vote plans + write one passage
    """
    instruction = STANDARD_PROMPT.format(input=" ".join(sentences))
    plan_prompt = COT_PROMPT.format(input=" ".join(sentences))

    plan_results = await asyncio.gather(*[
        model.generate(plan_prompt, temperature=1.0, n=1, max_tokens=1200)
        for _ in range(k_plans)
    ])

    plans = [extract_plan(r[0]) if r else "" for r in plan_results]
    best_plan = await vote(instruction, plans, model, n_votes=n_votes)
    selected_plan_index = plans.index(best_plan) if best_plan in plans else -1

    write_prompt = f"{instruction}\n\nPlan:\n{best_plan}\n\nPassage:\n"
    passage_out = await model.generate(write_prompt, temperature=0.7, n=1, max_tokens=1600)
    passage = passage_out[0].strip() if passage_out else ""

    info = {
        "plans": plans,
        "selected_plan_index": selected_plan_index,
    }

    return passage, info
