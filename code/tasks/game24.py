"""Game-of-24 task primitives: state representation, output parsing, verification.

Nothing here makes API calls. This file is pure logic, fully unit-testable
without spending a single token. Build & test this BEFORE wiring up the LLM.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import sympy


# --------------------------------------------------------------------
# State
# --------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    """A node in the search tree.

    `numbers` is the multiset of values still to be combined. `steps` is the
    history of arithmetic operations taken to reach this state. `inputs` is
    the original puzzle (carried along so the value/finish prompts can
    reference it without us having to thread it through every call).
    """
    numbers: tuple    # remaining numbers, e.g. (4.0, 9.0)
    steps: tuple      # operation history, e.g. ("5 * 6 = 30", "30 - 10 = 20")
    inputs: tuple     # original 4 inputs, e.g. (4.0, 5.0, 6.0, 10.0)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def is_terminal(self) -> bool:
        # Game of 24 has 4 inputs -> 3 binary ops -> 1 number left
        return self.depth >= 3


# --------------------------------------------------------------------
# Formatting helper (used by every prompt)
# --------------------------------------------------------------------

def fmt_nums(nums) -> str:
    """Format a tuple of floats as a space-separated string.

    Whole numbers render without a decimal: (4.0, 5.0, 6.0) -> "4 5 6".
    Fractions render compactly: (0.5, 4.0) -> "0.5 4".
    """
    parts = []
    for x in nums:
        if x == int(x):
            parts.append(str(int(x)))
        else:
            parts.append(f"{x:g}")
    return " ".join(parts)


# --------------------------------------------------------------------
# Propose-prompt output parser
# --------------------------------------------------------------------

# Matches one line of "a OP b = c (left: x y z)"
# Numbers can be negative and may have a decimal part.
_PROPOSE_LINE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*"            # a
    r"([+\-*/])\s*"                         # op
    r"(-?\d+(?:\.\d+)?)\s*=\s*"             # b = ...
    r"(-?\d+(?:\.\d+)?)\s*"                 # c
    r"\(left:\s*([-\d.\s]+)\)\s*$"          # (left: ...)
)


def parse_propose(text: str, parent: State) -> list[State]:
    """Parse a propose-prompt completion into a list of new States.

    Lines that don't match the expected format are silently dropped --
    LMs are noisy, and that's fine; the search has many candidates per step.
    """
    new_states = []
    for line in text.strip().split("\n"):
        m = _PROPOSE_LINE.match(line)
        if not m:
            continue
        a, op, b, c, left_str = m.groups()
        try:
            left_nums = tuple(float(x) for x in left_str.split())
        except ValueError:
            continue
        # Preserve original spacing in the step (helpful for the value prompt)
        step = f"{a} {op} {b} = {c}"
        new_states.append(State(
            numbers=left_nums,
            steps=parent.steps + (step,),
            inputs=parent.inputs,
        ))
    return new_states


# --------------------------------------------------------------------
# Verifier  (the most important function in this file)
# --------------------------------------------------------------------

def verify(equation: str, inputs: tuple, tol: float = 1e-6) -> bool:
    """Return True iff `equation` is a valid Game-of-24 solution for `inputs`.

    Built on the paper's reference implementation
    (princeton-nlp/tree-of-thought-llm, src/tot/tasks/game24.py::test_output)
    with two deliberate strengthenings:

      - Like the paper: sympy.simplify for arithmetic (handles `8 / (3 - 8/3)`
        exactly via rationals; no float-tolerance hack), and operand-multiset
        equality check.
      - UNLIKE the paper: rejects `**` (power) and `//` (floor division). The
        Game of 24 rules stated in the propose/IO/CoT prompts only permit
        `+ - * /`, but the paper's verifier accepts any expression sympy can
        evaluate -- so `4 ** 2 + 8 = 24` would slip through. We reject it.
        This is what the paper *intended*; their code happened to be more
        permissive than their stated rules.
      - UNLIKE the paper: strips markdown emphasis (`**bold**`, `__bold__`)
        and backticks BEFORE running the paper's preprocessing. Models
        frequently output `**Answer: 13 + 4 + 4 + 3 = 24**` or
        `**Answer:** 13 + 4 + 4 + 3 = 24`. The underlying equations are
        valid; the markdown is formatting noise. We strip emphasis only
        when the markers come in a balanced pair around non-`*` content,
        so a real `4 ** 2` attempt is preserved and rejected by the
        operator-whitelist check below.

    The `tol` argument is accepted for backward compatibility but ignored;
    sympy comparison is exact.
    """
    # 1) Strip markdown emphasis BEFORE paper preprocessing.
    #    (a) `**Answer:**` / `__Answer:__` -> `Answer:` (handles bolded prefix)
    #    (b) Whole-line wrappers `**...**` / `__...__` / `*...*` / `_..._`
    #        anchored at line start AND end. We deliberately do NOT strip
    #        wrappers that only match on one side, so a real `4 ** 2`
    #        (no closing `**` to its right) survives intact and is rejected
    #        by the operator-whitelist check below.
    cleaned = re.sub(r"\*\*\s*answer\s*:\s*\*\*", "Answer:", equation, flags=re.IGNORECASE)
    cleaned = re.sub(r"__\s*answer\s*:\s*__",     "Answer:", cleaned,  flags=re.IGNORECASE)

    cleaned_lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        # Repeatedly peel off line-wrapping emphasis markers (max 3 nestings)
        for _ in range(3):
            for marker in ("**", "__", "*", "_"):
                if (stripped.startswith(marker) and stripped.endswith(marker)
                        and len(stripped) > 2 * len(marker)):
                    stripped = stripped[len(marker):-len(marker)].strip()
                    break
            else:
                break
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines).replace("`", "")

    # 2) Paper's preprocessing: take last line, lowercase, strip "answer: "
    #    prefix, split on "=" and keep the LHS.
    expression = (
        cleaned.strip().split("\n")[-1]
               .lower()
               .replace("answer: ", "")
               .split("=")[0]
               .strip()
    )

    # 3) Strengthening: reject `**` and `//` (the only Python operators that
    #    sneak in via two characters from `+ - * /`).
    if re.search(r"[*/]{2}", expression):
        return False

    # 4) Operand multiset must match (compared as strings, like the paper does)
    numbers = re.findall(r"\d+", expression)
    problem_numbers = [str(x) for x in inputs]
    if sorted(numbers) != sorted(problem_numbers):
        return False

    try:
        return bool(sympy.simplify(expression) == 24)
    except Exception:
        return False


# --------------------------------------------------------------------
# Puzzle loader
# --------------------------------------------------------------------

def load_puzzles(path: str = "data/24.csv",
                 start: int = 900, end: int = 1000) -> list[tuple[int, ...]]:
    """Load Game-of-24 puzzles from the original repo's CSV.

    Default range is rows 900..999 (0-indexed) which corresponds to the
    paper's "100 hardest" subset (puzzle ranks 901..1000, since 4nums.com
    sorts by human solve time).

    Returns: list of int-tuples, one per puzzle, e.g. [(4, 5, 6, 10), ...].
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    puzzles = []
    for row in rows[start:end]:
        nums = tuple(int(x) for x in row["Puzzles"].split())
        puzzles.append(nums)
    return puzzles
