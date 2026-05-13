"""Tests for the verifier. Run with:  pytest tests/

These cover the failure modes that bit the original ToT authors and
will bite anyone re-implementing this. Do NOT skip these.
"""

import sys
from pathlib import Path

# Make the parent dir (with game24.py) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.game24 import verify, parse_propose, State  # noqa: E402


# ----------------------------- verify -----------------------------

def test_basic_correct():
    assert verify("(4 + 8) * (6 - 4) = 24", (4, 4, 6, 8))


def test_bare_expression_no_equals_sign():
    assert verify("(4 + 8) * (6 - 4)", (4, 4, 6, 8))


def test_missing_input_number():
    # Inputs include two 4s; expression uses only one
    assert not verify("4 + 8 + 6 = 18", (4, 4, 6, 8))


def test_extra_input_number_not_in_puzzle():
    # 5 isn't in the inputs (4,4,6,8)
    assert not verify("4 + 5 + 6 + 8 = 23", (4, 4, 6, 8))


def test_wrong_total():
    assert not verify("4 + 5 + 6 + 10 = 25", (4, 5, 6, 10))


def test_extra_literal_smuggled_in():
    # 4 used twice (only once in inputs); 6 and 10 missing
    assert not verify("4 * 5 + 4 = 24", (4, 5, 6, 10))


def test_famous_float_case():
    """`8 / (3 - 8/3)` equals 24 only via float arithmetic. Paper accepts it."""
    assert verify("8 / (3 - 8/3) = 24", (3, 3, 8, 8))


def test_division_by_zero_returns_false():
    assert not verify("(4 - 4) / (6 - 6) = 24", (4, 4, 6, 6))


def test_disallowed_operator():
    assert not verify("4 ** 2 + 8 = 24", (4, 2, 8))


def test_disallowed_function():
    assert not verify("sqrt(36) * 4 + 0 = 24", (6, 4, 0, 0))


# --------------------------- parse_propose --------------------------

def _empty_state(inputs):
    return State(numbers=tuple(map(float, inputs)), steps=(),
                 inputs=tuple(map(float, inputs)))


def test_parse_propose_happy_path():
    text = (
        "4 + 5 = 9 (left: 6 9 10)\n"
        "5 * 6 = 30 (left: 4 10 30)\n"
        "10 - 4 = 6 (left: 5 6 6)\n"
    )
    parent = _empty_state((4, 5, 6, 10))
    states = parse_propose(text, parent)
    assert len(states) == 3
    assert states[0].numbers == (6.0, 9.0, 10.0)
    assert states[0].steps == ("4 + 5 = 9",)


def test_parse_propose_drops_malformed_lines():
    text = (
        "4 + 5 = 9 (left: 6 9 10)\n"
        "this line is garbage\n"
        "5 * 6 = 30 (left: 4 10 30)\n"
    )
    states = parse_propose(text, _empty_state((4, 5, 6, 10)))
    assert len(states) == 2


def test_parse_propose_handles_decimals():
    text = "10 / 4 = 2.5 (left: 2.5 5 6)\n"
    states = parse_propose(text, _empty_state((4, 5, 6, 10)))
    assert len(states) == 1
    assert states[0].numbers == (2.5, 5.0, 6.0)


# --------------------- batched value parser ----------------------

from algorithms.methods_24 import _parse_batch_value  # noqa: E402


def test_parse_batch_value_happy_path():
    text = (
        "State 1: sure\n"
        "State 2: likely\n"
        "State 3: impossible\n"
    )
    scores = _parse_batch_value(text, n_states=3)
    assert scores == [20.0, 1.0, 0.001]


def test_parse_batch_value_with_trailing_explanation():
    text = (
        "State 1: sure (4 + 8 + 12 = 24)\n"
        "State 2: impossible because numbers too small.\n"
    )
    scores = _parse_batch_value(text, n_states=2)
    assert scores == [20.0, 0.001]


def test_parse_batch_value_missing_lines_score_zero():
    text = "State 2: sure\n"
    scores = _parse_batch_value(text, n_states=3)
    # State 1 and State 3 missing -> 0; State 2 -> 20
    assert scores == [0.0, 20.0, 0.0]


def test_parse_batch_value_out_of_range_indices_ignored():
    text = (
        "State 1: sure\n"
        "State 99: likely\n"      # out of range, ignored
        "State 0: impossible\n"   # 0 is out of range (we use 1-indexed)
    )
    scores = _parse_batch_value(text, n_states=2)
    assert scores == [20.0, 0.0]


def test_parse_batch_value_unknown_label_score_zero():
    text = (
        "State 1: sure\n"
        "State 2: confused\n"        # not in LABEL_SCORE
    )
    scores = _parse_batch_value(text, n_states=2)
    assert scores == [20.0, 0.0]
