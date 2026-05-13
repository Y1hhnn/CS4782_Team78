"""Tests for the Mini Crosswords physical engine. Run with: pytest tests/test_crosswords.py"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.crosswords import BoardState, filter_hallucinations


# --------------------------------------------------------------------
# Helper function to create an empty board state for testing
# --------------------------------------------------------------------
def create_empty_state() -> BoardState:
    dummy_clues = (
        ("h1", "clue"), ("h2", "clue"), ("h3", "clue"), ("h4", "clue"), ("h5", "clue"),
        ("v1", "clue"), ("v2", "clue"), ("v3", "clue"), ("v4", "clue"), ("v5", "clue")
    )
    clue_ids = tuple(c[0] for c in dummy_clues)
    empty_grid = tuple(tuple(['_'] * 5) for _ in range(5))
    
    return BoardState(
        grid=empty_grid,
        filled_clues=(),
        unfilled_clues=clue_ids,
        inputs=dummy_clues,
        steps=0
    )


# --------------------------------------------------------------------
# Test 1: Immutability & Constraint Extraction
# --------------------------------------------------------------------
def test_board_state_apply_and_constraints():
    initial_state = create_empty_state()
    
    assert initial_state.get_constraint_for("h1") == "_ _ _ _ _"
    assert initial_state.get_constraint_for("v3") == "_ _ _ _ _"
    
    state_2 = initial_state.apply_word("h1", "APPLE")
    
    assert initial_state.get_constraint_for("v1") == "_ _ _ _ _"
    assert "h1" in initial_state.unfilled_clues
    
    assert ("h1", "APPLE") in state_2.filled_clues
    assert "h1" not in state_2.unfilled_clues
    
    assert state_2.get_constraint_for("v1") == "A _ _ _ _"
    assert state_2.get_constraint_for("v2") == "P _ _ _ _"
    assert state_2.get_constraint_for("v5") == "E _ _ _ _"
    
    state_3 = state_2.apply_word("v2", "PEARL")
    assert state_3.get_constraint_for("h2") == "_ E _ _ _"
    assert state_3.get_constraint_for("h5") == "_ L _ _ _"


# --------------------------------------------------------------------
# Test 2: Exception Handling & Edge Cases
# --------------------------------------------------------------------
def test_apply_word_invalid_length():
    state = create_empty_state()
    with pytest.raises(ValueError, match="must be exactly 5 letters"):
        state.apply_word("h1", "APPLES") 
    with pytest.raises(ValueError, match="must be exactly 5 letters"):
        state.apply_word("v2", "APP")


# --------------------------------------------------------------------
# Test 3: Hallucination Filter
# --------------------------------------------------------------------
def test_filter_hallucinations_happy_path():
    constraint = "A _ _ L E"
    proposals = ["APPLE", "AMPLE", "AGILE"]
    valid = filter_hallucinations(proposals, constraint)
    assert valid == ["APPLE", "AMPLE", "AGILE"]


def test_filter_hallucinations_reject_mismatches():
    constraint = "A _ _ L E"
    # ALONE: pos3=N (≠L); AGREE: pos3=E (≠L); AROSE: pos3=S (≠L)
    proposals = ["APPLE", "ALONE", "AGREE", "AROSE"]
    valid = filter_hallucinations(proposals, constraint)
    assert valid == ["APPLE"]


def test_filter_hallucinations_reject_wrong_lengths():
    constraint = "_ _ _ _ _"  
    proposals = ["CAT", "DOGS", "HOUSE", "ELEPHANT"]
    valid = filter_hallucinations(proposals, constraint)
    assert valid == ["HOUSE"]


def test_filter_hallucinations_handles_formatting_noise():
    constraint = "T _ R _ _"
    # TIGER: pos2=G (≠R); TART: 4 letters, wrong length
    proposals = [" tHreE ", " TIGER", "TART", "TAROT"]
    valid = filter_hallucinations(proposals, constraint)
    assert valid == ["THREE", "TAROT"]