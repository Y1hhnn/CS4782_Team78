"""Mini Crosswords task primitives: state representation, constraint mapping, and data loading.

This file is pure logic. It maintains the 5x5 grid, extracts letter constraints
for intersecting words, and aggressively filters out LLM hallucinations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# --------------------------------------------------------------------
# State Representation
# --------------------------------------------------------------------

@dataclass(frozen=True)
class BoardState:
    """A node in the DFS search tree for Mini Crosswords.
    
    Uses frozen=True (immutable) so that backtracking in DFS is trivially safe.
    Whenever a word is applied, it returns a strictly new BoardState.
    """
    grid: tuple[tuple[str, ...], ...]
    filled_clues: tuple[Tuple[str, str], ...]
    unfilled_clues: tuple[str, ...]
    inputs: tuple[Tuple[str, str], ...]
    ground_truth: tuple[str, ...] = ()  
    steps: int = 0
    
    def get_constraint_for(self, clue_id: str) -> str:
        """Extract the letter constraint for a given clue based on the current grid.
        
        Example: If h1 is filled with 'APPLE', then get_constraint_for('v1') 
        will return 'A _ _ _ _'.
        """
        direction = clue_id[0]
        idx = int(clue_id[1]) - 1  # 'h1' -> 0, 'v3' -> 2
        
        if direction == 'h':
            chars = self.grid[idx]
        elif direction == 'v':
            chars = [self.grid[r][idx] for r in range(5)]
        else:
            raise ValueError(f"Invalid clue_id: {clue_id}")
            
        return " ".join(chars)

    def apply_word(self, clue_id: str, word: str) -> BoardState:
        """Return a NEW state with the word applied to the grid. (Immutable update)"""
        word = word.strip().upper()
        if len(word) != 5:
            raise ValueError(f"Word '{word}' must be exactly 5 letters.")

        # Convert frozen tuple grid to mutable list of lists
        new_grid = [list(row) for row in self.grid]
        direction = clue_id[0]
        idx = int(clue_id[1]) - 1

        if direction == 'h':
            for c in range(5):
                new_grid[idx][c] = word[c]
        else:  # 'v'
            for r in range(5):
                new_grid[r][idx] = word[r]

        # Update dictionaries and lists
        new_filled = dict(self.filled_clues)
        new_filled[clue_id] = word
        
        new_unfilled = list(self.unfilled_clues)
        if clue_id in new_unfilled:
            new_unfilled.remove(clue_id)

        # Freeze back into tuples
        return BoardState(
            grid=tuple(tuple(row) for row in new_grid),
            filled_clues=tuple(sorted(new_filled.items())),
            unfilled_clues=tuple(new_unfilled),
            inputs=self.inputs,
            ground_truth=self.ground_truth,
            steps=self.steps  + 1
        )

    def apply_full_grid(self, new_grid: tuple[tuple[str, ...], ...]) -> 'BoardState':
        """
        Creates a new BoardState by forcing a complete 5x5 grid.
        Used strictly by IO and CoT baselines to parse the final answer.
        """
        # Assume all inputs are now technically "filled"
        return BoardState(
            grid=new_grid,
            filled_clues=(), 
            unfilled_clues=(),
            inputs=self.inputs,
            ground_truth=self.ground_truth,
            steps=self.steps + 1
        )

    def get_clue_text(self, clue_id: str) -> str:
        """Helper to get the human-readable clue text."""
        for cid, text in self.inputs:
            if cid == clue_id:
                return text
        return ""

    def render(self) -> str:
        """Render the 5x5 grid as a string for debugging or output."""
        return "\n".join(" ".join(row) for row in self.grid)


# --------------------------------------------------------------------
# Heuristics & Filters
# --------------------------------------------------------------------

def select_most_constrained_clue(state: BoardState) -> str:
    """Heuristic: Pick the unfilled clue that has the fewest blank spaces left.
    
    This drastically reduces the branching factor in DFS.
    """
    if not state.unfilled_clues:
        raise ValueError("No unfilled clues left.")
        
    best_clue = None
    max_filled = -1

    for cid in state.unfilled_clues:
        constraint = state.get_constraint_for(cid)
        # Count how many actual letters exist (not '_')
        filled_count = 5 - constraint.count('_')
        if filled_count > max_filled:
            max_filled = filled_count
            best_clue = cid

    return best_clue


def filter_hallucinations(proposals: List[str], constraint: str) -> List[str]:
    """The Iron Gate: Filters out LLM-proposed words that violate the grid.
    
    Converts constraint "A _ _ L E" to regex "^A..LE$" and tests each word.
    Any word with length != 5 or matching failure is silently dropped.
    """
    valid_words = []
    # Build regex: "A _ _ L E" -> "^A..LE$"
    pattern_str = "^" + constraint.replace(" ", "").replace("_", ".") + "$"
    matcher = re.compile(pattern_str, re.IGNORECASE)

    for word in proposals:
        # LLM might output e.g. "APPLE (certain)" depending on parsing, 
        # assume upstream parser cleans it to just the word, but we strip anyway.
        word = word.strip().upper()
        
        if len(word) != 5:
            continue
        if not matcher.match(word):
            continue
            
        valid_words.append(word)
        
    # Deduplicate while preserving order (Python 3.7+ dict maintains insertion order)
    return list(dict.fromkeys(valid_words))


# --------------------------------------------------------------------
# Data Loader
# --------------------------------------------------------------------

def load_crosswords(path: str = "data/mini0505.json",
                    start: int = 0, end: int = 100) -> List[BoardState]:
    """Load Mini Crossword puzzles from the JSON dataset.
    
    The dataset format expects: [ [[10 clues], [25 solution letters]], ... ]
    Clues are strictly ordered: h1, h2, h3, h4, h5, v1, v2, v3, v4, v5.
    
    Returns a list of initial BoardStates.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_states = []
    for item in data[start:end]:
        clue_texts = item[0]  # length 10
        solution_letters = tuple(item[1])  # length 25
        # Build identifiers
        clue_ids = [f"h{i}" for i in range(1, 6)] + [f"v{i}" for i in range(1, 6)]
        inputs_tuple = tuple(zip(clue_ids, clue_texts))

        # Blank 5x5 grid
        empty_grid = tuple(tuple(['_'] * 5) for _ in range(5))

        state = BoardState(
            grid=empty_grid,
            filled_clues=(),
            unfilled_clues=tuple(clue_ids),
            inputs=inputs_tuple,
            ground_truth=solution_letters,
        )
        initial_states.append(state)

    return initial_states

# --------------------------------------------------------------------
# Helper to check correctness (if ground truth is provided)
# --------------------------------------------------------------------
def check_correctness(state: BoardState) -> bool:
    """Check if the final board state perfectly matches the ground truth."""
    if not state.ground_truth:
        return False
    flat_grid = [char for row in state.grid for char in row]
    # Cast tuple to list to ensure equality check works properly
    return flat_grid == list(state.ground_truth)