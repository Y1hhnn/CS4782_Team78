"""Creative Writing task primitives: input loading.

Each input is a dict with an integer `id` and a list of 4 `sentences` that must
appear as the closing sentence of paragraphs 1..4. There is no deterministic
verifier; passages are scored by an LLM judge at run time.
"""

from __future__ import annotations

import json


def load_text_inputs(path: str = "data/text_inputs.json",
                     start: int = 0, end: int | None = None) -> list[dict]:
    """Load Creative Writing inputs from the JSON dataset.

    Each item: {"id": int, "sentences": [str, str, str, str]}.
    """
    with open(path) as f:
        rows = json.load(f)
    if end is None:
        end = len(rows)
    return rows[start:end]
