"""Prompts for the Creative Writing extension of Tree of Thoughts.

Mirrors the Creative Writing setup from Yao et al. and the implementation guide:
- IO writes a passage directly.
- CoT makes a plan, then writes.
- ToT samples multiple plans/passages and uses a vote prompt to select.
- Evaluation uses an LLM-as-judge scalar coherence score.
"""

STANDARD_PROMPT = """\
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
"""

COT_PROMPT = """\
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Passage:
Your passage here.
"""

SAMPLE_PROMPT = COT_PROMPT

VOTE_PROMPT = """\
Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {{s}}", where s the integer id of the choice.
"""

SCORE_PROMPT = """\
Analyze the following passage, then at the last line conclude "Thus the coherency score is {{s}}", where s is an integer from 1 to 10.
"""
