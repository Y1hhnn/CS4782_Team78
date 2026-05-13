"""All prompts used by IO, CoT, and ToT for Game of 24.

Mirrors:
    github.com/princeton-nlp/tree-of-thought-llm/src/tot/prompts/game24.py

Six prompts in total + two batched variants for cost optimization:
  - IO_PROMPT          : one-shot baseline (5 fewshot exemplars)
  - COT_PROMPT         : think-step-by-step baseline (5 exemplars)
  - PROPOSE_PROMPT     : ToT generator -- "what are the next arithmetic moves?"
  - COT_FINISH_PROMPT  : ToT generator (last step) -- "write the final equation"
  - VALUE_PROMPT       : ToT evaluator -- judge an intermediate state (sure/likely/impossible)
  - VALUE_LAST_PROMPT  : ToT evaluator -- judge a final equation (sure/impossible)
  - BATCH_VALUE_PROMPT      : evaluate N intermediate states in one call (cost optimization)
  - BATCH_VALUE_LAST_PROMPT : judge N final equations in one call (cost optimization)

The two BATCH_* prompts are NOT in the original paper -- they're an engineering
optimization to fit ToT into 15 RPM / 1000 RPD free-tier limits. They keep the
paper's exemplars verbatim and only change the output format. Document this as
a methodology deviation.

DO NOT EDIT these casually. Prompt text is a hyperparameter; small wording
changes can swing accuracy by 10+ percentage points.
"""

# --------------------------------------------------------------------
# Baseline: IO  (one-shot, no reasoning)
# --------------------------------------------------------------------
IO_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each input number must be used exactly once.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
Answer:"""


# --------------------------------------------------------------------
# Baseline: CoT  (intermediate steps then final answer)
# --------------------------------------------------------------------
COT_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
"""


# --------------------------------------------------------------------
# ToT generator: propose next arithmetic step
# --------------------------------------------------------------------
PROPOSE_PROMPT = """\
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
"""


# --------------------------------------------------------------------
# ToT generator (used at last expansion): write final equation
# --------------------------------------------------------------------
COT_FINISH_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Given an input and the steps so far, write a single line "Answer: <expression> = 24" where the expression uses each input number exactly once.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: {input}
Steps:
{steps}
Answer:"""


# --------------------------------------------------------------------
# ToT evaluator: judge an intermediate state
# --------------------------------------------------------------------
VALUE_PROMPT = """\
Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 18
4 * 10 - 4 = 36
(10 - 4) * 4 = 24
sure
4 9 11
9 + 11 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 20
(8 - 5) * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
"""


# --------------------------------------------------------------------
# ToT evaluator: judge a final equation
# --------------------------------------------------------------------
VALUE_LAST_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible
Input: {input}
Answer: {answer}
Judge:"""


# --------------------------------------------------------------------
# ToT evaluator (BATCHED): judge multiple intermediate states in one call
# --------------------------------------------------------------------
# Same exemplars as VALUE_PROMPT, but we then ask for N verdicts at once.
# Slot {states} gets filled with lines like:
#     State 1: 4 9 11
#     State 2: 5 7 8

BATCH_VALUE_PROMPT = """\
Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 18
4 * 10 - 4 = 36
(10 - 4) * 4 = 24
sure
4 9 11
9 + 11 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 20
(8 - 5) * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

Now evaluate each of the following states. Output exactly one line per state in the format:
State <N>: <sure|likely|impossible>

{states}

Output:
"""


# --------------------------------------------------------------------
# ToT evaluator (BATCHED): judge multiple final equations in one call
# --------------------------------------------------------------------
# Same exemplars as VALUE_LAST_PROMPT.
# Slot {states} gets filled with blocks like:
#     State 1:
#       Input: 4 4 6 8
#       Answer: (4 + 8) * (6 - 4) = 24

BATCH_VALUE_LAST_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible

Now judge each of the following (Input, Answer) pairs. Output exactly one line per state in the format:
State <N>: <sure|impossible>

{states}

Output:
"""
