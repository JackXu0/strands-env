# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward function for math problems using math-verify for symbolic equivalence.

Uses HuggingFace's math-verify library to parse LaTeX/expressions from both
the model's ``\\boxed{}`` output and the ground truth, then checks equivalence
via SymPy (handling fractions, sets, nested expressions, etc.).
"""

from __future__ import annotations

import logging

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)

# Use both LaTeX and expression extraction (math-verify's recommended default).
# LatexExtractionConfig handles \boxed{}, $...$, etc.
# ExprExtractionConfig handles plain "4", "0.5", "-3", etc.
_EXTRACTION_CONFIG = (LatexExtractionConfig(), ExprExtractionConfig())


class MathRewardFunction(RewardFunction):
    """Reward 1.0 if the model's ``\\boxed{}`` answer is mathematically equivalent to ground truth.

    Uses ``math_verify.parse`` to extract and convert answers to SymPy,
    then ``math_verify.verify`` for equivalence checking. This handles:

    - Fraction / decimal equivalence (``1/2`` = ``0.5``)
    - Symbolic simplification (``\\sqrt{3}/3`` = ``1/\\sqrt{3}``)
    - Sets and intervals (``{1,3} \\cup {2,4}`` = ``{1,2,3,4}``)
    - Nested expressions and LaTeX normalization

    When either side parses to multiple candidate expressions (e.g. several
    ``\\boxed{}`` in the response), ``verify`` returns True if **any**
    gold-target pair matches (Cartesian product).

    Args:
        float_rounding: Decimal places for float comparison (default 6).
        timeout_seconds: Max seconds for SymPy simplification per comparison (default 5).
    """

    def __init__(self, float_rounding: int = 6, timeout_seconds: int = 5) -> None:
        self.float_rounding = float_rounding
        self.timeout_seconds = timeout_seconds

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        ground_truth = action.task_context.ground_truth
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return RewardResult(reward=0.0, info={"reason": "invalid_ground_truth"})

        content = step_result.observation.final_response
        if content is None:
            return RewardResult(reward=0.0, info={"reason": "no_final_response"})

        gold = parse(ground_truth, extraction_config=_EXTRACTION_CONFIG)
        if not gold:
            return RewardResult(reward=0.0, info={"reason": "gold_parse_failed", "ground_truth": ground_truth})

        answer = parse(content, extraction_config=_EXTRACTION_CONFIG)
        if not answer:
            return RewardResult(reward=0.0, info={"reason": "answer_parse_failed", "response": content})

        try:
            matched = verify(gold, answer, float_rounding=self.float_rounding, timeout_seconds=self.timeout_seconds)
        except Exception as e:
            logger.error(f"math-verify failed: {e} (gold={gold}, answer={answer})")
            matched = False

        return RewardResult(
            reward=1.0 if matched else 0.0,
            info={
                "matched": matched,
                "gold_parsed": [str(g) for g in gold],
                "answer_parsed": [str(a) for a in answer],
                "ground_truth": ground_truth,
            },
        )
