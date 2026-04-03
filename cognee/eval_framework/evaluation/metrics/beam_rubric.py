"""BEAM paper rubric metric with 3-level scoring (0.0, 0.5, 1.0).

Implements the official BEAM evaluation methodology where each rubric criterion
is scored as 0.0 (no compliance), 0.5 (partial compliance), or 1.0 (complete compliance).
Final score = sum of per-item scores / number of rubric items.

Reference: https://github.com/mohammadtavakoli78/BEAM/blob/main/src/evaluation/compute_metrics.py
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _get_llm_client():
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
        get_llm_client,
    )

    return get_llm_client()


class BEAMJudgeVerdict(BaseModel):
    score: float
    reason: str


_BEAM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): {criterion}
- RESPONSE TO EVALUATE: {response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).
2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
3. **Check Compliance**: For positive requirements look for the presence and quality of the required element. For negative constraints look for the absence of the prohibited element.
4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.
5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation as JSON with two fields:

{{"score": <your score: 1.0, 0.5, or 0.0>, "reason": "<detailed explanation>"}}

NOTE: ONLY output the JSON object, without any explanation before or after."""


def _parse_verdict(raw: str) -> BEAMJudgeVerdict:
    """Parse LLM response into a BEAMJudgeVerdict, with fallback regex extraction."""
    text = raw.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
        score = float(data["score"])
        # Clamp to valid values
        if score >= 0.75:
            score = 1.0
        elif score >= 0.25:
            score = 0.5
        else:
            score = 0.0
        return BEAMJudgeVerdict(score=score, reason=data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: extract score from text
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if score_match:
        score = float(score_match.group(1))
        if score >= 0.75:
            score = 1.0
        elif score >= 0.25:
            score = 0.5
        else:
            score = 0.0
        return BEAMJudgeVerdict(score=score, reason=text)

    return BEAMJudgeVerdict(score=0.0, reason=f"Failed to parse judge response: {text[:200]}")


class BEAMRubricMetric:
    """BEAM paper rubric metric with 3-level scoring.

    Each rubric criterion is scored 0.0, 0.5, or 1.0 by an LLM judge.
    Final score = sum of per-item scores / number of items.
    """

    def __init__(self):
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self._verdicts: List[Dict[str, Any]] = []

    def measure(self, test_case) -> float:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, self.a_measure(test_case)).result()
        else:
            result = asyncio.run(self.a_measure(test_case))

        return result

    async def a_measure(self, test_case) -> float:
        response = test_case.actual_output
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        rubric: List[str] = metadata.get("rubric", [])

        if not rubric:
            self.score = 0.0
            self.reason = "No rubric items provided"
            self._verdicts = []
            return self.score

        llm_client = _get_llm_client()

        verdicts = []
        total_score = 0.0

        for criterion in rubric:
            prompt = _BEAM_JUDGE_SYSTEM_PROMPT.format(
                criterion=criterion,
                response=response,
            )

            try:
                judge_response = await llm_client.acreate_structured_output(
                    text_input=prompt,
                    system_prompt="You are an evaluation judge. Return only valid JSON.",
                    response_model=str,
                )

                verdict = _parse_verdict(str(judge_response))
                total_score += verdict.score

                verdicts.append(
                    {
                        "criterion": criterion,
                        "score": verdict.score,
                        "reason": verdict.reason,
                    }
                )

            except Exception as e:
                logger.warning(f"BEAM judge failed for criterion: {criterion}: {e}")
                verdicts.append(
                    {
                        "criterion": criterion,
                        "score": 0.0,
                        "reason": f"ERROR: {e}",
                    }
                )

        self._verdicts = verdicts
        self.score = total_score / len(rubric) if rubric else 0.0

        scores = [v["score"] for v in verdicts]
        full = scores.count(1.0)
        partial = scores.count(0.5)
        none_ = scores.count(0.0)
        self.reason = (
            f"BEAM rubric: {full} full, {partial} partial, {none_} none "
            f"out of {len(rubric)} criteria (score={self.score:.3f})"
        )

        return self.score

    @property
    def verdicts(self) -> List[Dict[str, Any]]:
        return self._verdicts
