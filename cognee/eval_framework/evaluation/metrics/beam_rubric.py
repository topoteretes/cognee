"""BEAM rubric metric with 3-level scoring (0.0, 0.5, 1.0).

Uses the judge prompt from the official BEAM paper repository. Each rubric
criterion is scored independently by an LLM judge. Final score = mean across
all criteria for that question.

Reference: https://github.com/mohammadtavakoli78/BEAM/blob/main/src/evaluation/compute_metrics.py
"""

import json
import re
from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger()


# Exact prompt from the BEAM paper repository (src/prompts.py, line 11547).
# Fix applied: <question> is actually substituted — the paper's code never fills
# it in, leaving the literal string "<question>" for the judge.
_BEAM_JUDGE_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): {question}
- RUBRIC CRITERION (what to check): {criterion}
- RESPONSE TO EVALUATE: {response}

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT (anchored to the QUESTION)
A compliant response must be **on-topic with respect to the QUESTION** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

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


def _parse_verdict(raw: str) -> tuple[float, str]:
    """Parse score and reason from LLM judge response. Returns (score, reason)."""
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    data: dict = {}
    try:
        data = json.loads(text)
        score = float(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group())
                score = float(data["score"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                score_match = re.search(r'"score"\s*:\s*([\d.]+)', text)
                score = float(score_match.group(1)) if score_match else 0.0
        else:
            score = 0.0

    if score >= 0.75:
        score = 1.0
    elif score >= 0.25:
        score = 0.5
    else:
        score = 0.0

    reason = data.get("reason", "") if isinstance(data, dict) else ""
    return score, reason


class BEAMRubricMetric:
    """BEAM rubric metric using the paper's judge prompt.

    Each rubric criterion is scored 0.0, 0.5, or 1.0 by an LLM judge.
    Final score = mean across all criteria for that question.
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
        from cognee.infrastructure.llm.LLMGateway import LLMGateway

        question = test_case.input
        response = test_case.actual_output
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        rubric: List[str] = metadata.get("rubric", [])

        if not rubric:
            self.score = 0.0
            self.reason = "No rubric items provided"
            self._verdicts = []
            return self.score

        verdicts = []
        total_score = 0.0

        for criterion in rubric:
            prompt = _BEAM_JUDGE_PROMPT.format(
                question=question,
                criterion=criterion,
                response=response,
            )

            try:
                raw = await LLMGateway.acreate_structured_output(
                    text_input=prompt,
                    system_prompt="You are an evaluation judge. Return only valid JSON.",
                    response_model=str,
                )
                score, reason = _parse_verdict(str(raw))
            except Exception as e:
                logger.warning(f"BEAM judge failed for criterion '{criterion}': {e}")
                score, reason = 0.0, f"ERROR: {e}"

            total_score += score
            verdicts.append({"criterion": criterion, "score": score, "reason": reason})

        self._verdicts = verdicts
        self.score = total_score / len(rubric)

        full = sum(1 for v in verdicts if v["score"] == 1.0)
        partial = sum(1 for v in verdicts if v["score"] == 0.5)
        none_ = sum(1 for v in verdicts if v["score"] == 0.0)
        self.reason = (
            f"BEAM rubric: {full} full, {partial} partial, {none_} none "
            f"out of {len(rubric)} criteria (score={self.score:.3f})"
        )

        return self.score

    @property
    def verdicts(self) -> List[Dict[str, Any]]:
        return self._verdicts
