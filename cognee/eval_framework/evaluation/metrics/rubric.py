"""Rubric-based evaluation metric for BEAM benchmark.

Scores an LLM response against a list of rubric criteria using an LLM judge.
Each rubric item describes something the response "should contain" or "should state".
The score is the fraction of rubric items satisfied (0.0 to 1.0).

Unlike DeepEval's GEval, this metric:
  - Evaluates each rubric item independently (not holistically)
  - Returns per-item verdicts for transparency
  - Does not require deepeval's GEval infrastructure
"""

from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _get_llm_client():
    from cognee.infrastructure.llm.get_llm_client import get_llm_client

    return get_llm_client()


_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge. You will be given:
- A question that was asked
- An LLM response to evaluate
- A single rubric criterion that the response should satisfy

Determine whether the response satisfies the criterion.
Respond with exactly one word: "YES" or "NO".
Do not explain."""

_JUDGE_USER_PROMPT = """Question: {question}

Response to evaluate:
{response}

Rubric criterion: {criterion}

Does the response satisfy this criterion? Answer YES or NO."""


class RubricMetric:
    """Evaluates an LLM response against a rubric (list of criteria).

    The score is the fraction of rubric items satisfied.
    Requires an LLM client for judging — uses cognee's LLM infrastructure.

    Usage::

        metric = RubricMetric()
        metric.measure(test_case)
        # test_case must have: input, actual_output, expected_output (ignored),
        # and a "rubric" field (list of strings) in additional_metadata
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self._verdicts: List[Dict[str, Any]] = []

    def measure(self, test_case) -> float:
        """Synchronous measure — runs the async version in a loop.

        Accepts a deepeval LLMTestCase or any object with:
          - input (str): the question
          - actual_output (str): the LLM response
          - additional_metadata (dict): must contain "rubric" (list of str)
        """
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
        """Evaluate the response against each rubric item using an LLM judge."""
        question = test_case.input
        response = test_case.actual_output

        # Get rubric from additional_metadata
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        rubric: List[str] = metadata.get("rubric", [])

        if not rubric:
            self.score = 0.0
            self.reason = "No rubric items provided"
            self._verdicts = []
            return self.score

        llm_client = _get_llm_client()

        verdicts = []
        satisfied = 0

        for criterion in rubric:
            prompt = _JUDGE_USER_PROMPT.format(
                question=question,
                response=response,
                criterion=criterion,
            )

            try:
                judge_response = await llm_client.acreate_structured_output(
                    text_input=prompt,
                    system_prompt=_JUDGE_SYSTEM_PROMPT,
                    response_model=str,
                )

                # Parse YES/NO from response
                answer = str(judge_response).strip().upper()
                passed = answer.startswith("YES")

                if passed:
                    satisfied += 1

                verdicts.append(
                    {
                        "criterion": criterion,
                        "passed": passed,
                        "raw_response": str(judge_response).strip(),
                    }
                )

            except Exception as e:
                logger.warning(f"Rubric judge failed for criterion: {criterion}: {e}")
                verdicts.append(
                    {
                        "criterion": criterion,
                        "passed": False,
                        "raw_response": f"ERROR: {e}",
                    }
                )

        self._verdicts = verdicts
        self.score = satisfied / len(rubric) if rubric else 0.0

        failed_items = [v["criterion"] for v in verdicts if not v["passed"]]

        parts = [f"Rubric: {satisfied}/{len(rubric)} criteria satisfied."]
        if failed_items:
            parts.append(f"Failed: {'; '.join(failed_items[:3])}")
        self.reason = " ".join(parts)

        return self.score

    @property
    def verdicts(self) -> List[Dict[str, Any]]:
        """Per-criterion verdicts from the last evaluation."""
        return self._verdicts
