from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from cognee.eval_framework.evaluation.metrics.exact_match import ExactMatchMetric
from cognee.eval_framework.evaluation.metrics.f1 import F1ScoreMetric
from cognee.eval_framework.evaluation.metrics.context_coverage import ContextCoverageMetric
from typing import Any, Dict, List
from deepeval.metrics import ContextualRelevancyMetric
import time
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class DeepEvalAdapter(BaseEvalAdapter):
    def __init__(self):
        self.n_retries = 5
        self.g_eval_metrics = {
            "correctness": self.g_eval_correctness(),
            "EM": ExactMatchMetric(),
            "f1": F1ScoreMetric(),
            "contextual_relevancy": ContextualRelevancyMetric(),
            "context_coverage": ContextCoverageMetric(),
        }

    def _calculate_metric(self, metric: str, test_case: LLMTestCase) -> Dict[str, Any]:
        """Calculate a single metric for a test case with retry logic."""
        metric_to_calculate = self.g_eval_metrics[metric]

        for attempt in range(self.n_retries):
            try:
                metric_to_calculate.measure(test_case)
                return {
                    "score": metric_to_calculate.score,
                    "reason": metric_to_calculate.reason,
                }
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.n_retries} failed for metric '{metric}': {e}"
                )
                if attempt < self.n_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All {self.n_retries} attempts failed for metric '{metric}'. Returning None values."
                    )

        return {
            "score": None,
            "reason": None,
        }

    async def evaluate_answers(
        self, answers: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        # evaluator_metrics contains all the necessary metrics that are gonna be evaluated dynamically
        for metric in evaluator_metrics:
            if metric not in self.g_eval_metrics:
                raise ValueError(f"Unsupported metric: {metric}")

        results = []
        for answer in answers:
            test_case = LLMTestCase(
                input=answer["question"],
                actual_output=answer["answer"],
                expected_output=answer["golden_answer"],
                retrieval_context=[answer["retrieval_context"]]
                if "golden_context" in answer
                else None,
                context=[answer["golden_context"]] if "golden_context" in answer else None,
            )
            metric_results = {}
            for metric in evaluator_metrics:
                metric_results[metric] = self._calculate_metric(metric, test_case)
            results.append({**answer, "metrics": metric_results})

        return results

    def g_eval_correctness(self):
        return GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            model=EvalConfig().to_dict()["deepeval_model"],
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )
