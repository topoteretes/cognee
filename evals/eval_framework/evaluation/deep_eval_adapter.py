from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from evals.eval_framework.eval_config import EvalConfig
from evals.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from evals.eval_framework.evaluation.metrics.exact_match import ExactMatchMetric
from evals.eval_framework.evaluation.metrics.f1 import F1ScoreMetric
from typing import Any, Dict, List


class DeepEvalAdapter(BaseEvalAdapter):
    def __init__(self):
        self.g_eval_metrics = {
            "correctness": self.g_eval_correctness(),
            "EM": ExactMatchMetric(),
            "f1": F1ScoreMetric(),
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
            )
            metric_results = {}
            for metric in evaluator_metrics:
                metric_to_calculate = self.g_eval_metrics[metric]
                metric_to_calculate.measure(test_case)
                metric_results[metric] = {
                    "score": metric_to_calculate.score,
                    "reason": metric_to_calculate.reason,
                }
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
