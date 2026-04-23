import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cognee.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from cognee.eval_framework.evaluation.metrics.beam_rubric import BEAMRubricMetric
from cognee.eval_framework.evaluation.metrics.kendall_tau import KendallTauMetric


DEFAULT_BEAM_EVAL_MAX_CONCURRENT = 10


@dataclass
class BeamLLMTestCase:
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: Optional[list[str]] = None
    context: Optional[list[str]] = None
    additional_metadata: Optional[dict[str, Any]] = None


class BeamEvalAdapter(BaseEvalAdapter):
    """Dedicated async evaluator for BEAM metrics.

    Reuses the existing BEAM metric implementations and prompts, but evaluates
    answers concurrently with fresh metric instances per task.
    """

    def __init__(self, max_concurrent_evaluations: Optional[int] = None):
        env_override = os.getenv("COGNEE_BEAM_EVAL_MAX_CONCURRENT")
        if max_concurrent_evaluations is None and env_override:
            try:
                max_concurrent_evaluations = int(env_override)
            except ValueError:
                max_concurrent_evaluations = None

        self.max_concurrent_evaluations = max(
            1, max_concurrent_evaluations or DEFAULT_BEAM_EVAL_MAX_CONCURRENT
        )
        self._metric_factories = {
            "beam_rubric": BEAMRubricMetric,
            "kendall_tau": KendallTauMetric,
        }

    def _build_test_case(self, answer: Dict[str, Any]) -> BeamLLMTestCase:
        additional_metadata = {}
        if "rubric" in answer:
            additional_metadata["rubric"] = answer["rubric"]
        if "question_type" in answer:
            additional_metadata["question_type"] = answer["question_type"]

        return BeamLLMTestCase(
            input=answer["question"],
            actual_output=answer["answer"],
            expected_output=answer["golden_answer"],
            retrieval_context=[answer["retrieval_context"]] if "golden_context" in answer else None,
            context=[answer["golden_context"]] if "golden_context" in answer else None,
            additional_metadata=additional_metadata or None,
        )

    async def _evaluate_metric(
        self,
        metric_name: str,
        test_case: BeamLLMTestCase,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        metric = self._metric_factories[metric_name]()

        async with semaphore:
            if hasattr(metric, "a_measure"):
                await metric.a_measure(test_case)
            else:
                await asyncio.to_thread(metric.measure, test_case)

        return {
            "score": getattr(metric, "score", None),
            "reason": getattr(metric, "reason", None),
        }

    async def _evaluate_single_answer(
        self,
        answer: Dict[str, Any],
        evaluator_metrics: List[str],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        test_case = self._build_test_case(answer)
        metric_names = list(evaluator_metrics)
        metric_tasks = [
            asyncio.create_task(self._evaluate_metric(metric_name, test_case, semaphore))
            for metric_name in metric_names
        ]
        metric_values = await asyncio.gather(*metric_tasks)
        return {
            **answer,
            "metrics": {
                metric_name: metric_value
                for metric_name, metric_value in zip(metric_names, metric_values)
            },
        }

    async def evaluate_answers(
        self, answers: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        if not answers:
            return []

        for metric in evaluator_metrics:
            if metric not in self._metric_factories:
                raise ValueError(f"Unsupported metric: {metric}")

        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        tasks = [
            asyncio.create_task(self._evaluate_single_answer(answer, evaluator_metrics, semaphore))
            for answer in answers
        ]
        return await asyncio.gather(*tasks)
