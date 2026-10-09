"""Async LoCoMo evaluator: token-F1 plus a binary LLM judge, no deepeval dependency.

Registered as the ``LocomoEval`` engine in ``evaluation/evaluator_adapters.py``. Mirrors
``BeamEvalAdapter``: fresh metric instance per (answer, metric), bounded concurrency.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cognee.eval_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from cognee.eval_framework.locomo.metrics.f1 import LocomoF1Metric
from cognee.eval_framework.locomo.metrics.llm_judge import LocomoLLMJudgeMetric

DEFAULT_LOCOMO_EVAL_MAX_CONCURRENT = 8
LOCOMO_METRICS = ("f1", "llm_judge")


@dataclass
class LocomoTestCase:
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str] | None = None
    context: list[str] | None = None
    additional_metadata: dict[str, Any] | None = None


class LocomoEvalAdapter(BaseEvalAdapter):
    def __init__(self, max_concurrent_evaluations: int | None = None):
        env_override = os.getenv("COGNEE_LOCOMO_EVAL_MAX_CONCURRENT")
        if max_concurrent_evaluations is None and env_override:
            try:
                max_concurrent_evaluations = int(env_override)
            except ValueError:
                max_concurrent_evaluations = None
        self.max_concurrent_evaluations = max(
            1, max_concurrent_evaluations or DEFAULT_LOCOMO_EVAL_MAX_CONCURRENT
        )
        self._metric_factories = {
            "f1": LocomoF1Metric,
            "llm_judge": LocomoLLMJudgeMetric,
        }

    @staticmethod
    def _build_test_case(answer: dict[str, Any]) -> LocomoTestCase:
        metadata: dict[str, Any] = {}
        for key in ("question_type", "category", "adversarial_answer", "evidence"):
            if key in answer:
                metadata[key] = answer[key]
        return LocomoTestCase(
            input=answer["question"],
            actual_output=answer.get("answer") or "",
            expected_output=answer.get("golden_answer") or "",
            retrieval_context=[answer["retrieval_context"]]
            if answer.get("retrieval_context")
            else None,
            context=[answer["golden_context"]] if answer.get("golden_context") else None,
            additional_metadata=metadata or None,
        )

    async def _evaluate_metric(
        self, metric_name: str, test_case: LocomoTestCase, semaphore: asyncio.Semaphore
    ) -> dict[str, Any]:
        metric = self._metric_factories[metric_name]()
        async with semaphore:
            try:
                await metric.a_measure(test_case)
            except Exception as error:
                return {"score": None, "reason": f"ERROR: {error}"}
        return {"score": metric.score, "reason": metric.reason}

    async def _evaluate_single_answer(
        self,
        answer: dict[str, Any],
        evaluator_metrics: list[str],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        test_case = self._build_test_case(answer)
        metric_names = list(evaluator_metrics)
        values = await asyncio.gather(
            *(self._evaluate_metric(name, test_case, semaphore) for name in metric_names)
        )
        return {**answer, "metrics": dict(zip(metric_names, values))}

    async def evaluate_answers(
        self, answers: list[dict[str, Any]], evaluator_metrics: list[str]
    ) -> list[dict[str, Any]]:
        if not answers:
            return []
        for metric in evaluator_metrics:
            if metric not in self._metric_factories:
                raise ValueError(
                    f"Unsupported LoCoMo metric: {metric}. Available: {', '.join(LOCOMO_METRICS)}"
                )
        semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
        return list(
            await asyncio.gather(
                *(
                    self._evaluate_single_answer(answer, evaluator_metrics, semaphore)
                    for answer in answers
                )
            )
        )
