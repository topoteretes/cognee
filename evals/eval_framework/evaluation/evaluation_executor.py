from typing import List, Dict, Any, Union
from evals.eval_framework.evaluation.evaluator_adapters import EvaluatorAdapter
from evals.eval_framework.evaluation.naive_adapter import NaiveAdapter
from evals.eval_framework.evaluation.deep_eval_adapter import DeepEvalAdapter


class EvaluationExecutor:
    def __init__(self, evaluator_engine: Union[str, EvaluatorAdapter, Any] = "DeepEval") -> None:
        if isinstance(evaluator_engine, EvaluatorAdapter):
            self.eval_adapter = evaluator_engine.adapter_class()
            return

        if not isinstance(evaluator_engine, str):
            self.eval_adapter = evaluator_engine
            return

        if evaluator_engine == "Naive":
            self.eval_adapter = NaiveAdapter()
            return
        if evaluator_engine == "DeepEval":
            self.eval_adapter = DeepEvalAdapter()
            return

        try:
            adapter_enum = EvaluatorAdapter(evaluator_engine)
            self.eval_adapter = adapter_enum.adapter_class()
        except ValueError:
            raise ValueError(f"Unsupported evaluator: {evaluator_engine}")

    async def execute(self, answers: List[Dict[str, str]], evaluator_metrics: Any) -> Any:
        metrics = await self.eval_adapter.evaluate_answers(answers, evaluator_metrics)
        return metrics
