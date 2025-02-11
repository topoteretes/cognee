from typing import List, Dict, Any, Union
from evals.eval_framework.evaluation.evaluator_adapters import EvaluatorAdapter


class EvaluationExecutor:
    def __init__(self, evaluator_engine: Union[str, EvaluatorAdapter, Any] = "DeepEval") -> None:
        if isinstance(evaluator_engine, str):
            try:
                adapter_enum = EvaluatorAdapter(evaluator_engine)
            except ValueError:
                raise ValueError(f"Unsupported evaluator: {evaluator_engine}")
            self.eval_adapter = adapter_enum.adapter_class()
        elif isinstance(evaluator_engine, EvaluatorAdapter):
            self.eval_adapter = evaluator_engine.adapter_class()
        else:
            self.eval_adapter = evaluator_engine

    async def execute(self, answers: List[Dict[str, str]], evaluator_metrics: Any) -> Any:
        metrics = await self.eval_adapter.evaluate_answers(answers, evaluator_metrics)
        return metrics
