from typing import List, Dict, Any, Union
from cognee.eval_framework.evaluation.evaluator_adapters import EvaluatorAdapter


class EvaluationExecutor:
    def __init__(
        self,
        evaluator_engine: Union[str, EvaluatorAdapter, Any] = "DeepEval",
        evaluate_contexts: bool = False,
    ) -> None:
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
        self.evaluate_contexts = evaluate_contexts

    async def execute(self, answers: List[Dict[str, str]], evaluator_metrics: Any) -> Any:
        if self.evaluate_contexts:
            evaluator_metrics.append("contextual_relevancy")
            evaluator_metrics.append("context_coverage")
        metrics = await self.eval_adapter.evaluate_answers(answers, evaluator_metrics)
        return metrics
