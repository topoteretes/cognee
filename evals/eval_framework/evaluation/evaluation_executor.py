from typing import List, Dict, Any
from evals.eval_framework.evaluation.deep_eval_adapter import DeepEvalAdapter


class EvaluationExecutor:
    evaluator_options = {"DeepEval": DeepEvalAdapter}

    eval_adapter = None

    async def execute(self, answers: List[Dict[str, str]], evaluator_engine, evaluator_metrics):
        if evaluator_engine not in self.evaluator_options:
            raise ValueError(f"Unsupported evaluator: {evaluator_engine}")

        self.eval_adapter = self.evaluator_options[evaluator_engine]()

        metrics = await self.eval_adapter.evaluate_answers(answers, evaluator_metrics)

        return metrics
