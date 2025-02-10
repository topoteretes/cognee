import logging
import json
from evals.eval_framework.evaluation.evaluation_executor import EvaluationExecutor
from evals.eval_framework.metrics_dashboard import generate_metrics_dashboard


async def run_evaluation(params: dict) -> None:
    if params.get("evaluating_answers"):
        logging.info("Evaluation started...")
        try:
            with open(params["answers_path"], "r", encoding="utf-8") as f:
                answers = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {params['answers_path']}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {params['answers_path']}: {e}")

        logging.info(f"Loaded {len(answers)} answers from {params['answers_path']}")
        evaluator = EvaluationExecutor(evaluator_engine=params["evaluation_engine"])
        metrics = await evaluator.execute(
            answers=answers, evaluator_metrics=params["evaluation_metrics"]
        )
        with open(params["metrics_path"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        logging.info("Evaluation End...")

    if params.get("dashboard"):
        generate_metrics_dashboard(
            json_data=params["metrics_path"],
            output_file=params["dashboard_path"],
            benchmark=params["benchmark"],
        )
