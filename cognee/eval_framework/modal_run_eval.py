import modal
import os
import json
import asyncio
import datetime
from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

logger = get_logger()


def read_and_combine_metrics(eval_params: dict) -> dict:
    """Read and combine metrics files into a single result dictionary."""
    try:
        with open(eval_params["metrics_path"], "r") as f:
            metrics = json.load(f)
        with open(eval_params["aggregate_metrics_path"], "r") as f:
            aggregate_metrics = json.load(f)

        return {
            "task_getter_type": eval_params["task_getter_type"],
            "number_of_samples": eval_params["number_of_samples_in_corpus"],
            "metrics": metrics,
            "aggregate_metrics": aggregate_metrics,
        }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading metrics files: {e}")
        return None


app = modal.App("modal-run-eval")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env(
        {
            "ENV": os.getenv("ENV"),
            "LLM_API_KEY": os.getenv("LLM_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        }
    )
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2", "deepeval", "gdown", "plotly")
)


@app.function(image=image, concurrency_limit=2, timeout=1800, retries=1)
async def modal_run_eval(eval_params=None):
    """Runs evaluation pipeline and returns combined metrics results."""
    if eval_params is None:
        eval_params = EvalConfig().to_dict()

    logger.info(f"Running evaluation with params: {eval_params}")

    # Run the evaluation pipeline
    await run_corpus_builder(eval_params)
    await run_question_answering(eval_params)
    await run_evaluation(eval_params)

    # Early return if metrics calculation wasn't requested
    if not eval_params.get("evaluating_answers") or not eval_params.get("calculate_metrics"):
        logger.info(
            "Skipping metrics collection as either evaluating_answers or calculate_metrics is False"
        )
        return None

    return read_and_combine_metrics(eval_params)


@app.local_entrypoint()
async def main():
    # List of configurations to run
    configs = [
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=2,
            building_corpus_from_scratch=True,
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=False,
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=10,
            building_corpus_from_scratch=True,
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=False,
        ),
    ]

    # Run evaluations in parallel with different configurations
    modal_tasks = [modal_run_eval.remote.aio(config.to_dict()) for config in configs]
    results = await asyncio.gather(*modal_tasks)

    # Filter out None results and save combined results
    results = [r for r in results if r is not None]
    if results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"combined_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Completed parallel evaluation runs. Results saved to {output_file}")
    else:
        logger.info("No metrics were collected from any of the evaluation runs")
