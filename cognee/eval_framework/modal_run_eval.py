import modal
import os
import asyncio
import datetime
import hashlib
import json
from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder
from cognee.eval_framework.answer_generation.run_question_answering_module import (
    run_question_answering,
)
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

logger = get_logger()
vol = modal.Volume.from_name("evaluation_dashboard_results", create_if_missing=True)


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
    .pip_install("protobuf", "h2", "deepeval", "gdown", "plotly")
)


@app.function(image=image, concurrency_limit=10, timeout=86400, volumes={"/data": vol})
async def modal_run_eval(eval_params=None):
    """Runs evaluation pipeline and returns combined metrics results."""
    if eval_params is None:
        eval_params = EvalConfig().to_dict()

    version_name = "baseline"
    benchmark_name = os.environ.get("BENCHMARK", eval_params.get("benchmark", "benchmark"))
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    answers_filename = (
        f"{version_name}_{benchmark_name}_{timestamp}_{eval_params.get('answers_path')}"
    )
    html_filename = (
        f"{version_name}_{benchmark_name}_{timestamp}_{eval_params.get('dashboard_path')}"
    )

    logger.info(f"Running evaluation with params: {eval_params}")

    # Run the evaluation pipeline
    await run_corpus_builder(eval_params, instance_filter=eval_params.get("instance_filter"))
    await run_question_answering(eval_params)
    answers = await run_evaluation(eval_params)

    with open("/data/" + answers_filename, "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    vol.commit()

    if eval_params.get("dashboard"):
        logger.info("Generating dashboard...")
        html_output = create_dashboard(
            metrics_path=eval_params["metrics_path"],
            aggregate_metrics_path=eval_params["aggregate_metrics_path"],
            output_file=eval_params["dashboard_path"],
            benchmark=eval_params["benchmark"],
        )

    with open("/data/" + html_filename, "w") as f:
        f.write(html_output)
    vol.commit()

    logger.info("Evaluation set finished...")

    return True


@app.local_entrypoint()
async def main():
    # List of configurations to run
    configs = [
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=10,
            benchmark="HotPotQA",
            qa_engine="cognee_graph_completion",
            building_corpus_from_scratch=True,
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=10,
            benchmark="TwoWikiMultiHop",
            qa_engine="cognee_graph_completion",
            building_corpus_from_scratch=True,
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
        ),
        EvalConfig(
            task_getter_type="Default",
            number_of_samples_in_corpus=10,
            benchmark="Musique",
            qa_engine="cognee_graph_completion",
            building_corpus_from_scratch=True,
            answering_questions=True,
            evaluating_answers=True,
            calculate_metrics=True,
            dashboard=True,
        ),
    ]

    # Run evaluations in parallel with different configurations
    modal_tasks = [modal_run_eval.remote.aio(config.to_dict()) for config in configs]
    await asyncio.gather(*modal_tasks)
