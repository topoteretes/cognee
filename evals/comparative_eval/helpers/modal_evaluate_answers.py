import modal
import os
import asyncio
import datetime
import hashlib
import json
from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.metrics_dashboard import create_dashboard

logger = get_logger()
vol = modal.Volume.from_name("comparison-eval-answers", create_if_missing=True)

app = modal.App("comparison-eval-answerst")

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
async def modal_evaluate_answers(
    answers_json_content: dict, answers_filename: str, eval_config: dict = None
):
    """Evaluates answers from JSON content and returns metrics results."""
    if eval_config is None:
        eval_config = EvalConfig().to_dict()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Create temporary file path for the JSON content
    base_name = os.path.splitext(answers_filename)[0]
    temp_answers_path = f"/data/temp_answers_{base_name}_{timestamp}.json"

    # Write JSON content to temporary file
    with open(temp_answers_path, "w") as f:
        json.dump(answers_json_content, f, ensure_ascii=False, indent=4)

    # Set up output paths with simplified naming: prefix_original_file_name
    eval_params = eval_config.copy()
    eval_params["answers_path"] = temp_answers_path
    eval_params["metrics_path"] = f"/data/metrics_{answers_filename}"
    eval_params["aggregate_metrics_path"] = f"/data/aggregate_metrics_{answers_filename}"
    eval_params["dashboard_path"] = f"/data/dashboard_{os.path.splitext(answers_filename)[0]}.html"

    # eval_params["evaluation_engine"] = "DirectLLM"
    # eval_params["evaluation_metrics"] = ["correctness"]

    logger.info(f"Evaluating answers from: {answers_filename}")
    logger.info(f"Using eval params: {eval_params}")

    try:
        # Only run evaluation (skip corpus building and question answering)
        evaluated_answers = await run_evaluation(eval_params)

        # Save evaluated answers
        evaluated_answers_path = f"/data/evaluated_{answers_filename}"
        with open(evaluated_answers_path, "w") as f:
            json.dump(evaluated_answers, f, ensure_ascii=False, indent=4)
        vol.commit()

        # Generate dashboard if requested
        if eval_params.get("dashboard"):
            logger.info("Generating dashboard...")
            html_output = create_dashboard(
                metrics_path=eval_params["metrics_path"],
                aggregate_metrics_path=eval_params["aggregate_metrics_path"],
                output_file=eval_params["dashboard_path"],
                benchmark=eval_params.get("benchmark", "Unknown"),
            )

            with open(eval_params["dashboard_path"], "w") as f:
                f.write(html_output)
            vol.commit()

        logger.info(f"Evaluation completed for {answers_filename}")

        # Return metrics results
        result = {
            "answers_file": answers_filename,
            "metrics_path": eval_params["metrics_path"],
            "aggregate_metrics_path": eval_params["aggregate_metrics_path"],
            "dashboard_path": eval_params["dashboard_path"]
            if eval_params.get("dashboard")
            else None,
            "evaluated_answers_path": evaluated_answers_path,
        }

        return result

    except Exception as e:
        logger.error(f"Error evaluating {answers_filename}: {e}")
        raise


@app.local_entrypoint()
async def main():
    """Main entry point that evaluates multiple JSON answer files in parallel."""

    json_files_dir = ""
    json_files = [f for f in os.listdir(json_files_dir) if f.endswith(".json")]
    json_file_paths = [os.path.join(json_files_dir, f) for f in json_files]

    # Manually specify your evaluation configuration here
    eval_config = EvalConfig(
        # Only evaluation-related settings
        evaluating_answers=True,
        evaluating_contexts=False,
        evaluation_engine="DeepEval",
        evaluation_metrics=["correctness", "EM", "f1"],
        calculate_metrics=True,
        dashboard=True,
        deepeval_model="gpt-4o-mini",
    ).to_dict()

    logger.info(f"Starting evaluation of {len(json_file_paths)} JSON files")

    # Read JSON files locally and prepare tasks
    modal_tasks = []
    for json_path in json_file_paths:
        try:
            # Read JSON content locally
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)

            filename = os.path.basename(json_path)

            # Create remote evaluation task with JSON content
            task = modal_evaluate_answers.remote.aio(json_content, filename, eval_config)
            modal_tasks.append(task)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading {json_path}: {e}")
            continue

    if not modal_tasks:
        logger.error("No valid JSON files found to process")
        return []

    # Run evaluations in parallel
    results = await asyncio.gather(*modal_tasks, return_exceptions=True)

    # Log results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to evaluate {json_file_paths[i]}: {result}")
        else:
            logger.info(f"Successfully evaluated {result['answers_file']}")

    return results
