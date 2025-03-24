from cognee.shared.logging_utils import get_logger
import json
from typing import List
from cognee.eval_framework.evaluation.evaluation_executor import EvaluationExecutor
from cognee.eval_framework.analysis.metrics_calculator import calculate_metrics_statistics
from cognee.eval_framework.analysis.dashboard_generator import create_dashboard
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from cognee.modules.data.models.metrics_data import Metrics
from cognee.modules.data.models.metrics_base import MetricsBase


logger = get_logger()


async def create_and_insert_metrics_table(questions_payload):
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        LocalStorage.ensure_directory_exists(relational_config.db_path)

    async with relational_engine.engine.begin() as connection:
        if len(MetricsBase.metadata.tables.keys()) > 0:
            await connection.run_sync(MetricsBase.metadata.create_all)

    async with relational_engine.get_async_session() as session:
        data_point = Metrics(payload=questions_payload)
        session.add(data_point)
        await session.commit()


async def execute_evaluation(params: dict) -> None:
    """Execute the evaluation step and save results."""
    logger.info("Evaluation started...")
    try:
        with open(params["answers_path"], "r", encoding="utf-8") as f:
            answers = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {params['answers_path']}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {params['answers_path']}: {e}")

    logger.info(f"Loaded {len(answers)} answers from {params['answers_path']}")
    evaluator = EvaluationExecutor(
        evaluator_engine=params["evaluation_engine"],
        evaluate_contexts=params["evaluating_contexts"],
    )
    metrics = await evaluator.execute(
        answers=answers, evaluator_metrics=params["evaluation_metrics"]
    )
    with open(params["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    await create_and_insert_metrics_table(metrics)
    logger.info("Evaluation completed")
    return metrics


async def run_evaluation(params: dict) -> List[dict]:
    """Run each step of the evaluation pipeline based on configuration flags."""
    # Step 1: Evaluate answers if requested
    if params.get("evaluating_answers"):
        metrics = await execute_evaluation(params)
    else:
        logger.info("Skipping evaluation as evaluating_answers is False")

    # Step 2: Calculate metrics if requested
    if params.get("calculate_metrics"):
        logger.info("Calculating metrics statistics...")
        calculate_metrics_statistics(
            json_data=params["metrics_path"], aggregate_output_path=params["aggregate_metrics_path"]
        )
        logger.info("Metrics calculation completed")
        return metrics
    else:
        logger.info("Skipping metrics calculation as calculate_metrics is False")
        return []
