from cognee.eval_framework.analysis.metrics_calculator import calculate_metrics_statistics
from cognee.eval_framework.evaluation.evaluation_executor import EvaluationExecutor
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.io import atomic_write_json, load_json_list
from cognee.eval_framework.integrity.status import answer_status
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_config,
    get_relational_engine,
)
from cognee.infrastructure.files.storage import get_file_storage
from cognee.modules.data.models.metrics_base import MetricsBase
from cognee.modules.data.models.metrics_data import Metrics
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def create_and_insert_metrics_table(questions_payload):
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        await get_file_storage(relational_config.db_path).ensure_directory_exists()

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
    answers = load_json_list(params["answers_path"], "answers")
    logger.info(f"Loaded {len(answers)} answers from {params['answers_path']}")

    # A failed answer is not an answer: scoring it would put a number on a failure. It is not
    # sent to the evaluator but keeps its place in the metrics file, with no scores.
    scorable = [answer for answer in answers if answer_status(answer).scorable]
    if len(scorable) < len(answers):
        logger.warning(
            f"Not evaluating {len(answers) - len(scorable)} of {len(answers)} answers that failed"
        )
    evaluator = EvaluationExecutor(
        evaluator_engine=params["evaluation_engine"],
        evaluate_contexts=params["evaluating_contexts"],
    )
    scored = (
        await evaluator.execute(answers=scorable, evaluator_metrics=params["evaluation_metrics"])
        if scorable
        else []
    )
    if len(scored) != len(scorable):
        raise RunIntegrityError(
            f"The evaluator returned {len(scored)} results for {len(scorable)} answers"
        )
    scored_iter = iter(scored)
    metrics = [
        next(scored_iter) if answer_status(answer).scorable else {**answer, "metrics": {}}
        for answer in answers
    ]
    atomic_write_json(params["metrics_path"], metrics)

    await create_and_insert_metrics_table(metrics)
    logger.info("Evaluation completed")
    return metrics


async def run_evaluation(params: dict) -> list[dict]:
    """Run each step of the evaluation pipeline based on configuration flags.

    Returns the scored metrics entries when metrics are calculated, otherwise an empty list.
    """
    # Step 1: Evaluate answers if requested
    metrics: list[dict] = []
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
