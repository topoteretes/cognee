import logging
import json
from evals.eval_framework.evaluation.evaluation_executor import EvaluationExecutor
from evals.eval_framework.metrics_dashboard import generate_metrics_dashboard
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from cognee.modules.data.models.metrics_data import Metrics
from cognee.modules.data.models.metrics_base import MetricsBase


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

        await create_and_insert_metrics_table(metrics)

        logging.info("Evaluation End...")

    if params.get("dashboard"):
        generate_metrics_dashboard(
            json_data=params["metrics_path"],
            output_file=params["dashboard_path"],
            benchmark=params["benchmark"],
        )
