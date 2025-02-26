import logging
import json
from cognee.infrastructure.files.storage import LocalStorage
from evals.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from cognee.modules.data.models.questions_base import QuestionsBase
from cognee.modules.data.models.questions_data import Questions
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from evals.eval_framework.corpus_builder.task_getters.TaskGetters import TaskGetters


async def create_and_insert_questions_table(questions_payload):
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        LocalStorage.ensure_directory_exists(relational_config.db_path)

    async with relational_engine.engine.begin() as connection:
        if len(QuestionsBase.metadata.tables.keys()) > 0:
            await connection.run_sync(QuestionsBase.metadata.create_all)

    async with relational_engine.get_async_session() as session:
        data_point = Questions(payload=questions_payload)
        session.add(data_point)
        await session.commit()


async def run_corpus_builder(params: dict) -> None:
    if params.get("building_corpus_from_scratch"):
        logging.info("Corpus Builder started...")

        try:
            task_getter = TaskGetters(params.get("task_getter_type", "Default")).getter_func
        except KeyError:
            raise ValueError(f"Invalid task getter type: {params.get('task_getter_type')}")

        corpus_builder = CorpusBuilderExecutor(
            benchmark=params["benchmark"],
            task_getter=task_getter,
        )
        questions = await corpus_builder.build_corpus(
            limit=params.get("number_of_samples_in_corpus")
        )
        with open(params["questions_path"], "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        await create_and_insert_questions_table(questions_payload=questions)

        logging.info("Corpus Builder End...")
