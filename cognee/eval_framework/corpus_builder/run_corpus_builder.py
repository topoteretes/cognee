from cognee.shared.logging_utils import get_logger, ERROR
import json
from typing import List, Optional

from cognee.infrastructure.files.storage import LocalStorage
from cognee.eval_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from cognee.modules.data.models.questions_base import QuestionsBase
from cognee.modules.data.models.questions_data import Questions
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.eval_framework.corpus_builder.task_getters.TaskGetters import TaskGetters

logger = get_logger(level=ERROR)


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


async def run_corpus_builder(
    params: dict,
    chunk_size=1024,
    chunker=TextChunker,
    instance_filter=None,
) -> List[dict]:
    if params.get("building_corpus_from_scratch"):
        logger.info("Corpus Builder started...")

        try:
            task_getter = TaskGetters(params.get("task_getter_type", "Default")).getter_func
        except KeyError:
            raise ValueError(f"Invalid task getter type: {params.get('task_getter_type')}")

        corpus_builder = CorpusBuilderExecutor(
            benchmark=params["benchmark"],
            task_getter=task_getter,
        )
        questions = await corpus_builder.build_corpus(
            limit=params.get("number_of_samples_in_corpus"),
            chunker=chunker,
            chunk_size=chunk_size,
            load_golden_context=params.get("evaluating_contexts"),
            instance_filter=instance_filter,
        )
        with open(params["questions_path"], "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        await create_and_insert_questions_table(questions_payload=questions)

        logger.info("Corpus Builder End...")

        return questions
