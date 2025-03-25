from cognee.shared.logging_utils import get_logger
import json
from typing import List, Optional
from cognee.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
    retriever_options,
)
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from cognee.modules.data.models.answers_base import AnswersBase
from cognee.modules.data.models.answers_data import Answers


logger = get_logger()


async def create_and_insert_answers_table(questions_payload):
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        LocalStorage.ensure_directory_exists(relational_config.db_path)

    async with relational_engine.engine.begin() as connection:
        if len(AnswersBase.metadata.tables.keys()) > 0:
            await connection.run_sync(AnswersBase.metadata.create_all)

    async with relational_engine.get_async_session() as session:
        data_point = Answers(payload=questions_payload)
        session.add(data_point)
        await session.commit()


async def run_question_answering(
    params: dict, system_prompt="answer_simple_question.txt", top_k: Optional[int] = None
) -> List[dict]:
    if params.get("answering_questions"):
        logger.info("Question answering started...")
        try:
            with open(params["questions_path"], "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {params['questions_path']}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {params['questions_path']}: {e}")

        logger.info(f"Loaded {len(questions)} questions from {params['questions_path']}")
        answer_generator = AnswerGeneratorExecutor()
        answers = await answer_generator.question_answering_non_parallel(
            questions=questions,
            retriever=retriever_options[params["qa_engine"]](
                system_prompt_path=system_prompt, top_k=top_k
            ),
        )
        with open(params["answers_path"], "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        await create_and_insert_answers_table(answers)
        logger.info("Question answering End...")

        return answers
    else:
        logger.info(
            "The question answering module was not executed as answering_questions is not enabled"
        )
        return []
