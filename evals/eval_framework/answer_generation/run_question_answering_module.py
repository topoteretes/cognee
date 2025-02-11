import logging
import json
from evals.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
    question_answering_engine_options,
)
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_engine,
    get_relational_config,
)
from cognee.modules.data.models.answers_base import AnswersBase
from cognee.modules.data.models.answers_data import Answers


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


async def run_question_answering(params: dict) -> None:
    if params.get("answering_questions"):
        logging.info("Question answering started...")
        try:
            with open(params["questions_path"], "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {params['questions_path']}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {params['questions_path']}: {e}")

        logging.info(f"Loaded {len(questions)} questions from {params['questions_path']}")
        answer_generator = AnswerGeneratorExecutor()
        answers = await answer_generator.question_answering_non_parallel(
            questions=questions,
            answer_resolver=question_answering_engine_options[params["qa_engine"]],
        )
        with open(params["answers_path"], "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        await create_and_insert_answers_table(answers)
        logging.info("Question answering End...")
