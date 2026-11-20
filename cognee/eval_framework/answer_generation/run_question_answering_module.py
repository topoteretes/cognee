import json

from cognee.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
    retriever_options,
)
from cognee.eval_framework.integrity.checkpoint import AnswerCheckpoint
from cognee.eval_framework.integrity.fingerprint import answers_fingerprint
from cognee.eval_framework.integrity.io import atomic_write_json
from cognee.infrastructure.databases.relational.get_relational_engine import (
    get_relational_config,
    get_relational_engine,
)
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.data.models.answers_base import AnswersBase
from cognee.modules.data.models.answers_data import Answers
from cognee.shared.logging_utils import get_logger

logger = get_logger()

DEFAULT_SYSTEM_PROMPT = "answer_simple_question_benchmark.txt"


async def create_and_insert_answers_table(questions_payload):
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        await get_file_storage(relational_config.db_path).ensure_directory_exists()

    async with relational_engine.engine.begin() as connection:
        if len(AnswersBase.metadata.tables.keys()) > 0:
            await connection.run_sync(AnswersBase.metadata.create_all)

    async with relational_engine.get_async_session() as session:
        data_point = Answers(payload=questions_payload)
        session.add(data_point)
        await session.commit()


def answers_fingerprint_for(
    params: dict, system_prompt: str = DEFAULT_SYSTEM_PROMPT, top_k: int | None = None
) -> str:
    """Fingerprint of everything that decides what this stage would answer (see
    ``integrity.fingerprint``): config, questions file, prompts, top_k and LLM."""
    from cognee.infrastructure.llm.config import get_llm_config

    if params["qa_engine"] == "beam_router":
        from cognee.eval_framework.answer_generation.beam_router import BEAMRouter

        prompts = BEAMRouter.all_prompts()
    else:
        prompts = read_query_prompt(system_prompt)
    llm = get_llm_config()
    return answers_fingerprint(
        params,
        params["questions_path"],
        system_prompt,
        prompts,
        top_k,
        {"model": llm.llm_model, "provider": llm.llm_provider, "temperature": llm.llm_temperature},
    )


async def run_question_answering(
    params: dict, system_prompt=DEFAULT_SYSTEM_PROMPT, top_k: int | None = None
) -> list[dict]:
    """Answer the questions file and write the answers file.

    A retrieval or completion error propagates unless ``max_answer_failure_rate`` is set in
    ``params``, in which case the failed question is recorded with a failed ``status`` and the
    run goes on. ``resume_verified`` (set by the runner once the manifest and fingerprint check
    passed) reuses the ok answers already in the journal.
    """
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

        record_failures = params.get("max_answer_failure_rate") is not None
        checkpoint = AnswerCheckpoint(
            params["answers_path"], questions, resume=params.get("resume_verified", False)
        )
        pending = checkpoint.pending()
        if checkpoint.reused:
            logger.info(
                f"Resuming: reusing {checkpoint.reused} ok answers, answering {len(pending)} more"
            )

        if params["qa_engine"] == "beam_router":
            from cognee.eval_framework.answer_generation.beam_router import BEAMRouter

            router = BEAMRouter()
            await router.answer_questions(pending, on_answer=checkpoint.record)  # records failures
        else:
            answer_generator = AnswerGeneratorExecutor()
            await answer_generator.question_answering_non_parallel(
                questions=pending,
                retriever=retriever_options[params["qa_engine"]](
                    system_prompt_path=system_prompt, top_k=top_k
                ),
                on_answer=checkpoint.record,
                record_failures=record_failures,
            )
        answers = checkpoint.assemble()
        atomic_write_json(params["answers_path"], answers)

        await create_and_insert_answers_table(answers)
        logger.info("Question answering End...")

        return answers
    else:
        logger.info(
            "The question answering module was not executed as answering_questions is not enabled"
        )
        return []
