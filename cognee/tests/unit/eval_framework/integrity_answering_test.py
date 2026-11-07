import json

import pytest

from cognee.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from cognee.eval_framework.answer_generation.beam_router import BEAMRouter
from cognee.eval_framework.integrity.answering import answer_question
from cognee.eval_framework.integrity.status import AnswerStatus, answer_status
from cognee.tests.unit.eval_framework.integrity_helpers import StubRetriever, make_questions


@pytest.mark.asyncio
async def test_answer_question_ok():
    outcome = await answer_question(StubRetriever(), "q0")
    assert outcome == {
        "answer": "answer to q0",
        "retrieval_context": "context for q0",
        "status": "ok",
    }


@pytest.mark.asyncio
async def test_retrieval_failure_is_a_status_not_answer_text():
    outcome = await answer_question(StubRetriever(fail_on={"q0"}, fail_in="retrieval"), "q0")
    assert outcome["status"] == AnswerStatus.RETRIEVAL_ERROR.value
    assert outcome["answer"] == ""
    assert outcome["error"] == "RuntimeError: vector store unavailable"


@pytest.mark.asyncio
async def test_llm_failure_keeps_the_retrieved_context():
    outcome = await answer_question(StubRetriever(fail_on={"q0"}, fail_in="llm"), "q0")
    assert outcome["status"] == AnswerStatus.LLM_ERROR.value
    assert outcome["retrieval_context"] == "context for q0"
    assert outcome["error"].startswith("TimeoutError")


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", [[], [""], ["   "], [None], None, ""])
async def test_blank_completion_is_empty_not_ok(completion):
    # fixed_answer=None means "default answer", so route blank values through a subclass.
    class Blank(StubRetriever):
        async def get_completion_from_context(self, query, retrieved_objects, context, **kwargs):
            return completion

    outcome = await answer_question(Blank(), "q0")
    assert outcome["status"] == AnswerStatus.EMPTY.value
    assert outcome["answer"] == ""


@pytest.mark.asyncio
async def test_executor_records_failures_and_keeps_going():
    retriever = StubRetriever(fail_on={"q1"})
    seen = []
    answers = await AnswerGeneratorExecutor().question_answering_non_parallel(
        make_questions(3), retriever, on_answer=seen.append, record_failures=True
    )
    assert [answer_status(a) for a in answers] == [
        AnswerStatus.OK,
        AnswerStatus.RETRIEVAL_ERROR,
        AnswerStatus.OK,
    ]
    assert "ERROR" not in json.dumps(answers)
    assert answers[1]["golden_answer"] == "golden 1"
    assert seen == answers


@pytest.mark.asyncio
async def test_beam_router_failure_is_not_scored_as_an_answer():
    router = BEAMRouter()
    router._retriever_cache["GraphCompletionRetriever"] = StubRetriever(fail_on={"q0"})
    questions = [{**q, "question_type": "information_extraction"} for q in make_questions(2)]
    answers = await router.answer_questions(questions)
    assert answers[0]["status"] == "retrieval_error"
    assert not answers[0]["answer"].startswith("ERROR")
    assert answers[0]["question_type"] == "information_extraction"
    assert answers[1]["status"] == "ok"


@pytest.mark.asyncio
async def test_executor_propagates_answer_errors_by_default():
    with pytest.raises(RuntimeError, match="vector store unavailable"):
        await AnswerGeneratorExecutor().question_answering_non_parallel(
            make_questions(2), StubRetriever(fail_on={"q1"})
        )
