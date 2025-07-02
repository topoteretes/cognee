import pytest
from cognee.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from cognee.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_answer_generation():
    limit = 1
    corpus_list, qa_pairs = DummyAdapter().load_corpus(limit=limit)

    mock_retriever = AsyncMock()
    mock_retriever.get_context = AsyncMock(return_value="Mocked retrieval context")
    mock_retriever.get_completion = AsyncMock(return_value=["Mocked answer"])

    answer_generator = AnswerGeneratorExecutor()
    answers = await answer_generator.question_answering_non_parallel(
        questions=qa_pairs,
        retriever=mock_retriever,
    )

    mock_retriever.get_context.assert_any_await(qa_pairs[0]["question"])

    assert len(answers) == len(qa_pairs)
    assert answers[0]["question"] == qa_pairs[0]["question"], (
        "AnswerGeneratorExecutor is passing the question incorrectly"
    )
    assert answers[0]["golden_answer"] == qa_pairs[0]["answer"], (
        "AnswerGeneratorExecutor is passing the golden answer incorrectly"
    )
    assert answers[0]["answer"] == "Mocked answer", (
        "AnswerGeneratorExecutor is passing the generated answer incorrectly"
    )
