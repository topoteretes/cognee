import pytest
from evals.eval_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from evals.eval_framework.benchmark_adapters.dummy_adapter import DummyAdapter
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_answer_generation():
    limit = 1
    corpus_list, qa_pairs = DummyAdapter().load_corpus(limit=limit)

    mock_answer_resolver = AsyncMock()
    mock_answer_resolver.side_effect = lambda query: ["mock_answer"]

    answer_generator = AnswerGeneratorExecutor()
    answers = await answer_generator.question_answering_non_parallel(
        questions=qa_pairs,
        answer_resolver=mock_answer_resolver,
    )

    mock_answer_resolver.assert_called_once_with(qa_pairs[0]["question"])

    assert len(answers) == len(qa_pairs)
    assert answers[0]["question"] == qa_pairs[0]["question"], (
        "AnswerGeneratorExecutor is passing the question incorrectly"
    )
    assert answers[0]["golden_answer"] == qa_pairs[0]["answer"], (
        "AnswerGeneratorExecutor is passing the golden answer incorrectly"
    )
    assert answers[0]["answer"] == "mock_answer", (
        "AnswerGeneratorExecutor is passing the generated answer incorrectly"
    )
