import asyncio

import pytest

from cognee.eval_framework.evaluation.beam_eval_adapter import BeamEvalAdapter


def make_answer(question_idx: int) -> dict:
    return {
        "conversation_id": "conv-1",
        "question_idx": question_idx,
        "question": f"Question {question_idx}?",
        "answer": f"Answer {question_idx}",
        "golden_answer": f"Golden {question_idx}",
        "retrieval_context": f"Context {question_idx}",
        "question_type": "summarization",
        "rubric": [f"Criterion {question_idx}"],
    }


@pytest.mark.asyncio
async def test_beam_eval_adapter_empty_answers():
    adapter = BeamEvalAdapter()

    results = await adapter.evaluate_answers([], ["beam_rubric"])

    assert results == []


@pytest.mark.asyncio
async def test_beam_eval_adapter_unsupported_metric():
    adapter = BeamEvalAdapter()

    with pytest.raises(ValueError, match="Unsupported metric: EM"):
        await adapter.evaluate_answers([make_answer(0)], ["EM"])


@pytest.mark.asyncio
async def test_beam_eval_adapter_preserves_output_shape():
    adapter = BeamEvalAdapter(max_concurrent_evaluations=2)

    class FakeRubricMetric:
        def __init__(self):
            self.score = None
            self.reason = None

        async def a_measure(self, test_case):
            assert test_case.input == "Question 0?"
            assert test_case.actual_output == "Answer 0"
            assert test_case.expected_output == "Golden 0"
            assert test_case.additional_metadata["question_type"] == "summarization"
            assert test_case.additional_metadata["rubric"] == ["Criterion 0"]
            self.score = 1.0
            self.reason = "rubric ok"

    class FakeKendallMetric:
        def __init__(self):
            self.score = None
            self.reason = None

        async def a_measure(self, test_case):
            self.score = None
            self.reason = "Not applicable (not event_ordering)"

    adapter._metric_factories = {
        "beam_rubric": FakeRubricMetric,
        "kendall_tau": FakeKendallMetric,
    }

    results = await adapter.evaluate_answers([make_answer(0)], ["beam_rubric", "kendall_tau"])

    assert len(results) == 1
    assert results[0]["question"] == "Question 0?"
    assert results[0]["metrics"]["beam_rubric"] == {"score": 1.0, "reason": "rubric ok"}
    assert results[0]["metrics"]["kendall_tau"] == {
        "score": None,
        "reason": "Not applicable (not event_ordering)",
    }


@pytest.mark.asyncio
async def test_beam_eval_adapter_evaluates_answers_concurrently():
    adapter = BeamEvalAdapter(max_concurrent_evaluations=3)

    class ConcurrencyMetric:
        active = 0
        max_active = 0

        def __init__(self):
            self.score = None
            self.reason = None

        async def a_measure(self, test_case):
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            await asyncio.sleep(0.01)
            self.score = 1.0
            self.reason = f"ok {test_case.input}"
            type(self).active -= 1

    adapter._metric_factories = {"beam_rubric": ConcurrencyMetric}

    answers = [make_answer(index) for index in range(6)]
    results = await adapter.evaluate_answers(answers, ["beam_rubric"])

    assert len(results) == 6
    assert ConcurrencyMetric.max_active >= 2
    assert ConcurrencyMetric.max_active <= 3
