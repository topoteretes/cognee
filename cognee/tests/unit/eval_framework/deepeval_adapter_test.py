import pytest
from unittest.mock import patch
from evals.eval_framework.eval_config import EvalConfig
from evals.eval_framework.evaluation.deep_eval_adapter import DeepEvalAdapter


@pytest.fixture
def adapter():
    return DeepEvalAdapter()


@pytest.mark.asyncio
async def test_evaluate_answers_em_f1(adapter):
    answers = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "golden_answer": "4",
        }
    ]
    evaluator_metrics = ["EM", "f1"]

    results = await adapter.evaluate_answers(answers, evaluator_metrics)

    assert len(results) == 1
    assert "metrics" in results[0]
    assert "EM" in results[0]["metrics"]
    assert "f1" in results[0]["metrics"]
    assert results[0]["metrics"]["EM"]["score"] == 1.0
    assert results[0]["metrics"]["f1"]["score"] == 1.0


@pytest.mark.asyncio
async def test_unsupported_metric(adapter):
    answers = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "golden_answer": "4",
        }
    ]
    evaluator_metrics = ["unsupported_metric"]

    with pytest.raises(ValueError, match="Unsupported metric: unsupported_metric"):
        await adapter.evaluate_answers(answers, evaluator_metrics)


@pytest.mark.asyncio
async def test_empty_answers_list(adapter):
    results = await adapter.evaluate_answers([], ["EM", "f1"])
    assert results == []


@pytest.mark.asyncio
async def test_missing_fields_in_answer(adapter):
    answers = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
        }
    ]
    evaluator_metrics = ["EM", "f1"]

    with pytest.raises(KeyError):
        await adapter.evaluate_answers(answers, evaluator_metrics)


@pytest.mark.asyncio
async def test_none_values_in_answers(adapter):
    answers = [
        {
            "question": None,
            "answer": None,
            "golden_answer": None,
        }
    ]
    evaluator_metrics = ["EM", "f1"]

    results = await adapter.evaluate_answers(answers, evaluator_metrics)

    assert len(results) == 1
    assert results[0]["metrics"]["EM"]["score"] == 1.0
    assert results[0]["metrics"]["f1"]["score"] == 1.0


@pytest.mark.asyncio
async def test_g_eval_correctness(adapter):
    with patch.object(
        type(adapter.g_eval_correctness()), "measure", return_value=1.0
    ) as mock_measure:
        g_eval = adapter.g_eval_correctness()
        mock_measure.assert_not_called()

    eval_config = EvalConfig().to_dict()
    assert g_eval.model.model_name == eval_config["deepeval_model"]
