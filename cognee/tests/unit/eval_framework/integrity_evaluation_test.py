import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from cognee.eval_framework.analysis.metrics_calculator import calculate_metrics_statistics
from cognee.eval_framework.evaluation import run_evaluation_module as eval_module
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.tests.unit.eval_framework.integrity_helpers import make_answer, make_metrics_entry

with patch.dict(
    sys.modules,
    {
        "deepeval": MagicMock(),
        "deepeval.metrics": MagicMock(),
        "deepeval.test_case": MagicMock(),
        "cognee.eval_framework.evaluation.metrics.context_coverage": MagicMock(),
    },
):
    from cognee.eval_framework.evaluation.deep_eval_adapter import DeepEvalAdapter


class StubAdapter:
    """Scores every answer 1.0 and records which answers it was given."""

    def __init__(self):
        self.received = []

    async def evaluate_answers(self, answers, evaluator_metrics):
        self.received.extend(a["question"] for a in answers)
        return [{**a, "metrics": {"correctness": {"score": 1.0, "reason": "ok"}}} for a in answers]


@pytest.mark.asyncio
async def test_failed_answers_are_not_sent_to_the_evaluator(tmp_path, monkeypatch):
    answers = [make_answer(0), make_answer(1, "llm_error"), make_answer(2)]
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers))

    async def no_db(_metrics):
        return None

    monkeypatch.setattr(eval_module, "create_and_insert_metrics_table", no_db)
    adapter = StubAdapter()
    params = {
        "answers_path": str(answers_path),
        "metrics_path": str(tmp_path / "metrics.json"),
        "aggregate_metrics_path": str(tmp_path / "aggregate.json"),
        "evaluation_engine": adapter,
        "evaluating_contexts": False,
        "evaluation_metrics": ["correctness"],
        "evaluating_answers": True,
        "calculate_metrics": True,
    }
    metrics = await eval_module.run_evaluation(params)

    assert adapter.received == ["q0", "q2"]
    assert [m["question"] for m in metrics] == ["q0", "q1", "q2"]  # order and alignment kept
    assert metrics[1]["metrics"] == {} and metrics[1]["status"] == "llm_error"
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    # The failed answer counts as unscored: 2 of 3 questions, not 2 of 2.
    assert aggregate["correctness"]["coverage"]["scored"] == 2
    assert aggregate["correctness"]["coverage"]["total"] == 3
    assert aggregate["correctness"]["answers"] == {"total": 3, "ok": 2, "empty": 0, "failed": 1}


@pytest.mark.asyncio
async def test_evaluation_rejects_a_truncated_answers_file(tmp_path):
    answers_path = tmp_path / "answers.json"
    answers_path.write_text('[{"question": "q0", "ans')
    with pytest.raises(RunIntegrityError, match="truncated or corrupt"):
        await eval_module.execute_evaluation({"answers_path": str(answers_path)})


def test_none_metric_is_excluded_from_the_mean_but_reported_in_coverage(tmp_path):
    entries = [make_metrics_entry(0, 1.0), make_metrics_entry(1, 0.0), make_metrics_entry(2, None)]
    metrics_path, aggregate_path = tmp_path / "m.json", tmp_path / "agg.json"
    metrics_path.write_text(json.dumps(entries))

    calculate_metrics_statistics(str(metrics_path), str(aggregate_path))

    aggregate = json.loads(aggregate_path.read_text())["correctness"]
    assert aggregate["mean"] == 0.5
    assert aggregate["answers"]["failed"] == 0
    assert aggregate["coverage"] == {
        "scored": 2,
        "total": 3,
        "unscored": 1,
        "unscored_fraction": pytest.approx(1 / 3),
    }


def test_deepeval_adapter_reports_why_a_metric_has_no_score():
    adapter = DeepEvalAdapter()
    adapter.n_retries = 2
    failing = MagicMock()
    failing.measure.side_effect = ConnectionError("judge unreachable")
    adapter.g_eval_metrics["correctness"] = failing

    with patch("cognee.eval_framework.evaluation.deep_eval_adapter.time.sleep"):
        result = adapter._calculate_metric("correctness", MagicMock())

    assert result == {"score": None, "reason": None, "error": "ConnectionError: judge unreachable"}
    assert failing.measure.call_count == 2  # retries exhausted, then the cause is reported


def test_half_failed_run_reports_50_of_100(tmp_path):
    entries = [make_metrics_entry(i, 1.0) for i in range(50)] + [
        {**make_answer(i, "llm_error"), "metrics": {}} for i in range(50, 100)
    ]
    metrics_path, aggregate_path = tmp_path / "m.json", tmp_path / "agg.json"
    metrics_path.write_text(json.dumps(entries))

    calculate_metrics_statistics(str(metrics_path), str(aggregate_path))

    aggregate = json.loads(aggregate_path.read_text())["correctness"]
    assert (aggregate["coverage"]["scored"], aggregate["coverage"]["total"]) == (50, 100)
    assert aggregate["answers"] == {"total": 100, "ok": 50, "empty": 0, "failed": 50}


@pytest.mark.asyncio
async def test_evaluator_returning_the_wrong_count_raises(tmp_path):
    class Dropping:
        async def evaluate_answers(self, answers, evaluator_metrics):
            return answers[:-1]

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps([make_answer(0), make_answer(1)]))
    params = {
        "answers_path": str(answers_path),
        "metrics_path": str(tmp_path / "metrics.json"),
        "evaluation_engine": Dropping(),
        "evaluating_contexts": False,
        "evaluation_metrics": ["correctness"],
    }
    with pytest.raises(RunIntegrityError, match="returned 1 results for 2 answers"):
        await eval_module.execute_evaluation(params)
    assert not (tmp_path / "metrics.json").exists()
