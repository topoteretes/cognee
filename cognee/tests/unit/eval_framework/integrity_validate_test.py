import pytest

from cognee.eval_framework.integrity.coverage import compute_metric_coverage
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.io import atomic_write_json, load_json_list
from cognee.eval_framework.integrity.status import answer_status, question_id
from cognee.eval_framework.integrity.validate import Severity, validate_run
from cognee.tests.unit.eval_framework.integrity_helpers import (
    make_answer,
    make_metrics_entry,
    make_questions,
)


def codes(findings):
    return sorted(f.code for f in findings)


def test_consistent_run_has_no_findings():
    findings = validate_run(
        questions=make_questions(3),
        answers=[make_answer(i) for i in range(3)],
        metrics=[make_metrics_entry(i, 1.0) for i in range(3)],
    )
    assert findings == []


def test_missing_and_orphaned_entries():
    findings = validate_run(
        questions=make_questions(3),
        answers=[make_answer(0), make_answer(1), make_answer(9)],
    )
    assert codes(findings) == ["missing_answers", "orphan_answers"]
    missing = next(f for f in findings if f.code == "missing_answers")
    assert missing.question_id == question_id("q2", "golden 2")
    assert all(f.severity is Severity.ERROR for f in findings)


def test_duplicate_answers_are_reported_once_as_duplicate_and_once_as_orphan():
    findings = validate_run(
        questions=make_questions(2),
        answers=[make_answer(0), make_answer(0), make_answer(1)],
    )
    assert codes(findings) == ["duplicate_answers", "orphan_answers"]


def test_empty_golden_answer():
    questions = [{"question": "q", "answer": "  "}]
    answers = [{**make_answer(0), "question": "q", "golden_answer": "  "}]
    findings = validate_run(questions=questions, answers=answers)
    assert codes(findings) == ["empty_golden_answer_answers", "empty_golden_answer_questions"]


def test_only_given_artifacts_are_compared():
    assert validate_run(answers=[make_answer(0)]) == []


def test_failed_answers_stay_visible_in_metrics():
    # A failed answer has a metrics entry with no scores, so alignment holds.
    findings = validate_run(
        answers=[make_answer(0, "llm_error")],
        metrics=[{**make_answer(0, "llm_error"), "metrics": {}}],
    )
    assert findings == []


def test_coverage_counts_none_and_failed_answers_as_unscored():
    entries = [
        make_metrics_entry(0, 1.0),
        make_metrics_entry(1, None),
        {**make_answer(2, "retrieval_error"), "metrics": {}},
    ]
    coverage = compute_metric_coverage(entries)["correctness"]
    assert (coverage.scored, coverage.total, coverage.unscored) == (1, 3, 2)
    assert coverage.unscored_fraction == pytest.approx(2 / 3)


def test_unknown_status_raises():
    with pytest.raises(RunIntegrityError, match="Unknown answer status"):
        answer_status({"question": "q", "status": "weird"})


def test_legacy_record_without_status_is_ok():
    assert answer_status({"question": "q", "answer": "a"}).value == "ok"


def test_legacy_error_text_without_status_is_a_failure():
    assert answer_status({"question": "q", "answer": "ERROR: boom"}).failed
    # An explicit status always wins over the text.
    assert not answer_status({"question": "q", "answer": "ERROR: x", "status": "ok"}).failed


def test_locomo_aggregate_uses_the_single_status_policy(tmp_path):
    from cognee.eval_framework.locomo import aggregate

    qa = tmp_path / "qa"
    qa.mkdir()
    rows = [
        {"question": "a", "answer": "ERROR: old", "metrics": {"llm_judge": {"score": 0.0}}},
        {"question": "b", "answer": "", "status": "llm_error", "metrics": {}},
        {"question": "c", "answer": "fine", "metrics": {"llm_judge": {"score": 1.0}}},
    ]
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_run0.json"), rows)
    block = aggregate.aggregate_run_dir(tmp_path)["retrievers"]["hybrid"]
    assert block["answer_errors"] == 2

    rows[2]["status"] = "weird"
    aggregate.write_json(str(qa / "locomo_metrics_conv0_hybrid_run0.json"), rows)
    with pytest.raises(RunIntegrityError, match="Unknown answer status"):
        aggregate.aggregate_run_dir(tmp_path)


def test_load_json_list_rejects_truncated_missing_and_wrong_shape(tmp_path):
    truncated = tmp_path / "answers.json"
    truncated.write_text('[{"question": "q", "answ')
    with pytest.raises(RunIntegrityError, match="truncated or corrupt"):
        load_json_list(str(truncated), "answers")
    with pytest.raises(RunIntegrityError, match="not found"):
        load_json_list(str(tmp_path / "nope.json"), "answers")
    shaped = tmp_path / "dict.json"
    shaped.write_text("{}")
    with pytest.raises(RunIntegrityError, match="list of objects"):
        load_json_list(str(shaped), "answers")


def test_run_lock_refuses_a_second_runner_and_never_steals(tmp_path):
    from cognee.eval_framework.integrity.io import LOCK_FILENAME, run_lock

    with run_lock(str(tmp_path)):
        with pytest.raises(RunIntegrityError, match="locked"), run_lock(str(tmp_path)):
            pass
        assert (tmp_path / LOCK_FILENAME).exists()  # the failed attempt did not remove it
    assert not (tmp_path / LOCK_FILENAME).exists()

    (tmp_path / LOCK_FILENAME).write_text("99999")  # stale lock from a killed run
    with pytest.raises(RunIntegrityError, match="delete the file"), run_lock(str(tmp_path)):
        pass


def test_atomic_write_leaves_previous_file_and_no_temp_on_failure(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json(str(target), [{"a": 1}])

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(str(target), [Unserializable()])
    assert load_json_list(str(target), "out") == [{"a": 1}]
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
