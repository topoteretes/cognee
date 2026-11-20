import argparse
import json

import pytest

from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.manifest import (
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    StageStatus,
    enforce_failure_rate,
    summarize_answers,
)
from cognee.eval_framework.runner import (
    add_eval_arguments,
    config_from_namespace,
    run_eval,
    summarize_result,
)
from cognee.tests.unit.eval_framework.integrity_helpers import (
    make_answer,
    make_metrics_entry,
    make_questions,
)


def test_manifest_round_trip_and_atomic_save(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    manifest = RunManifest.new(EvalConfig().to_dict())
    manifest.finish("answers", StageStatus.COMPLETED, **summarize_answers([make_answer(0)]))
    manifest.save(path)
    loaded = RunManifest.load(path)
    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert loaded.stages["answers"].ok == 1
    assert [p.name for p in tmp_path.iterdir()] == ["run_manifest.json"]


def test_manifest_load_rejects_corrupt_wrong_version_and_malformed(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text('{"schema_ver')
    with pytest.raises(RunIntegrityError, match="corrupt"):
        RunManifest.load(str(path))
    path.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(RunIntegrityError, match="schema version 99"):
        RunManifest.load(str(path))
    path.write_text(json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION}))
    with pytest.raises(RunIntegrityError, match="malformed"):
        RunManifest.load(str(path))


def test_resume_refuses_a_changed_config():
    manifest = RunManifest.new({**EvalConfig().to_dict(), "seed": 1})
    with pytest.raises(RunIntegrityError, match="seed: 1 -> 2"):
        manifest.require_same_config({**EvalConfig().to_dict(), "seed": 2})


def test_summarize_answers_counts_and_lists_failures():
    summary = summarize_answers(
        [make_answer(0), make_answer(1, "empty"), make_answer(2, "retrieval_error")]
    )
    assert (summary["total"], summary["ok"], summary["empty"], summary["failed"]) == (3, 1, 1, 1)
    assert summary["failures"][0].status.value == "retrieval_error"


def test_begin_resets_the_stage_record():
    manifest = RunManifest.new(EvalConfig().to_dict())
    manifest.finish(
        "answers", StageStatus.FAILED, error="boom", **summarize_answers([make_answer(0)])
    )
    manifest.begin("answers")
    record = manifest.stages["answers"]
    assert record.status is StageStatus.RUNNING
    assert (record.error, record.total, record.ok, record.failures) == (None, None, None, [])


def test_fingerprint_covers_answer_inputs(tmp_path):
    from cognee.eval_framework.integrity.fingerprint import answers_fingerprint

    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(make_questions(2)))
    params = EvalConfig().to_dict()
    llm = {"model": "m"}

    def fp(**changes):
        args = {
            "params": params,
            "questions_path": str(questions),
            "system_prompt": "p.txt",
            "system_prompt_text": "be brief",
            "top_k": 5,
            "llm": llm,
        }
        return answers_fingerprint(**{**args, **changes})

    base = fp()
    assert fp() == base
    assert fp(top_k=6) != base
    assert fp(system_prompt_text="be long") != base
    assert fp(system_prompt="q.txt") != base
    assert fp(llm={"model": "other"}) != base
    assert fp(params={**params, "qa_engine": "cognee_completion"}) != base
    # Output locations and evaluation-only settings do not invalidate answers.
    assert fp(params={**params, "dashboard": False, "evaluation_metrics": ["f1"]}) == base
    questions.write_text(json.dumps(make_questions(3)))
    assert fp() != base


def test_enforce_failure_rate():
    enforce_failure_rate(5, 10, None, "x")  # no threshold: never fatal
    enforce_failure_rate(0, 0, 0.0, "x")
    enforce_failure_rate(1, 10, 0.1, "x")  # count == limit is allowed
    enforce_failure_rate(10, 10, 1.0, "x")
    with pytest.raises(RunIntegrityError, match="2/10"):
        enforce_failure_rate(2, 10, 0.1, "x")
    with pytest.raises(RunIntegrityError):
        enforce_failure_rate(1, 10, 0.0, "x")


def test_config_rejects_out_of_range_thresholds():
    with pytest.raises(ValueError):
        EvalConfig(max_answer_failure_rate=1.5)


def test_cli_flags_map_onto_config():
    parser = argparse.ArgumentParser()
    add_eval_arguments(parser)
    args = parser.parse_args(
        [
            "--resume",
            "--max-answer-failure-rate",
            "0.1",
            "--max-unscored-rate",
            "0",
            "--strict-integrity",
        ]
    )
    config = config_from_namespace(args)
    assert (config.resume, config.max_answer_failure_rate, config.max_unscored_rate) == (
        True,
        0.1,
        0.0,
    )
    assert config.strict_integrity is True
    defaults = config_from_namespace(parser.parse_args([]))
    assert defaults.resume is False and defaults.max_answer_failure_rate is None


# --- runner integration (steps stubbed, no LLM or DB) -------------------------------------- #


def _install_steps(monkeypatch, *, answers, metrics, calls=None):
    async def corpus(params):
        if calls is not None:
            calls.append("corpus")
        with open(params["questions_path"], "w") as f:
            json.dump(make_questions(len(answers)), f)

    async def answer_step(params):
        if calls is not None:
            calls.append("answers")
        with open(params["answers_path"], "w") as f:
            json.dump(answers, f)

    async def evaluation(params):
        if calls is not None:
            calls.append("evaluation")
        with open(params["metrics_path"], "w") as f:
            json.dump(metrics, f)
        aggregate = {
            "correctness": {"mean": 0.5, "coverage": {"scored": 1, "total": 2, "unscored": 1}}
        }
        with open(params["aggregate_metrics_path"], "w") as f:
            json.dump(aggregate, f)

    monkeypatch.setattr("cognee.eval_framework.runner._corpus_step", corpus)
    monkeypatch.setattr("cognee.eval_framework.runner._answer_step", answer_step)
    monkeypatch.setattr("cognee.eval_framework.runner._evaluation_step", evaluation)


def _config(tmp_path, **overrides):
    return EvalConfig(
        benchmark="Dummy",
        evaluation_engine="DirectLLM",
        results_dir=str(tmp_path),
        dashboard=False,
        **overrides,
    )


@pytest.mark.asyncio
async def test_run_writes_manifest_with_counts_failures_and_coverage(tmp_path, monkeypatch):
    answers = [make_answer(0), make_answer(1, "llm_error")]
    metrics = [make_metrics_entry(0, None), {**make_answer(1, "llm_error"), "metrics": {}}]
    _install_steps(monkeypatch, answers=answers, metrics=metrics)

    result = await run_eval(_config(tmp_path))

    manifest = RunManifest.load(result.manifest_path)
    assert all(
        manifest.stages[s].status is StageStatus.COMPLETED
        for s in ("corpus", "answers", "evaluation")
    )
    assert manifest.stages["dashboard"].status is StageStatus.SKIPPED
    assert (manifest.stages["answers"].ok, manifest.stages["answers"].failed) == (1, 1)
    coverage = manifest.stages["evaluation"].coverage["correctness"]
    assert (coverage.scored, coverage.total) == (0, 2)  # the failed answer is unscored too
    assert manifest.findings == []
    assert manifest.answers_fingerprint is None or isinstance(manifest.answers_fingerprint, str)
    assert any("1 ok, 0 empty, 1 failed" in line for line in summarize_result(result))


@pytest.mark.asyncio
async def test_failing_stage_is_recorded_and_error_propagates(tmp_path, monkeypatch):
    async def corpus(params):
        raise RuntimeError("corpus exploded")

    monkeypatch.setattr("cognee.eval_framework.runner._corpus_step", corpus)
    with pytest.raises(RuntimeError, match="corpus exploded"):
        await run_eval(_config(tmp_path))
    manifest = RunManifest.load(str(tmp_path / "Dummy_DirectLLM" / "run_manifest.json"))
    assert manifest.stages["corpus"].status is StageStatus.FAILED
    assert manifest.stages["corpus"].error == "RuntimeError: corpus exploded"
    assert manifest.stages["answers"].status is StageStatus.PENDING


@pytest.mark.asyncio
async def test_answer_failure_threshold_is_opt_in(tmp_path, monkeypatch):
    answers = [make_answer(0), make_answer(1, "retrieval_error")]
    metrics = [make_metrics_entry(0, 1.0), {**make_answer(1, "retrieval_error"), "metrics": {}}]
    _install_steps(monkeypatch, answers=answers, metrics=metrics)

    await run_eval(_config(tmp_path))  # no threshold: failures reported, run succeeds
    with pytest.raises(RunIntegrityError, match="1/2 questions failed to answer"):
        await run_eval(_config(tmp_path, max_answer_failure_rate=0.25))
    manifest = RunManifest.load(str(tmp_path / "Dummy_DirectLLM" / "run_manifest.json"))
    assert manifest.stages["answers"].status is StageStatus.FAILED


@pytest.mark.asyncio
async def test_unscored_threshold_names_the_metric(tmp_path, monkeypatch):
    answers = [make_answer(0), make_answer(1)]
    metrics = [make_metrics_entry(0, 1.0), make_metrics_entry(1, None)]
    _install_steps(monkeypatch, answers=answers, metrics=metrics)
    with pytest.raises(RunIntegrityError, match="metric 'correctness'"):
        await run_eval(_config(tmp_path, max_unscored_rate=0.0))


@pytest.mark.asyncio
async def test_strict_integrity_fails_on_misaligned_files_only_when_enabled(tmp_path, monkeypatch):
    answers = [make_answer(0), make_answer(1)]
    metrics = [make_metrics_entry(0, 1.0)]  # truncated metrics file: question 1 missing
    _install_steps(monkeypatch, answers=answers, metrics=metrics)

    result = await run_eval(_config(tmp_path))
    assert [f.code for f in result.manifest.findings] == ["missing_metrics"]
    with pytest.raises(RunIntegrityError, match="missing_metrics"):
        await run_eval(_config(tmp_path, strict_integrity=True))


@pytest.mark.asyncio
async def test_resume_skips_a_completed_corpus_and_requires_matching_config(tmp_path, monkeypatch):
    calls = []
    answers = [make_answer(0)]
    _install_steps(monkeypatch, answers=answers, metrics=[make_metrics_entry(0, 1.0)], calls=calls)

    await run_eval(_config(tmp_path))
    assert calls == ["corpus", "answers", "evaluation"]

    calls.clear()
    result = await run_eval(_config(tmp_path, resume=True))
    assert calls == ["answers", "evaluation"]
    assert result.manifest.stages["corpus"].note == "reused the corpus built by the resumed run"

    with pytest.raises(RunIntegrityError, match="config changed"):
        await run_eval(_config(tmp_path, resume=True, seed=7))


@pytest.mark.asyncio
async def test_resume_without_a_manifest_raises(tmp_path, monkeypatch):
    _install_steps(monkeypatch, answers=[make_answer(0)], metrics=[make_metrics_entry(0, 1.0)])
    with pytest.raises(RunIntegrityError, match="no run manifest"):
        await run_eval(_config(tmp_path, resume=True))
    assert not (tmp_path / "Dummy_DirectLLM" / "run_manifest.json").exists()


@pytest.mark.asyncio
async def test_resume_refuses_changed_top_k_or_system_prompt(tmp_path, monkeypatch):
    _install_steps(monkeypatch, answers=[make_answer(0)], metrics=[make_metrics_entry(0, 1.0)])
    from cognee.eval_framework import runner
    from cognee.eval_framework.integrity.fingerprint import answers_fingerprint

    settings = {"top_k": 5, "prompt": "be brief"}

    def fingerprint(params):
        return answers_fingerprint(
            params, params["questions_path"], "p.txt", settings["prompt"], settings["top_k"], {}
        )

    monkeypatch.setattr(runner, "_answers_fingerprint", fingerprint)
    await run_eval(_config(tmp_path))
    await run_eval(_config(tmp_path, resume=True))  # unchanged: allowed

    settings["top_k"] = 10
    with pytest.raises(RunIntegrityError, match="top_k|differ"):
        await run_eval(_config(tmp_path, resume=True))
    settings.update(top_k=5, prompt="be long")
    with pytest.raises(RunIntegrityError, match="differ"):
        await run_eval(_config(tmp_path, resume=True))


@pytest.mark.asyncio
async def test_second_runner_on_the_same_run_dir_is_refused(tmp_path, monkeypatch):
    from cognee.eval_framework.integrity.io import run_lock

    _install_steps(monkeypatch, answers=[make_answer(0)], metrics=[make_metrics_entry(0, 1.0)])
    (tmp_path / "Dummy_DirectLLM").mkdir()
    with (
        run_lock(str(tmp_path / "Dummy_DirectLLM")),
        pytest.raises(RunIntegrityError, match="locked"),
    ):
        await run_eval(_config(tmp_path))


@pytest.mark.asyncio
async def test_stage_without_output_is_tolerated_unless_strict(tmp_path, monkeypatch):
    async def noop(params):
        return []

    for step in ("_corpus_step", "_answer_step", "_evaluation_step"):
        monkeypatch.setattr(f"cognee.eval_framework.runner.{step}", noop)
    result = await run_eval(_config(tmp_path))
    assert result.manifest.stages["answers"].status is StageStatus.COMPLETED
    with pytest.raises(RunIntegrityError, match="produced no file"):
        await run_eval(_config(tmp_path, strict_integrity=True))
