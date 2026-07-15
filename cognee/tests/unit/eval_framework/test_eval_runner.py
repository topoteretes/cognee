"""Deterministic, keyless tests for the productized eval runner and the
optional-addon decoupling (issue #3623).

These cover the two chunks that productize the harness:
- the eval/DeepEval optional addon (lazy engine imports + actionable errors), and
- the one-command runner / CLI surface (path resolution, flag mapping, orchestration).

They run under CI with no real API keys and without requiring the ``eval`` extra.
"""

import argparse
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.evaluation.evaluator_adapters import EvaluatorAdapter
from cognee.eval_framework.runner import (
    EvalResult,
    _create_dashboard,
    add_eval_arguments,
    config_from_namespace,
    resolve_run_paths,
    run_eval,
    summarize_result,
)


# --------------------------------------------------------------------------- #
# Optional addon: lazy engine imports
# --------------------------------------------------------------------------- #


def test_importing_evaluator_registry_does_not_import_deepeval():
    """Importing the evaluator registry (or the executor) must not pull in the
    optional ``deepeval`` dependency. Verified in a clean subprocess so it is not
    affected by other tests importing deepeval."""
    code = (
        "import sys;"
        "import cognee.eval_framework.evaluation.evaluator_adapters;"
        "import cognee.eval_framework.evaluation.evaluation_executor;"
        "assert 'deepeval' not in sys.modules, 'deepeval was imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_runner_surface_does_not_import_optional_extras():
    """Importing the one-command runner surface must keep both the optional
    ``plotly`` (dashboard) and ``deepeval`` deps out of the import graph. Checked
    in a clean subprocess so it holds even in CI where those extras ARE installed
    — a bare ``--help`` returncode check would not catch a future eager hoist."""
    code = (
        "import sys;"
        "import cognee.eval_framework.runner;"
        "assert 'plotly' not in sys.modules, 'plotly was imported eagerly';"
        "assert 'deepeval' not in sys.modules, 'deepeval was imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_pipeline_steps_does_not_import_optional_extras():
    """Importing the pipeline step modules the runner chains must not pull in
    any optional eval-extra dependency, so a direct_llm run works without the
    extra. Guards against eager plotly/gdown/deepeval imports sneaking back in
    (e.g. the dead dashboard import run_evaluation_module used to carry)."""
    code = (
        "import sys;"
        "import cognee.eval_framework.corpus_builder.run_corpus_builder;"
        "import cognee.eval_framework.answer_generation.run_question_answering_module;"
        "import cognee.eval_framework.evaluation.run_evaluation_module;"
        "assert 'plotly' not in sys.modules, 'plotly was imported eagerly';"
        "assert 'gdown' not in sys.modules, 'gdown was imported eagerly';"
        "assert 'deepeval' not in sys.modules, 'deepeval was imported eagerly';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_direct_llm_engine_loads_without_extra():
    """The DirectLLM engine resolves without needing the eval extra."""
    adapter_cls = EvaluatorAdapter("DirectLLM").load_adapter_class()
    assert adapter_cls.__name__ == "DirectLLMEvalAdapter"


def test_missing_deepeval_raises_actionable_error():
    """Selecting DeepEval without the extra installed raises an actionable error
    pointing at ``cognee[eval]`` rather than a bare ImportError."""
    with patch(
        "cognee.eval_framework.evaluation.evaluator_adapters.import_module",
        side_effect=ImportError("No module named 'deepeval'"),
    ):
        with pytest.raises(ImportError) as excinfo:
            EvaluatorAdapter.DEEPEVAL.load_adapter_class()

    # The message names the engine and the extra, and chains the original error.
    assert "cognee[eval]" in str(excinfo.value)
    assert "'DeepEval'" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


# --------------------------------------------------------------------------- #
# Runner: path resolution & reproducibility
# --------------------------------------------------------------------------- #


def test_resolve_run_paths_legacy_flat_when_no_results_dir():
    config = EvalConfig(benchmark="HotPotQA", results_dir=None)
    paths = resolve_run_paths(config)
    assert paths["metrics_path"] == "metrics_output.json"


def test_resolve_run_paths_namespaces_by_benchmark_and_engine(tmp_path):
    config = EvalConfig(
        benchmark="HotPotQA",
        evaluation_engine="DirectLLM",
        results_dir=str(tmp_path),
    )
    paths = resolve_run_paths(config)
    for key in ("questions_path", "answers_path", "metrics_path"):
        assert str(tmp_path) in paths[key]
        assert "HotPotQA_DirectLLM" in paths[key]


# --------------------------------------------------------------------------- #
# CLI surface: flag -> EvalConfig mapping
# --------------------------------------------------------------------------- #


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_eval_arguments(parser)
    return parser.parse_args(argv)


def test_config_from_namespace_maps_flags(tmp_path):
    args = _parse(
        [
            "--benchmark",
            "HotPotQA",
            "--engine",
            "direct_llm",
            "--limit",
            "7",
            "--seed",
            "123",
            "--output-dir",
            str(tmp_path),
            "--no-dashboard",
        ]
    )
    config = config_from_namespace(args)

    assert config.benchmark == "HotPotQA"
    assert config.evaluation_engine == "DirectLLM"
    assert config.number_of_samples_in_corpus == 7
    assert config.seed == 123
    assert config.results_dir == str(tmp_path)
    assert config.dashboard is False
    # DirectLLM defaults to correctness-only scoring.
    assert config.evaluation_metrics == ["correctness"]


def test_config_from_namespace_defaults_untouched():
    args = _parse([])
    config = config_from_namespace(args)
    default = EvalConfig()
    assert config.benchmark == default.benchmark
    assert config.evaluation_engine == default.evaluation_engine


# --------------------------------------------------------------------------- #
# Runner orchestration (mocked steps, no LLM/DB)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_eval_orchestrates_pipeline_and_returns_result(tmp_path, monkeypatch):
    """run_eval chains the steps, feeds them namespaced paths, writes a resolved
    config, and returns an EvalResult carrying the aggregate metrics."""
    captured = {}

    async def fake_corpus(params):
        captured["params"] = params
        with open(params["questions_path"], "w", encoding="utf-8") as f:
            json.dump([{"question": "q", "answer": "a"}], f)
        return []

    async def fake_answers(params):
        with open(params["answers_path"], "w", encoding="utf-8") as f:
            json.dump([{"question": "q", "answer": "a", "golden_answer": "a"}], f)
        return []

    async def fake_evaluation(params):
        # Emulate the real step producing the aggregate metrics artifact.
        with open(params["aggregate_metrics_path"], "w", encoding="utf-8") as f:
            json.dump({"correctness": {"mean": 1.0, "ci_lower": 1.0, "ci_upper": 1.0}}, f)
        return []

    monkeypatch.setattr("cognee.eval_framework.runner._corpus_step", fake_corpus)
    monkeypatch.setattr("cognee.eval_framework.runner._answer_step", fake_answers)
    monkeypatch.setattr("cognee.eval_framework.runner._evaluation_step", fake_evaluation)

    config = EvalConfig(
        benchmark="Dummy",
        evaluation_engine="DirectLLM",
        results_dir=str(tmp_path),
        dashboard=False,
    )
    result = await run_eval(config)

    assert isinstance(result, EvalResult)
    assert result.benchmark == "Dummy"
    assert result.engine == "DirectLLM"
    assert result.aggregate_metrics["correctness"]["mean"] == 1.0

    # Steps received namespaced paths and the seed.
    assert "Dummy_DirectLLM" in captured["params"]["metrics_path"]
    assert captured["params"]["seed"] == config.seed

    # Resolved config artifact is written for reproducibility.
    with open(result.config_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["benchmark"] == "Dummy"
    assert saved["seed"] == config.seed

    # Summary is renderable.
    assert any("Dummy" in line for line in summarize_result(result))


@pytest.mark.asyncio
async def test_run_eval_runs_dashboard_step_when_enabled(tmp_path, monkeypatch):
    """With the dashboard enabled, run_eval preflights the dashboard deps, invokes
    the dashboard step, and surfaces its path on the result and in the summary.
    Patching _import_dashboard covers both the fail-fast call and the real
    _create_dashboard plumbing without needing plotly installed."""

    async def fake_noop(params):
        return []

    async def fake_evaluation(params):
        with open(params["aggregate_metrics_path"], "w", encoding="utf-8") as f:
            json.dump({"correctness": {"mean": 1.0}}, f)
        return []

    def fake_create_dashboard(metrics_path, aggregate_metrics_path, output_file, benchmark):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("<html></html>")

    monkeypatch.setattr("cognee.eval_framework.runner._corpus_step", fake_noop)
    monkeypatch.setattr("cognee.eval_framework.runner._answer_step", fake_noop)
    monkeypatch.setattr("cognee.eval_framework.runner._evaluation_step", fake_evaluation)
    monkeypatch.setattr(
        "cognee.eval_framework.runner._import_dashboard", lambda: fake_create_dashboard
    )

    config = EvalConfig(
        benchmark="Dummy",
        evaluation_engine="DirectLLM",
        results_dir=str(tmp_path),
        dashboard=True,
    )
    result = await run_eval(config)

    assert result.dashboard_path is not None
    assert "Dummy_DirectLLM" in result.dashboard_path
    # The real _create_dashboard plumbing routed output_file to the resolved path.
    with open(result.dashboard_path, "r", encoding="utf-8") as f:
        assert f.read() == "<html></html>"
    assert any("Dashboard" in line for line in summarize_result(result))


@pytest.mark.asyncio
async def test_run_eval_fails_fast_when_dashboard_deps_missing(tmp_path, monkeypatch):
    """When the dashboard is enabled but its deps are missing, run_eval raises
    before any pipeline step runs, so no LLM time or money is wasted."""
    calls = []

    async def spy_step(params):
        calls.append("step")
        return []

    monkeypatch.setattr("cognee.eval_framework.runner._corpus_step", spy_step)
    monkeypatch.setattr("cognee.eval_framework.runner._answer_step", spy_step)
    monkeypatch.setattr("cognee.eval_framework.runner._evaluation_step", spy_step)
    monkeypatch.setitem(sys.modules, "cognee.eval_framework.metrics_dashboard", None)

    config = EvalConfig(
        benchmark="Dummy",
        evaluation_engine="DirectLLM",
        results_dir=str(tmp_path),
        dashboard=True,
    )
    with pytest.raises(ImportError) as excinfo:
        await run_eval(config)

    assert calls == []
    assert "--no-dashboard" in str(excinfo.value)


def test_create_dashboard_missing_deps_is_actionable(monkeypatch):
    """When the dashboard deps (plotly) are absent, _create_dashboard raises an
    error pointing at both the eval extra and the --no-dashboard escape hatch."""
    # Setting the module to None makes the in-function import raise ImportError,
    # emulating a missing plotly even when the eval extra is installed in CI.
    monkeypatch.setitem(sys.modules, "cognee.eval_framework.metrics_dashboard", None)
    with pytest.raises(ImportError) as excinfo:
        _create_dashboard(
            {
                "metrics_path": "m.json",
                "aggregate_metrics_path": "a.json",
                "dashboard_path": "d.html",
                "benchmark": "Dummy",
            }
        )
    assert "cognee[eval]" in str(excinfo.value)
    assert "--no-dashboard" in str(excinfo.value)
