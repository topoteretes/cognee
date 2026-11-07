"""One-command runner for the cognee evaluation harness.

This module puts a thin, testable surface in front of the existing eval pipeline
(corpus -> answers -> evaluation -> dashboard). It exposes:

- ``run_eval(config) -> EvalResult``: a programmatic entry point that runs the
  full pipeline for a single deterministic config and returns the produced
  artifact paths plus aggregate metrics (instead of relying on side-effect files).
- ``add_eval_arguments`` / ``config_from_namespace``: argparse helpers shared by
  the ``cognee eval`` CLI command and ``python -m cognee.eval_framework`` so both
  map the same flags onto :class:`EvalConfig`.

No Modal/Docker is required on this path; ``modal_run_eval.py`` remains a separate
opt-in for distributed runs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cognee.eval_framework.eval_config import EvalConfig
from cognee.eval_framework.integrity.coverage import compute_metric_coverage
from cognee.eval_framework.integrity.errors import RunIntegrityError
from cognee.eval_framework.integrity.io import load_json_list, run_lock
from cognee.eval_framework.integrity.manifest import (
    RunManifest,
    StageStatus,
    enforce_failure_rate,
    manifest_path_for,
    summarize_answers,
)
from cognee.eval_framework.integrity.validate import Severity, validate_run
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Artifact path keys that get namespaced by run when ``results_dir`` is set.
ARTIFACT_KEYS = (
    "questions_path",
    "answers_path",
    "metrics_path",
    "aggregate_metrics_path",
    "dashboard_path",
)

# Engines available on the CLI, mapped to the EvalConfig engine names.
ENGINE_CHOICES = {
    "deepeval": "DeepEval",
    "direct_llm": "DirectLLM",
}


@dataclass
class EvalResult:
    """Structured result of a single evaluation run.

    Returning this from :func:`run_eval` makes the harness callable and assertable
    from tests and other code, rather than depending on files left in the cwd.
    """

    benchmark: str
    engine: str
    config: dict[str, Any]
    config_path: str
    questions_path: str
    answers_path: str
    metrics_path: str
    aggregate_metrics_path: str
    dashboard_path: str | None = None
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    manifest: RunManifest | None = None


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_") or "run"


def resolve_run_paths(config: EvalConfig) -> dict[str, str]:
    """Resolve artifact paths for a run.

    When ``results_dir`` is set, artifacts are namespaced under
    ``<results_dir>/<benchmark>_<engine>/`` so runs with different benchmarks or
    engines are comparable instead of clobbering each other. When it is unset the
    legacy flat filenames (relative to the cwd) are preserved for backward
    compatibility with existing scripts.
    """
    params = config.to_dict()
    if not config.results_dir:
        return {key: params[key] for key in ARTIFACT_KEYS}

    run_id = f"{_slugify(config.benchmark)}_{_slugify(config.evaluation_engine)}"
    run_dir = os.path.join(config.results_dir, run_id)
    resolved = {}
    for key in ARTIFACT_KEYS:
        joined = os.path.join(run_dir, params[key])
        # Absolute overrides discard run_dir entirely and ".." components walk
        # out of it; make the escape from the run directory visible, not silent.
        try:
            abs_run_dir = os.path.abspath(run_dir)
            escapes = os.path.commonpath([abs_run_dir, os.path.abspath(joined)]) != abs_run_dir
        except ValueError:  # e.g. different drives on Windows — an escape as well
            escapes = True
        if escapes:
            logger.warning("%s escapes the run directory %s: %s", key, run_dir, joined)
        resolved[key] = joined
    return resolved


def _load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Thin lazy wrappers around the pipeline steps. Importing them here (rather than
# at module top) keeps ``import cognee.eval_framework.runner`` - and therefore
# ``cognee eval --help`` - free of the optional ``eval`` extra. They are also the
# seams the tests patch to exercise orchestration without an LLM or database.
async def _corpus_step(params: dict[str, Any]) -> Any:
    from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder

    return await run_corpus_builder(params)


async def _answer_step(params: dict[str, Any]) -> Any:
    from cognee.eval_framework.answer_generation.run_question_answering_module import (
        run_question_answering,
    )

    return await run_question_answering(params)


def _answers_fingerprint(params: dict[str, Any]) -> str | None:
    """Fingerprint of what the answers stage is about to answer; None before a questions file
    exists (the stage itself fails then)."""
    if not os.path.exists(params["questions_path"]):
        return None
    from cognee.eval_framework.answer_generation.run_question_answering_module import (
        answers_fingerprint_for,
    )

    return answers_fingerprint_for(params)


async def _evaluation_step(params: dict[str, Any]) -> Any:
    from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

    return await run_evaluation(params)


def _import_dashboard() -> Callable[..., str]:
    """Import the dashboard generator lazily (it pulls in plotly), raising an
    actionable error when the optional deps are missing. Called before the
    pipeline runs so a costly LLM run is never wasted on a missing extra."""
    try:
        from cognee.eval_framework.metrics_dashboard import create_dashboard
    except ImportError as error:
        raise ImportError(
            "Dashboard generation requires optional dependencies. Install them with: "
            'pip install "cognee[eval]" (or pass --no-dashboard to skip it).'
        ) from error
    return create_dashboard


def _create_dashboard(params: dict[str, Any]) -> str:
    """Generate the HTML dashboard."""
    create_dashboard = _import_dashboard()
    create_dashboard(
        metrics_path=params["metrics_path"],
        aggregate_metrics_path=params["aggregate_metrics_path"],
        output_file=params["dashboard_path"],
        benchmark=params["benchmark"],
    )
    return params["dashboard_path"]


async def _run_stage(
    manifest: RunManifest, manifest_path: str, stage: str, step: Callable[[], Any]
) -> Any:
    """Run one pipeline step (sync or async), recording it in the manifest before and after.

    A failing step is recorded as failed and the error propagates unchanged.
    """
    manifest.begin(stage)
    manifest.save(manifest_path)
    try:
        result = step()
        if inspect.isawaitable(result):
            result = await result
    except Exception as error:
        manifest.finish(stage, StageStatus.FAILED, error=f"{type(error).__name__}: {error}")
        manifest.save(manifest_path)
        raise
    return result


def _artifact_or_none(params: dict[str, Any], key: str, name: str) -> list[dict] | None:
    """The stage's output file. Missing is tolerated (a stage a caller replaced may write
    nothing) unless ``strict_integrity`` is on; a present but unreadable file always raises."""
    if os.path.exists(params[key]):
        return load_json_list(params[key], name)
    if params.get("strict_integrity"):
        raise RunIntegrityError(f"The {name} stage produced no file at {params[key]}")
    logger.warning("No %s file at %s; it is not recorded in the manifest", name, params[key])
    return None


def _record_answers(manifest: RunManifest, params: dict[str, Any]) -> list[dict[str, Any]] | None:
    answers = _artifact_or_none(params, "answers_path", "answers")
    if answers is None:
        manifest.finish("answers", StageStatus.COMPLETED)
        return None
    summary = summarize_answers(answers)
    manifest.finish("answers", StageStatus.COMPLETED, **summary)
    if summary["failed"]:
        logger.warning(
            "%d of %d questions could not be answered; they are not scored",
            summary["failed"],
            len(answers),
        )
    return answers


def _record_evaluation(
    manifest: RunManifest, params: dict[str, Any]
) -> list[dict[str, Any]] | None:
    metrics = _artifact_or_none(params, "metrics_path", "metrics")
    if metrics is None:
        manifest.finish("evaluation", StageStatus.COMPLETED)
        return None
    coverage = compute_metric_coverage(metrics)
    manifest.finish("evaluation", StageStatus.COMPLETED, total=len(metrics), coverage=coverage)
    return metrics


def _enforce_thresholds(
    manifest: RunManifest, manifest_path: str, stage: str, params: dict[str, Any]
) -> None:
    """Strict mode: raise, after recording it, when a stage breaks a user-set threshold."""
    record = manifest.stages[stage]
    try:
        if stage == "answers":
            enforce_failure_rate(
                record.failed or 0,
                record.total or 0,
                params.get("max_answer_failure_rate"),
                "questions failed to answer",
            )
        else:
            for metric, coverage in (record.coverage or {}).items():
                enforce_failure_rate(
                    coverage.unscored,
                    coverage.total,
                    params.get("max_unscored_rate"),
                    f"questions left unscored by metric '{metric}'",
                )
    except RunIntegrityError as error:
        manifest.finish(stage, StageStatus.FAILED, error=str(error))
        manifest.save(manifest_path)
        raise


async def run_eval(config: EvalConfig | None = None) -> EvalResult:
    """Run the full eval pipeline for a single config and return an EvalResult.

    Chains corpus -> answers -> evaluation -> (optional) dashboard, writing the
    resolved config next to the artifacts for reproducibility, and a run manifest
    (``run_manifest.json``) recording each stage's status, counts and failures.

    With ``resume`` the run reuses the built corpus and every answer that was already
    answered ok; the manifest must come from a run with the same result-determining config.
    """
    config = config or EvalConfig()
    params = config.to_dict()

    resolved = resolve_run_paths(config)
    params.update(resolved)

    if params.get("dashboard"):
        # Fail fast if the dashboard deps are missing, before any pipeline work.
        _import_dashboard()

    run_dir = os.path.dirname(resolved["metrics_path"])
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    with run_lock(run_dir):
        return await _run_pipeline(config, params, resolved, run_dir)


async def _run_pipeline(
    config: EvalConfig, params: dict[str, Any], resolved: dict[str, str], run_dir: str
) -> EvalResult:
    manifest_path = manifest_path_for(run_dir)
    # The manifest and its config check are the single gate for reusing anything from an earlier
    # run (corpus and answer journal). Resuming without them is an error, not a fresh start.
    resumed = config.resume
    if resumed:
        if not os.path.exists(manifest_path):
            raise RunIntegrityError(
                f"Cannot resume: no run manifest at {manifest_path}. Run without resume first."
            )
        manifest = RunManifest.load(manifest_path)
        manifest.require_same_config(params)
    else:
        manifest = RunManifest.new(params)
    manifest.save(manifest_path)

    logger.info(
        "Running eval: benchmark=%s engine=%s samples=%s seed=%s",
        config.benchmark,
        config.evaluation_engine,
        config.number_of_samples_in_corpus,
        config.seed,
    )

    if not params.get("building_corpus_from_scratch"):
        manifest.skip("corpus", "building_corpus_from_scratch is False")
    elif (
        resumed
        and manifest.stages["corpus"].status is StageStatus.COMPLETED
        and os.path.exists(resolved["questions_path"])
    ):
        manifest.stages["corpus"].note = "reused the corpus built by the resumed run"
    else:
        await _run_stage(manifest, manifest_path, "corpus", lambda: _corpus_step(params))
        manifest.finish("corpus", StageStatus.COMPLETED)

    answers = None
    if params.get("answering_questions"):
        fingerprint = _answers_fingerprint(params)
        if resumed and manifest.answers_fingerprint not in (None, fingerprint):
            raise RunIntegrityError(
                "Cannot resume: the questions, prompts, top_k, model or answering config differ "
                "from the interrupted run, so its answers do not apply to this one."
            )
        # Reuse the journal only when a previous run recorded the same fingerprint.
        params["resume_verified"] = (
            resumed and fingerprint is not None and (manifest.answers_fingerprint == fingerprint)
        )
        manifest.answers_fingerprint = fingerprint
        await _run_stage(manifest, manifest_path, "answers", lambda: _answer_step(params))
        answers = _record_answers(manifest, params)
        _enforce_thresholds(manifest, manifest_path, "answers", params)
        manifest.save(manifest_path)
    else:
        manifest.skip("answers", "answering_questions is False")

    metrics = None
    if params.get("evaluating_answers"):
        await _run_stage(manifest, manifest_path, "evaluation", lambda: _evaluation_step(params))
        metrics = _record_evaluation(manifest, params)
        _enforce_thresholds(manifest, manifest_path, "evaluation", params)
        manifest.save(manifest_path)
    else:
        manifest.skip("evaluation", "evaluating_answers is False")

    # Artifacts that do not exist (a skipped stage, a corpus supplied elsewhere) are not compared.
    questions = (
        load_json_list(resolved["questions_path"], "questions")
        if os.path.exists(resolved["questions_path"])
        else None
    )
    manifest.findings = validate_run(questions=questions, answers=answers, metrics=metrics)
    for finding in manifest.findings:
        logger.warning(
            "Integrity %s [%s]: %s", finding.severity.value, finding.code, finding.message
        )
    errors = [f for f in manifest.findings if f.severity is Severity.ERROR]
    if params.get("strict_integrity") and errors:
        manifest.save(manifest_path)
        raise RunIntegrityError(
            f"{len(errors)} integrity errors, first: [{errors[0].code}] {errors[0].message}; "
            f"see {manifest_path}"
        )

    dashboard_path = None
    if params.get("dashboard"):
        logger.info("Generating dashboard...")
        await _run_stage(manifest, manifest_path, "dashboard", lambda: _create_dashboard(params))
        manifest.finish("dashboard", StageStatus.COMPLETED)
        dashboard_path = params["dashboard_path"]
    else:
        manifest.skip("dashboard", "dashboard is False")
    manifest.save(manifest_path)

    aggregate_metrics = _load_json(resolved["aggregate_metrics_path"])

    config_path = os.path.join(run_dir, "eval_config.json") if run_dir else "eval_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    return EvalResult(
        benchmark=config.benchmark,
        engine=config.evaluation_engine,
        config=params,
        config_path=config_path,
        questions_path=resolved["questions_path"],
        answers_path=resolved["answers_path"],
        metrics_path=resolved["metrics_path"],
        aggregate_metrics_path=resolved["aggregate_metrics_path"],
        dashboard_path=dashboard_path,
        aggregate_metrics=aggregate_metrics,
        manifest_path=manifest_path,
        manifest=manifest,
    )


# --------------------------------------------------------------------------- #
# Shared argparse surface (used by the CLI command and ``python -m``)
# --------------------------------------------------------------------------- #


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``cognee eval`` flags onto a parser."""
    parser.add_argument(
        "--benchmark",
        "-b",
        default=None,
        help="Benchmark dataset to evaluate (e.g. HotPotQA, Musique, Dummy, and - once "
        "registered - LongMemEval). Defaults to the configured EvalConfig value.",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=sorted(ENGINE_CHOICES.keys()),
        default=None,
        help="Evaluation engine. 'deepeval' requires the eval extra; 'direct_llm' uses the "
        "default LLM from your .env.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Number of samples to include in the corpus.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic corpus sampling (default: 42).",
    )
    parser.add_argument(
        "--qa-engine",
        default=None,
        help="Retriever used to answer questions (e.g. cognee_graph_completion).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for run artifacts. Artifacts are namespaced by benchmark/engine so "
        "runs stay comparable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume an interrupted run in --output-dir: reuse the built corpus and every "
        "answer already answered ok. Refused if the benchmark, seed, sample count, QA engine "
        "or evaluation settings differ from the interrupted run.",
    )
    parser.add_argument(
        "--max-answer-failure-rate",
        type=float,
        default=None,
        help="Fail the run if more than this fraction (0-1) of questions could not be "
        "answered. 0 tolerates none. Unset: failures are reported, never fatal.",
    )
    parser.add_argument(
        "--max-unscored-rate",
        type=float,
        default=None,
        help="Fail the run if any metric left more than this fraction (0-1) of its questions "
        "unscored. Unset: coverage is reported, never fatal.",
    )
    parser.add_argument(
        "--strict-integrity",
        action="store_true",
        default=None,
        help="Fail the run if cross-file validation finds an error (missing or orphaned "
        "entries, empty golden answers) between questions, answers and metrics.",
    )
    dashboard_group = parser.add_mutually_exclusive_group()
    dashboard_group.add_argument(
        "--dashboard",
        dest="dashboard",
        action="store_true",
        default=None,
        help="Generate an HTML dashboard (default: on; requires the eval extra).",
    )
    dashboard_group.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Skip dashboard generation.",
    )


def config_from_namespace(args: argparse.Namespace) -> EvalConfig:
    """Build an EvalConfig from parsed CLI args, applying only provided overrides."""
    overrides: dict[str, Any] = {}

    if getattr(args, "benchmark", None) is not None:
        overrides["benchmark"] = args.benchmark
    if getattr(args, "engine", None) is not None:
        overrides["evaluation_engine"] = ENGINE_CHOICES[args.engine]
    if getattr(args, "limit", None) is not None:
        overrides["number_of_samples_in_corpus"] = args.limit
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "qa_engine", None) is not None:
        overrides["qa_engine"] = args.qa_engine
    if getattr(args, "output_dir", None) is not None:
        overrides["results_dir"] = args.output_dir
    if getattr(args, "dashboard", None) is not None:
        overrides["dashboard"] = args.dashboard
    if getattr(args, "resume", None) is not None:
        overrides["resume"] = args.resume
    if getattr(args, "max_answer_failure_rate", None) is not None:
        overrides["max_answer_failure_rate"] = args.max_answer_failure_rate
    if getattr(args, "max_unscored_rate", None) is not None:
        overrides["max_unscored_rate"] = args.max_unscored_rate
    if getattr(args, "strict_integrity", None) is not None:
        overrides["strict_integrity"] = args.strict_integrity

    config = EvalConfig(**overrides)

    # DirectLLM only scores 'correctness', so pin the metrics list to it.
    if config.evaluation_engine == "DirectLLM":
        config.evaluation_metrics = ["correctness"]

    return config


def summarize_result(result: EvalResult) -> list[str]:
    """Human-readable summary lines for a completed run."""
    lines = [
        f"Benchmark: {result.benchmark}",
        f"Engine:    {result.engine}",
        f"Metrics:   {result.metrics_path}",
        f"Config:    {result.config_path}",
    ]
    if result.dashboard_path:
        lines.append(f"Dashboard: {result.dashboard_path}")
    if result.manifest_path:
        lines.append(f"Manifest:  {result.manifest_path}")
    answers = result.manifest.stages["answers"] if result.manifest else None
    if answers and answers.total is not None:
        lines.append(
            f"Answers:   {answers.ok} ok, {answers.empty} empty, "
            f"{answers.failed} failed (failed ones are not scored) of {answers.total}"
        )
    if result.aggregate_metrics:
        lines.append("Aggregate metrics:")
        for metric, stats in result.aggregate_metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                line = f"  - {metric}: mean={stats['mean']:.4f}"
                coverage = stats.get("coverage")
                if coverage:
                    line += f" (scored {coverage['scored']}/{coverage['total']})"
                lines.append(line)
            else:
                lines.append(f"  - {metric}: {stats}")
    return lines
