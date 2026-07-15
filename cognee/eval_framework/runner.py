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
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.eval_framework.eval_config import EvalConfig

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
    config: Dict[str, Any]
    config_path: str
    questions_path: str
    answers_path: str
    metrics_path: str
    aggregate_metrics_path: str
    dashboard_path: Optional[str] = None
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_") or "run"


def resolve_run_paths(config: EvalConfig) -> Dict[str, str]:
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
    return {key: os.path.join(run_dir, params[key]) for key in ARTIFACT_KEYS}


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Thin lazy wrappers around the pipeline steps. Importing them here (rather than
# at module top) keeps ``import cognee.eval_framework.runner`` - and therefore
# ``cognee eval --help`` - free of the optional ``eval`` extra. They are also the
# seams the tests patch to exercise orchestration without an LLM or database.
async def _corpus_step(params: Dict[str, Any]) -> Any:
    from cognee.eval_framework.corpus_builder.run_corpus_builder import run_corpus_builder

    return await run_corpus_builder(params)


async def _answer_step(params: Dict[str, Any]) -> Any:
    from cognee.eval_framework.answer_generation.run_question_answering_module import (
        run_question_answering,
    )

    return await run_question_answering(params)


async def _evaluation_step(params: Dict[str, Any]) -> Any:
    from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation

    return await run_evaluation(params)


def _create_dashboard(params: Dict[str, Any]) -> str:
    """Generate the HTML dashboard, importing plotly lazily so core runs (and the
    ``--no-dashboard`` path) never require the ``eval`` extra."""
    try:
        from cognee.eval_framework.metrics_dashboard import create_dashboard
    except ImportError as error:
        raise ImportError(
            "Dashboard generation requires optional dependencies. Install them with: "
            'pip install "cognee[eval]" (or pass --no-dashboard to skip it).'
        ) from error

    create_dashboard(
        metrics_path=params["metrics_path"],
        aggregate_metrics_path=params["aggregate_metrics_path"],
        output_file=params["dashboard_path"],
        benchmark=params["benchmark"],
    )
    return params["dashboard_path"]


async def run_eval(config: Optional[EvalConfig] = None) -> EvalResult:
    """Run the full eval pipeline for a single config and return an EvalResult.

    Chains corpus -> answers -> evaluation -> (optional) dashboard, writing the
    resolved config next to the artifacts for reproducibility.
    """
    config = config or EvalConfig()
    params = config.to_dict()

    resolved = resolve_run_paths(config)
    params.update(resolved)

    run_dir = os.path.dirname(resolved["metrics_path"])
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)

    logger.info(
        "Running eval: benchmark=%s engine=%s samples=%s seed=%s",
        config.benchmark,
        config.evaluation_engine,
        config.number_of_samples_in_corpus,
        config.seed,
    )

    await _corpus_step(params)
    await _answer_step(params)
    await _evaluation_step(params)

    dashboard_path = None
    if params.get("dashboard"):
        logger.info("Generating dashboard...")
        dashboard_path = _create_dashboard(params)

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
    dashboard_group = parser.add_mutually_exclusive_group()
    dashboard_group.add_argument(
        "--dashboard",
        dest="dashboard",
        action="store_true",
        default=None,
        help="Generate an HTML dashboard (requires the eval extra).",
    )
    dashboard_group.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Skip dashboard generation.",
    )


def config_from_namespace(args: argparse.Namespace) -> EvalConfig:
    """Build an EvalConfig from parsed CLI args, applying only provided overrides."""
    overrides: Dict[str, Any] = {}

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

    config = EvalConfig(**overrides)

    # DirectLLM only scores 'correctness', so pin the metrics list to it.
    if config.evaluation_engine == "DirectLLM":
        config.evaluation_metrics = ["correctness"]

    return config


def summarize_result(result: EvalResult) -> List[str]:
    """Human-readable summary lines for a completed run."""
    lines = [
        f"Benchmark: {result.benchmark}",
        f"Engine:    {result.engine}",
        f"Metrics:   {result.metrics_path}",
        f"Config:    {result.config_path}",
    ]
    if result.dashboard_path:
        lines.append(f"Dashboard: {result.dashboard_path}")
    if result.aggregate_metrics:
        lines.append("Aggregate metrics:")
        for metric, stats in result.aggregate_metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                lines.append(f"  - {metric}: mean={stats['mean']:.4f}")
            else:
                lines.append(f"  - {metric}: {stats}")
    return lines
