"""Ablation harness for the self-improvement loop.

Answers the question the loop has never had data for: does turning a learned signal
on actually improve answers? Each axis runs the SAME question set twice — baseline
arm (signal off) and variant arm (signal on) — through the existing retriever sweep
machinery, then reports the per-question and mean deltas of the primary metric.

Axes shipped here:

- ``feedback_influence``: GraphCompletionRetriever with ``feedback_influence`` 0
  vs a non-zero value (reads learned ``feedback_weight`` during triplet ranking).
- ``truth_subspace``: HybridRetriever with ``use_truth_weight`` off vs on (applies
  the truth-alignment factor in the chunk lane).

The report is the gate for flipping DEFAULT_FEEDBACK_INFLUENCE off 0.0
(SELF_IMPROVEMENT_PLAN.md, item 2.5): flip only on a non-negative mean delta.

A lesson-utility metric rides along: the fraction of answers whose retrieval
context contains at least one distilled session-learning document, so lesson
usefulness is measured instead of just lesson counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from cognee.eval_framework.evaluation.run_evaluation_module import run_evaluation
from cognee.eval_framework.reporting.io import write_json
from cognee.eval_framework.sweeps.retriever_sweep_runner import (
    RetrieverSweepSettings,
    answer_with_config,
    evaluate_batch,
    validate_retriever_configs,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Marker rendered by session_distillation.render_lesson_document into every
# published lesson document; its presence in a retrieval context means a
# distilled lesson was actually served to the answer.
LESSON_DOCUMENT_MARKER = "Session learning —"


@dataclass(frozen=True)
class AblationAxis:
    """One on/off comparison: same questions, one knob flipped."""

    name: str
    baseline_config: dict[str, Any]
    variant_config: dict[str, Any]


@dataclass(frozen=True)
class AblationSettings:
    """Paths and eval knobs for one ablation run.

    ``primary_metric_name`` is the key inside each metrics row's ``metrics`` dict
    (e.g. ``correctness``); its ``score`` field is what deltas are computed on.
    """

    output_dir: Path
    primary_metric_name: str = "correctness"
    max_concurrent_questions: int = 10
    artifact_prefix: str = "ablation"
    summary_tags: dict[str, Any] = field(default_factory=dict)


def build_self_improvement_axes(
    *,
    feedback_influence: float = 0.2,
    graph_completion_kwargs: Optional[dict[str, Any]] = None,
    hybrid_kwargs: Optional[dict[str, Any]] = None,
) -> list[AblationAxis]:
    """Default axes for the two learned ranking signals.

    Retriever classes are imported lazily so this module stays importable in
    environments without the full retrieval stack.
    """
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

    graph_kwargs = dict(graph_completion_kwargs or {})
    hybrid_base_kwargs = dict(hybrid_kwargs or {})

    return [
        AblationAxis(
            name="feedback_influence",
            baseline_config={
                "name": "graph_completion_influence_off",
                "mode": "fixed_retriever",
                "retriever_cls": GraphCompletionRetriever,
                "retriever_kwargs": {**graph_kwargs, "feedback_influence": 0.0},
            },
            variant_config={
                "name": f"graph_completion_influence_{feedback_influence}",
                "mode": "fixed_retriever",
                "retriever_cls": GraphCompletionRetriever,
                "retriever_kwargs": {**graph_kwargs, "feedback_influence": feedback_influence},
            },
        ),
        AblationAxis(
            name="truth_subspace",
            baseline_config={
                "name": "hybrid_truth_off",
                "mode": "fixed_retriever",
                "retriever_cls": HybridRetriever,
                "retriever_kwargs": {**hybrid_base_kwargs, "use_truth_weight": False},
            },
            variant_config={
                "name": "hybrid_truth_on",
                "mode": "fixed_retriever",
                "retriever_cls": HybridRetriever,
                "retriever_kwargs": {**hybrid_base_kwargs, "use_truth_weight": True},
            },
        ),
    ]


def lesson_hit_stats(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Lesson utility: how many answers actually retrieved a distilled lesson."""
    per_question = []
    hits = 0
    for answer in answers:
        context = answer.get("retrieval_context") or ""
        hit = LESSON_DOCUMENT_MARKER in context
        hits += 1 if hit else 0
        per_question.append(
            {
                "question_idx": answer.get("question_idx"),
                "lesson_served": hit,
            }
        )
    total = len(answers)
    return {
        "total_answers": total,
        "answers_with_lesson": hits,
        "lesson_hit_rate": (hits / total) if total else 0.0,
        "per_question": per_question,
    }


def _metric_by_question(
    metrics: Any,
    metric_name: str,
) -> dict[Any, float]:
    """Extract {question_idx: metric value} from a metrics artifact. Tolerant of shape."""
    values: dict[Any, float] = {}
    rows = metrics if isinstance(metrics, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_idx = row.get("question_idx")
        metric_value = row.get("metrics", {})
        if isinstance(metric_value, dict):
            entry = metric_value.get(metric_name)
            score = entry.get("score") if isinstance(entry, dict) else entry
        else:
            score = row.get(metric_name)
        try:
            values[question_idx] = float(score)
        except (TypeError, ValueError):
            continue
    return values


def summarize_axis(
    axis_name: str,
    baseline_batch: dict[str, Any],
    variant_batch: dict[str, Any],
    metric_name: str = "correctness",
) -> dict[str, Any]:
    """Per-question and mean delta of ``metric_name`` (variant - baseline)."""
    baseline_scores = _metric_by_question(baseline_batch.get("metrics"), metric_name)
    variant_scores = _metric_by_question(variant_batch.get("metrics"), metric_name)

    shared_questions = sorted(set(baseline_scores) & set(variant_scores), key=lambda idx: str(idx))
    per_question = [
        {
            "question_idx": question_idx,
            "baseline": baseline_scores[question_idx],
            "variant": variant_scores[question_idx],
            "delta": variant_scores[question_idx] - baseline_scores[question_idx],
        }
        for question_idx in shared_questions
    ]
    deltas = [entry["delta"] for entry in per_question]
    regressions = [entry for entry in per_question if entry["delta"] < 0]

    return {
        "axis": axis_name,
        "metric": metric_name,
        "baseline_name": baseline_batch.get("retriever_name"),
        "variant_name": variant_batch.get("retriever_name"),
        "questions_compared": len(per_question),
        "baseline_mean": (sum(baseline_scores[q] for q in shared_questions) / len(per_question))
        if per_question
        else None,
        "variant_mean": (sum(variant_scores[q] for q in shared_questions) / len(per_question))
        if per_question
        else None,
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "regression_count": len(regressions),
        "per_question": per_question,
    }


async def run_self_improvement_ablation(
    questions: list[dict[str, Any]],
    settings: AblationSettings,
    base_eval_params: dict[str, Any],
    axes: Optional[list[AblationAxis]] = None,
    run_evaluation_fn: Callable[[dict[str, Any]], Awaitable[None]] = run_evaluation,
) -> dict[str, Any]:
    """Run every axis (baseline + variant arms) on the same questions and report deltas.

    Writes ``<artifact_prefix>_report.json`` under ``settings.output_dir`` and
    returns the same report dict. Answer/metric artifacts per arm reuse the
    retriever-sweep naming, so individual runs stay inspectable.
    """
    if axes is None:
        axes = build_self_improvement_axes()

    all_configs = [
        config for axis in axes for config in (axis.baseline_config, axis.variant_config)
    ]
    validate_retriever_configs(all_configs)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_settings = RetrieverSweepSettings(
        output_dir=settings.output_dir,
        max_concurrent_questions=settings.max_concurrent_questions,
        primary_metric_name=settings.primary_metric_name,
        artifact_prefix=settings.artifact_prefix,
    )

    async def run_arm(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        answers = await answer_with_config(
            questions=questions,
            config=config,
            run_idx=0,
            max_concurrent=sweep_settings.max_concurrent_questions,
        )
        batch = await evaluate_batch(
            base_params=base_eval_params,
            output_dir=sweep_settings.output_dir,
            artifact_prefix=sweep_settings.artifact_prefix,
            conversation_index=0,
            retriever_name=config["name"],
            run_idx=0,
            answers=answers,
            run_evaluation_fn=run_evaluation_fn,
        )
        return batch, lesson_hit_stats(answers)

    axis_reports = []
    for axis in axes:
        logger.info("[ablation] Running axis '%s'...", axis.name)
        baseline_batch, baseline_lessons = await run_arm(axis.baseline_config)
        variant_batch, variant_lessons = await run_arm(axis.variant_config)

        summary = summarize_axis(
            axis.name, baseline_batch, variant_batch, settings.primary_metric_name
        )
        summary["lesson_utility"] = {
            "baseline": baseline_lessons,
            "variant": variant_lessons,
        }
        axis_reports.append(summary)
        logger.info(
            "[ablation] Axis '%s': mean_delta=%s over %s question(s), %s regression(s)",
            axis.name,
            summary["mean_delta"],
            summary["questions_compared"],
            summary["regression_count"],
        )

    report = {
        "artifact_prefix": settings.artifact_prefix,
        "primary_metric": settings.primary_metric_name,
        "question_count": len(questions),
        "tags": settings.summary_tags,
        "axes": axis_reports,
    }
    report_path = settings.output_dir / f"{settings.artifact_prefix}_report.json"
    write_json(str(report_path), report)
    report["report_path"] = str(report_path)
    return report


__all__ = [
    "AblationAxis",
    "AblationSettings",
    "LESSON_DOCUMENT_MARKER",
    "build_self_improvement_axes",
    "lesson_hit_stats",
    "run_self_improvement_ablation",
    "summarize_axis",
]
