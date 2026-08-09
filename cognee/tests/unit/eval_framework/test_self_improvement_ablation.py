"""Tests for the self-improvement ablation harness.

The end-to-end test injects a fake run_evaluation_fn (scores derived from the
answer text) and stub retrievers, so no LLM, DB, or network is involved.
"""

import json

import pytest

from cognee.eval_framework.sweeps.self_improvement_ablation import (
    LESSON_DOCUMENT_MARKER,
    AblationAxis,
    AblationSettings,
    build_self_improvement_axes,
    lesson_hit_stats,
    run_self_improvement_ablation,
    summarize_axis,
)


def _metrics_row(question_idx, score, **extra):
    return {
        "question_idx": question_idx,
        "metrics": {"correctness": {"score": score, "reason": "r"}},
        **extra,
    }


def test_summarize_axis_computes_mean_delta_and_regressions():
    baseline = {
        "retriever_name": "off",
        "metrics": [_metrics_row(1, 0.5), _metrics_row(2, 0.8), _metrics_row(3, 0.4)],
    }
    variant = {
        "retriever_name": "on",
        "metrics": [_metrics_row(1, 0.7), _metrics_row(2, 0.6), _metrics_row(3, 0.4)],
    }

    summary = summarize_axis("feedback_influence", baseline, variant)

    assert summary["questions_compared"] == 3
    assert summary["mean_delta"] == pytest.approx((0.2 - 0.2 + 0.0) / 3)
    assert summary["regression_count"] == 1
    assert summary["baseline_name"] == "off"
    assert summary["variant_name"] == "on"


def test_summarize_axis_ignores_questions_missing_from_one_arm():
    baseline = {"retriever_name": "off", "metrics": [_metrics_row(1, 0.5)]}
    variant = {
        "retriever_name": "on",
        "metrics": [_metrics_row(1, 0.9), _metrics_row(2, 0.1)],
    }

    summary = summarize_axis("axis", baseline, variant)

    assert summary["questions_compared"] == 1
    assert summary["mean_delta"] == pytest.approx(0.4)


def test_lesson_hit_stats_detects_served_lessons():
    answers = [
        {"question_idx": 1, "retrieval_context": f"# {LESSON_DOCUMENT_MARKER} 2026-01-01\nBody"},
        {"question_idx": 2, "retrieval_context": "plain chunk text"},
        {"question_idx": 3, "retrieval_context": ""},
    ]

    stats = lesson_hit_stats(answers)

    assert stats["total_answers"] == 3
    assert stats["answers_with_lesson"] == 1
    assert stats["lesson_hit_rate"] == pytest.approx(1 / 3)
    assert stats["per_question"][0]["lesson_served"] is True


def test_build_self_improvement_axes_flips_exactly_one_knob():
    axes = build_self_improvement_axes(feedback_influence=0.3)

    by_name = {axis.name: axis for axis in axes}
    influence_axis = by_name["feedback_influence"]
    assert influence_axis.baseline_config["retriever_kwargs"]["feedback_influence"] == 0.0
    assert influence_axis.variant_config["retriever_kwargs"]["feedback_influence"] == 0.3

    truth_axis = by_name["truth_subspace"]
    assert truth_axis.baseline_config["retriever_kwargs"]["use_truth_weight"] is False
    assert truth_axis.variant_config["retriever_kwargs"]["use_truth_weight"] is True


class _StubRetriever:
    """Deterministic retriever: baseline arm answers poorly, variant arm well."""

    def __init__(self, quality: str = "good", serve_lesson: bool = False, **_kwargs):
        self.quality = quality
        self.serve_lesson = serve_lesson

    async def get_retrieved_objects(self, query=None, query_batch=None):
        return {"query": query}

    async def get_context_from_objects(self, query=None, retrieved_objects=None):
        if self.serve_lesson:
            return f"# {LESSON_DOCUMENT_MARKER} 2026-01-01 (session s1)\nlesson body"
        return "chunk context"

    async def get_completion_from_context(self, query=None, retrieved_objects=None, context=None):
        return "right answer" if self.quality == "good" else "wrong answer"


async def _fake_run_evaluation(params):
    """Score 1.0 for 'right answer', 0.0 otherwise; write both artifacts."""
    with open(params["answers_path"], "r", encoding="utf-8") as f:
        answers = json.load(f)
    metrics = [
        {
            **answer,
            "metrics": {
                "correctness": {"score": 1.0 if answer["answer"] == "right answer" else 0.0}
            },
        }
        for answer in answers
    ]
    with open(params["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    with open(params["aggregate_metrics_path"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "correctness": {
                    "mean": sum(m["metrics"]["correctness"]["score"] for m in metrics)
                    / len(metrics)
                }
            },
            f,
        )


@pytest.mark.asyncio
async def test_run_self_improvement_ablation_end_to_end(tmp_path):
    questions = [
        {"question_idx": 1, "question": "q1", "answer": "golden", "conversation_id": "c"},
        {"question_idx": 2, "question": "q2", "answer": "golden", "conversation_id": "c"},
    ]
    axis = AblationAxis(
        name="feedback_influence",
        baseline_config={
            "name": "influence_off",
            "mode": "fixed_retriever",
            "retriever_cls": _StubRetriever,
            "retriever_kwargs": {"quality": "bad"},
        },
        variant_config={
            "name": "influence_on",
            "mode": "fixed_retriever",
            "retriever_cls": _StubRetriever,
            "retriever_kwargs": {"quality": "good", "serve_lesson": True},
        },
    )
    settings = AblationSettings(output_dir=tmp_path, artifact_prefix="test_ablation")

    report = await run_self_improvement_ablation(
        questions=questions,
        settings=settings,
        base_eval_params={},
        axes=[axis],
        run_evaluation_fn=_fake_run_evaluation,
    )

    assert report["question_count"] == 2
    (axis_report,) = report["axes"]
    assert axis_report["mean_delta"] == pytest.approx(1.0)
    assert axis_report["regression_count"] == 0
    assert axis_report["lesson_utility"]["baseline"]["lesson_hit_rate"] == 0.0
    assert axis_report["lesson_utility"]["variant"]["lesson_hit_rate"] == 1.0

    report_path = tmp_path / "test_ablation_report.json"
    assert report_path.exists()
    persisted = json.loads(report_path.read_text())
    assert persisted["axes"][0]["variant_name"] == "influence_on"
