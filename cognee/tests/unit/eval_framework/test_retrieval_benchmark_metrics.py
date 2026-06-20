import pytest

from cognee.eval_framework.retrieval_benchmark.metrics import (
    aggregate_metric,
    context_overlap_f1,
    context_recall,
    normalize_context_text,
)


def test_normalize_context_text_flattens_lists_and_dicts():
    assert normalize_context_text(["Alpha", {"text": "Beta"}]) == "alpha\nbeta"


def test_context_recall_counts_golden_lines_found():
    golden = "Paris is the capital.\nFrance is in Europe."
    retrieved = "Background: Paris is the capital. More text."
    assert context_recall(retrieved, golden) == 0.5


def test_context_recall_returns_zero_for_empty_golden():
    assert context_recall("anything", "") == 0.0


def test_context_overlap_f1_token_match():
    golden = "quantum computer uses superposition"
    retrieved = "a quantum computer leverages superposition and entanglement"
    score = context_overlap_f1(retrieved, golden)
    assert 0.4 < score < 1.0


def test_aggregate_metric_handles_empty_series():
    assert aggregate_metric([])["count"] == 0.0


def test_aggregate_metric_summarizes_values():
    summary = aggregate_metric([0.2, 0.8])
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["min"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.8)
    assert summary["count"] == 2.0
