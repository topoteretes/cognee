import pytest
from typing import Optional
import sys
from unittest.mock import patch, MagicMock
import unittest
import numpy as np
from cognee.eval_framework.analysis.metrics_calculator import bootstrap_ci


with patch.dict(
    sys.modules,
    {"deepeval": MagicMock(), "deepeval.test_case": MagicMock()},
):
    from cognee.eval_framework.evaluation.metrics.exact_match import ExactMatchMetric
    from cognee.eval_framework.evaluation.metrics.f1 import F1ScoreMetric


class MockTestCase:
    def __init__(self, actual_output: Optional[str], expected_output: Optional[str]):
        self.actual_output = actual_output
        self.expected_output = expected_output


@pytest.fixture
def metrics():
    return {
        "exact_match": ExactMatchMetric(),
        "f1": F1ScoreMetric(),
    }


@pytest.mark.parametrize(
    "actual, expected, expected_exact_score, expected_f1_range",
    [
        ("Hello World", "Hello World", 1.0, (1.0, 1.0)),
        ("Hello World", "hello world", 1.0, (1.0, 1.0)),
        ("Hello   World", "Hello World", 0.0, (0.0, 1.0)),
        ("  Hello World  ", "Hello World", 1.0, (1.0, 1.0)),
        ("", "Hello World", 0.0, (0.0, 0.0)),
        ("Hello World", "", 0.0, (0.0, 0.0)),
        ("", "", 1.0, (1.0, 1.0)),
        ("Hello World", "Goodbye World", 0.0, (0.0, 1.0)),
        ("Hello", "Hello World", 0.0, (0.0, 1.0)),
        ("Hello, World!", "hello, world!", 1.0, (1.0, 1.0)),
        ("123", "123", 1.0, (1.0, 1.0)),
        ("123", "456", 0.0, (0.0, 0.0)),
        ("Café", "café", 1.0, (1.0, 1.0)),
        ("Café", "Cafe", 0.0, (0.0, 0.0)),
    ],
)
def test_metrics(metrics, actual, expected, expected_exact_score, expected_f1_range):
    test_case = MockTestCase(actual, expected)

    exact_match_score = metrics["exact_match"].measure(test_case)
    assert exact_match_score == expected_exact_score, (
        f"Exact match failed for '{actual}' vs '{expected}'"
    )

    f1_score = metrics["f1"].measure(test_case)
    assert expected_f1_range[0] <= f1_score <= expected_f1_range[1], (
        f"F1 score failed for '{actual}' vs '{expected}'"
    )


class TestBootstrapCI(unittest.TestCase):
    def test_bootstrap_ci_basic(self):
        scores = [1, 2, 3, 4, 5]
        mean, lower, upper = bootstrap_ci(scores, num_samples=1000, confidence_level=0.95)

        self.assertAlmostEqual(mean, np.mean(scores), places=2)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

    def test_bootstrap_ci_single_value(self):
        scores = [3, 3, 3, 3, 3]
        mean, lower, upper = bootstrap_ci(scores, num_samples=1000, confidence_level=0.95)

        self.assertEqual(mean, 3)
        self.assertEqual(lower, 3)
        self.assertEqual(upper, 3)

    def test_bootstrap_ci_empty_list(self):
        mean, lower, upper = bootstrap_ci([])

        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(upper))
