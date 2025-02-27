import unittest
from unittest.mock import patch
import json
import os
import tempfile
from evals.eval_framework.metrics_dashboard import generate_metrics_dashboard, bootstrap_ci
import numpy as np


class TestGenerateMetricsDashboard(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence",
                "golden_answer": "Artificial Intelligence",
                "metrics": {
                    "accuracy": {"score": 0.9, "reason": "Close enough"},
                    "relevance": {"score": 0.8},
                },
            },
            {
                "question": "What is ML?",
                "answer": "Machine Learning",
                "golden_answer": "Machine Learning",
                "metrics": {
                    "accuracy": {"score": 0.95, "reason": "Exact match"},
                    "relevance": {"score": 0.85},
                },
            },
        ]

        self.temp_json = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
        json.dump(self.test_data, self.temp_json)
        self.temp_json.close()
        self.output_file = "test_dashboard.html"

    def tearDown(self):
        os.remove(self.temp_json.name)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_generate_metrics_dashboard_valid_json(self):
        """Test if the function processes valid JSON correctly and creates an output file."""
        result = generate_metrics_dashboard(
            self.temp_json.name, self.output_file, benchmark="Test Benchmark"
        )

        self.assertTrue(os.path.exists(self.output_file))
        self.assertEqual(result, self.output_file)

        with open(self.output_file, "r", encoding="utf-8") as f:
            html_content = f.read()
            self.assertIn("<title>LLM Evaluation Dashboard Test Benchmark</title>", html_content)
            self.assertIn("accuracy", html_content)
            self.assertIn("relevance", html_content)

    @patch("evals.eval_framework.metrics_dashboard.bootstrap_ci", return_value=(0.9, 0.85, 0.95))
    def test_generate_metrics_dashboard_ci_calculation(self, mock_bootstrap_ci):
        """Test if bootstrap_ci is called with the correct parameters."""
        generate_metrics_dashboard(self.temp_json.name, self.output_file)

        mock_bootstrap_ci.assert_any_call([0.9, 0.95])  # For accuracy
        mock_bootstrap_ci.assert_any_call([0.8, 0.85])  # For relevance

    @patch("plotly.graph_objects.Figure.to_html", return_value="<div>Plotly Chart</div>")
    def test_generate_metrics_dashboard_plotly_charts(self, mock_to_html):
        """Test if Plotly figures are generated correctly."""
        generate_metrics_dashboard(self.temp_json.name, self.output_file)

        self.assertGreaterEqual(mock_to_html.call_count, 3)  # 2 metrics + CI chart

        with open(self.output_file, "r", encoding="utf-8") as f:
            file_content = f.read()
            self.assertIn(
                "<div>Plotly Chart</div>",
                file_content,
                "The output file does not contain the expected Plotly chart HTML.",
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
