import unittest
import json
import os


from cognee.eval_framework.analysis.dashboard_generator import (
    create_distribution_plots,
    create_ci_plot,
    generate_details_html,
    get_dashboard_html_template,
    create_dashboard,
)


class TestDashboardFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.metrics_data = {
            "accuracy": [0.8, 0.85, 0.9, 0.95, 1.0],
            "f1_score": [0.7, 0.75, 0.8, 0.85, 0.9],
        }

        self.ci_data = {
            "accuracy": (0.9, 0.85, 0.95),
            "f1_score": (0.8, 0.75, 0.85),
        }

        self.detail_data = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence",
                "golden_answer": "Artificial Intelligence",
                "metrics": {
                    "accuracy": {"score": 1.0, "reason": "Exact match"},
                    "f1_score": {"score": 0.9, "reason": "High similarity"},
                },
            }
        ]

    def test_generate_details_html(self):
        """Test HTML details generation."""
        html_output = generate_details_html(self.detail_data)

        self.assertIn("<h3>accuracy Details</h3>", html_output[0])
        self.assertIn("<th>Question</th>", html_output[1])
        self.assertIn("Exact match", "".join(html_output))

    def test_get_dashboard_html_template(self):
        """Test full dashboard HTML generation."""
        figures = create_distribution_plots(self.metrics_data)
        ci_plot = create_ci_plot(self.ci_data)
        dashboard_html = get_dashboard_html_template(
            figures + [ci_plot], generate_details_html(self.detail_data), "Benchmark 1"
        )

        self.assertIn("<title>LLM Evaluation Dashboard Benchmark 1</title>", dashboard_html)
        self.assertIn("<h2>Metrics Distribution</h2>", dashboard_html)
        self.assertIn("<h2>95% confidence interval for all the metrics</h2>", dashboard_html)
        self.assertIn("Benchmark 1", dashboard_html)

    def test_create_dashboard(self):
        """Test the full dashboard generation and file creation."""
        metrics_path = "test_metrics.json"
        aggregate_metrics_path = "test_aggregate.json"
        output_file = "test_dashboard.html"

        with open(metrics_path, "w") as f:
            json.dump(self.detail_data, f)

        with open(aggregate_metrics_path, "w") as f:
            json.dump(
                {
                    metric: {"mean": v[0], "ci_lower": v[1], "ci_upper": v[2]}
                    for metric, v in self.ci_data.items()
                },
                f,
            )

        output = create_dashboard(
            metrics_path, aggregate_metrics_path, output_file, "Test Benchmark"
        )

        self.assertEqual(output, output_file)
        self.assertTrue(os.path.exists(output_file))

        os.remove(metrics_path)
        os.remove(aggregate_metrics_path)
        os.remove(output_file)
