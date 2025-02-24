"""Facade module for metrics calculation and dashboard generation."""

from evals.eval_framework.analysis.metrics_calculator import calculate_metrics_statistics
from evals.eval_framework.analysis.dashboard_generator import create_dashboard


def generate_metrics_dashboard(
    json_data: str,
    output_file: str = "dashboard_with_ci.html",
    aggregate_metrics_path: str = "aggregate_metrics.json",
    benchmark: str = "",
) -> str:
    """Generate metrics dashboard with visualizations and save aggregate metrics."""
    # Calculate metrics statistics
    metrics_data, metric_details, ci_results = calculate_metrics_statistics(
        json_data, aggregate_metrics_path
    )

    # Create and save dashboard
    return create_dashboard(metrics_data, metric_details, ci_results, output_file, benchmark)
