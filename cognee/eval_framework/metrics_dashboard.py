import json
import plotly.graph_objects as go
from typing import Dict, List, Tuple
from collections import defaultdict

metrics_fields = {
    "contextual_relevancy": ["question", "retrieval_context"],
    "context_coverage": ["question", "retrieval_context", "golden_context"],
}
default_metrics_fields = ["question", "answer", "golden_answer"]


def create_distribution_plots(metrics_data: Dict[str, List[float]]) -> List[str]:
    """Create distribution histogram plots for each metric."""
    figures = []
    for metric, scores in metrics_data.items():
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, name=metric, nbinsx=10, marker_color="#1f77b4"))

        fig.update_layout(
            title=f"{metric} Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            bargap=0.1,
            template="seaborn",
        )
        figures.append(fig.to_html(full_html=False))
    return figures


def create_ci_plot(ci_results: Dict[str, Tuple[float, float, float]]) -> str:
    """Create confidence interval bar plot."""
    fig = go.Figure()
    for metric, (mean_score, lower, upper) in ci_results.items():
        fig.add_trace(
            go.Bar(
                x=[metric],
                y=[mean_score],
                error_y=dict(
                    type="data",
                    array=[upper - mean_score],
                    arrayminus=[mean_score - lower],
                    visible=True,
                ),
                name=metric,
            )
        )

    fig.update_layout(
        title="95% confidence interval for all the metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        template="seaborn",
    )
    return fig.to_html(full_html=False)


def generate_details_html(metrics_data: List[Dict]) -> List[str]:
    """Generate HTML for detailed metric information."""
    details_html = []
    metric_details = {}

    # Organize metrics by type
    for entry in metrics_data:
        for metric, values in entry["metrics"].items():
            if metric not in metric_details:
                metric_details[metric] = []
            current_metrics_fields = metrics_fields.get(metric, default_metrics_fields)
            metric_details[metric].append(
                {key: entry[key] for key in current_metrics_fields}
                | {
                    "reason": values.get("reason", ""),
                    "score": values["score"],
                }
            )

    for metric, details in metric_details.items():
        formatted_column_names = [key.replace("_", " ").title() for key in details[0].keys()]
        details_html.append(f"<h3>{metric} Details</h3>")
        details_html.append(f"""
            <table class="metric-table">
                <tr>
                    {"".join(f"<th>{col}</th>" for col in formatted_column_names)}
                </tr>
        """)
        for item in details:
            details_html.append(f"""
                <tr>
                    {"".join(f"<td>{value}</td>" for value in item.values())}
                </tr>
            """)
        details_html.append("</table>")
    return details_html


def get_dashboard_html_template(
    figures: List[str], details_html: List[str], benchmark: str = ""
) -> str:
    """Generate the complete HTML dashboard template."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Evaluation Dashboard {benchmark}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; }}
            .metric-table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metric-table th {{ background-color: #f2f2f2; }}
            h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>LLM Evaluation Metrics Dashboard {benchmark}</h1>

        <h2>Metrics Distribution</h2>
        {"".join([f'<div class="chart">{fig}</div>' for fig in figures[:-1]])}

        <h2>95% confidence interval for all the metrics</h2>
        <div class="chart">{figures[-1]}</div>

        <h2>Detailed Explanations</h2>
        {"".join(details_html)}
    </body>
    </html>
    """


def create_dashboard(
    metrics_path: str,
    aggregate_metrics_path: str,
    output_file: str = "dashboard_with_ci.html",
    benchmark: str = "",
) -> str:
    """Create and save the dashboard with all visualizations."""
    # Read metrics files
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    with open(aggregate_metrics_path, "r") as f:
        aggregate_data = json.load(f)

    # Extract data for visualizations
    metrics_by_type = defaultdict(list)
    for entry in metrics_data:
        for metric, values in entry["metrics"].items():
            metrics_by_type[metric].append(values["score"])

    # Generate visualizations
    distribution_figures = create_distribution_plots(metrics_by_type)
    ci_plot = create_ci_plot(
        {
            metric: (data["mean"], data["ci_lower"], data["ci_upper"])
            for metric, data in aggregate_data.items()
        }
    )

    # Combine all figures
    figures = distribution_figures + [ci_plot]

    # Generate HTML components
    details_html = generate_details_html(metrics_data)
    dashboard_html = get_dashboard_html_template(figures, details_html, benchmark)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(dashboard_html)

    return output_file
