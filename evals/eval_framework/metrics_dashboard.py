import json
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np


def bootstrap_ci(scores, num_samples=10000, confidence_level=0.95):
    means = []
    n = len(scores)
    for _ in range(num_samples):
        sample = np.random.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
    return np.mean(scores), lower_bound, upper_bound


def generate_metrics_dashboard(json_data, output_file="dashboard_with_ci.html", benchmark=""):
    try:
        with open(json_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {json_data}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {json_data}: {e}")

    metrics_data = defaultdict(list)
    metric_details = defaultdict(list)

    for entry in data:
        for metric, values in entry["metrics"].items():
            score = values["score"]
            metrics_data[metric].append(score)
            if "reason" in values:
                metric_details[metric].append(
                    {
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "golden_answer": entry["golden_answer"],
                        "reason": values["reason"],
                        "score": score,
                    }
                )

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

    ci_results = {}
    for metric, scores in metrics_data.items():
        mean_score, lower, upper = bootstrap_ci(scores)
        ci_results[metric] = (mean_score, lower, upper)

    # Bar chart with confidence intervals
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
    figures.append(fig.to_html(full_html=False))

    details_html = []
    for metric, details in metric_details.items():
        details_html.append(f"<h3>{metric} Details</h3>")
        details_html.append("""
            <table class="metric-table">
                <tr>
                    <th>Question</th>
                    <th>Answer</th>
                    <th>Golden Answer</th>
                    <th>Reason</th>
                    <th>Score</th>
                </tr>
        """)
        for item in details:
            details_html.append(
                f"<tr>"
                f"<td>{item['question']}</td>"
                f"<td>{item['answer']}</td>"
                f"<td>{item['golden_answer']}</td>"
                f"<td>{item['reason']}</td>"
                f"<td>{item['score']}</td>"
                f"</tr>"
            )
        details_html.append("</table>")

    html_template = f"""
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
        {"".join([f'<div class="chart">{fig}</div>' for fig in figures[: len(metrics_data)]])}

        <h2>95% confidence interval for all the metrics</h2>
        <div class="chart">{figures[-1]}</div>

        <h2>Detailed Explanations</h2>
        {"".join(details_html)}
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    return output_file
