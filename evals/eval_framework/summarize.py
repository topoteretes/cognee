import json
from collections import defaultdict
import plotly.graph_objects as go


def generate_metrics_dashboard(json_data, output_file="dashboard.html", benchmark=""):
    with open(json_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics_data = defaultdict(list)
    metric_details = defaultdict(list)

    # Include the score in the details if a reason is provided.
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
                        "score": score,  # Added metric score here
                    }
                )

    figures = []

    # Create histogram figures for each metric
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

    # Create a bar chart for average scores by metric
    avg_scores = {metric: sum(scores) / len(scores) for metric, scores in metrics_data.items()}
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(avg_scores.keys()), y=list(avg_scores.values()), marker_color="#2ca02c")
    )
    fig.update_layout(
        title="Average Scores by Metric",
        yaxis_title="Average Score",
        xaxis_title="Metric",
        template="seaborn",
    )
    figures.append(fig.to_html(full_html=False))

    # Generate detailed explanations including metric score
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
                f"<td>{item['score']}</td>"  # Add score column value here
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

        <h2>Average Scores</h2>
        <div class="chart">{figures[-1]}</div>

        <h2>Detailed Explanations</h2>
        {"".join(details_html)}
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    return output_file
