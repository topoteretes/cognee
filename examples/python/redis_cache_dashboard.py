import base64
from io import BytesIO
from pathlib import Path

import numpy as np


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_html_dashboard(
    stats: dict,
    per_query_stats: list[dict],
    queries: list[str],
    output_path: Path,
    title: str = "Redis cache evaluation",
    graph_size: dict | None = None,
    executions_per_query: int | None = None,
) -> None:
    """Generate an HTML dashboard with summary table, per-query table, and figures.

    graph_size: optional dict with 'num_nodes' and 'num_edges' to show graph size (e.g. from get_graph_metrics).
    executions_per_query: optional total number of times each query was run (e.g. n_runs * num_search_types).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(stats.keys())
    p50_vals = [stats[n]["p50"] for n in names]
    p95_vals = [stats[n]["p95"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.bar(x - width / 2, p50_vals, width, label="p50", color="#2ecc71")
    bars2 = ax1.bar(x + width / 2, p95_vals, width, label="p95", color="#e74c3c")
    ax1.set_ylabel("Latency (s)")
    ax1.set_title("Search latency: p50 and p95 by search type")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}s", ha="center", fontsize=9)
    for b in bars2:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}s", ha="center", fontsize=9)
    img1 = _fig_to_base64(fig1)
    plt.close(fig1)

    fig2, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names, strict=True):
        ax.boxplot(stats[name]["latencies"], vert=True)
        ax.set_ylabel("Latency (s)")
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
    fig2.suptitle("Latency distribution per search type", fontsize=12)
    img2 = _fig_to_base64(fig2)
    plt.close(fig2)

    fig3, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names, strict=True):
        ax.hist(stats[name]["latencies"], bins=min(15, max(3, len(stats[name]["latencies"]) // 2)), color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(stats[name]["p50"], color="#2ecc71", linestyle="--", linewidth=2, label="p50")
        ax.axvline(stats[name]["p95"], color="#e74c3c", linestyle="--", linewidth=2, label="p95")
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig3.suptitle("Latency histograms with p50/p95", fontsize=12)
    img3 = _fig_to_base64(fig3)
    plt.close(fig3)

    rows = "".join(
        f"""
        <tr>
            <td>{name}</td>
            <td>{s['count']}</td>
            <td>{s['mean']:.3f}</td>
            <td>{s['min']:.3f}</td>
            <td>{s['max']:.3f}</td>
            <td><strong>{s['p50']:.3f}</strong></td>
            <td><strong>{s['p95']:.3f}</strong></td>
        </tr>"""
        for name, s in stats.items()
    )
    type_names = list(stats.keys())
    query_rows = ""
    for pq in per_query_stats:
        q_display = pq["query"][:60] + ("..." if len(pq["query"]) > 60 else "")
        cells = f'<td title="{pq["query"]}">{q_display}</td>'
        for t in type_names:
            s = pq["stats"].get(t, {})
            cells += f"<td>{s.get('p50', 0):.3f}</td><td>{s.get('p95', 0):.3f}</td>"
        query_rows += f"<tr>{cells}</tr>"
    th_cols = "".join(f"<th colspan=\"2\">{t}</th>" for t in type_names)
    sub_th = "".join("<th>p50</th><th>p95</th>" for _ in type_names)
    if graph_size is not None and ("num_nodes" in graph_size or "num_edges" in graph_size):
        n_nodes = int(graph_size.get("num_nodes") or 0)
        n_edges = int(graph_size.get("num_edges") or 0)
        graph_card = f"""
    <div class="card">
        <h2>Graph size</h2>
        <p><strong>{n_nodes:,}</strong> nodes &nbsp; · &nbsp; <strong>{n_edges:,}</strong> edges</p>
    </div>"""
    else:
        graph_card = ""
    if executions_per_query is not None and executions_per_query > 0:
        query_heading = f"Queries executed (each run {executions_per_query:,} times)"
        query_items = "".join(f"<li>{q} <span class=\"muted\">({executions_per_query:,} runs)</span></li>" for q in queries)
    else:
        query_heading = "Queries executed"
        query_items = "".join(f"<li>{q}</li>" for q in queries)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{ --bg: #0f1419; --card: #1a2332; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149; }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }}
        h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; }}
        .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
        .muted {{ color: var(--muted); font-size: 0.9em; }}
        .card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        .card h2 {{ margin-top: 0; font-size: 1.25rem; color: var(--accent); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.08); }}
        th {{ color: var(--muted); font-weight: 600; }}
        td strong {{ color: var(--green); }}
        .figure {{ margin: 1.5rem 0; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="subtitle">Search speed comparison: TRIPLET_COMPLETION, TRIPLET_COMPLETION_CACHE, GRAPH_COMPLETION</p>
    {graph_card}

    <div class="card">
        <h2>{query_heading}</h2>
        <ol>
            {query_items}
        </ol>
    </div>

    <div class="card">
        <h2>Per-query p50 / p95 (seconds)</h2>
        <table>
            <thead>
                <tr><th>Query</th>{th_cols}</tr>
                <tr><th></th>{sub_th}</tr>
            </thead>
            <tbody>
                {query_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Summary – pooled across all queries (latency in seconds)</h2>
        <table>
            <thead>
                <tr>
                    <th>Search type</th>
                    <th>Runs</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>p50</th>
                    <th>p95</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>p50 and p95 by search type</h2>
        <div class="figure"><img src="data:image/png;base64,{img1}" alt="p50 p95 bar chart" /></div>
    </div>

    <div class="card">
        <h2>Latency distribution (box plot)</h2>
        <div class="figure"><img src="data:image/png;base64,{img2}" alt="Box plots" /></div>
    </div>

    <div class="card">
        <h2>Latency histograms with p50 / p95</h2>
        <div class="figure"><img src="data:image/png;base64,{img3}" alt="Histograms" /></div>
    </div>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
