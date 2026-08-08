"""
Percentile report runner for bench_cognee.py.

Executes bench_cognee.py N times sequentially, collects timing results,
and produces a p50/p75/p90/p95/p99/max percentile report with HTML visualization.

Usage:
    python bench_report.py                  # 5 runs with defaults
    python bench_report.py --runs 10        # 10 runs
    python bench_report.py --runs 3 -o report.json  # save raw data + stats
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


BENCH_SCRIPT = (Path(__file__).parent / "statistics_percentile" / "bench_cognee.py").resolve()
RESULTS_DIR = Path(__file__).parent / "results"
COGNEE_DIR = Path(__file__).resolve().parents[3]
METRICS = [
    "add_time_s",
    "cognify_time_s",
    "total_ingest_time_s",
    "search_time",
    "prune_time_s",
    "db_setup_time_s",
    # Local + cloud --create-tenant modes (populated-dataset deletion).
    "dataset_delete_time_s",
    # Cloud --create-tenant runs only; build_report skips metrics absent
    # from the run results, so local/fixed-tenant runs are unaffected.
    "tenant_create_time_s",
    "tenant_delete_time_s",
]
PERCENTILES = [50, 75, 90, 95, 99]
LABELS = {
    "add_time_s": "cognee.add()",
    "cognify_time_s": "cognee.cognify()",
    "total_ingest_time_s": "Total ingest",
    "search_time": "Search",
    "prune_time_s": "Prune",
    "db_setup_time_s": "DB setup",
    "tenant_create_time_s": "Tenant create (cloud)",
    "dataset_delete_time_s": "Dataset delete",
    "tenant_delete_time_s": "Tenant delete (cloud)",
    "wall_time_s": "Wall clock (e2e)",
}


def percentile(sorted_values: list[float], p: int) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (p / 100) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def run_single(run_num: int, total: int, extra_args: list[str]) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    # BENCH_CMD lets another SDK (e.g. the Rust cognee-cli bench) drive this
    # orchestrator unchanged. When unset, fall back to the Python bench script.
    bench_cmd = os.environ.get("BENCH_CMD")
    base = shlex.split(bench_cmd) if bench_cmd else [sys.executable, str(BENCH_SCRIPT)]
    cmd = base + ["--output", tmp_path] + extra_args
    print(f"\n{'=' * 60}")
    print(f"  RUN {run_num}/{total}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, text=True, cwd=str(COGNEE_DIR))
    wall = time.time() - t0

    # A non-zero exit with results present is a FAILED RUN (the bench exits 1
    # after writing its JSON when any phase failed) — keep it as data so the
    # percentile report covers it; the report itself exits non-zero at the
    # end. Only a crash that produced no parseable results aborts the report.
    try:
        with open(tmp_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        Path(tmp_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Run {run_num} crashed with exit code {result.returncode} and produced no results"
        )

    Path(tmp_path).unlink(missing_ok=True)
    data["wall_time_s"] = round(wall, 3)
    return data


def build_report(runs: list[dict]) -> dict:
    stats = {}
    # Only report metrics the runs actually produced: cloud runs carry
    # tenant metrics, local runs don't, and vice versa for local-only ones.
    all_metrics = [m for m in METRICS + ["wall_time_s"] if m in runs[0]]
    for metric in all_metrics:
        values = sorted(r[metric] for r in runs)
        entry = {
            "min": values[0],
            "max": values[-1],
            "mean": round(sum(values) / len(values), 3),
        }
        for p in PERCENTILES:
            entry[f"p{p}"] = round(percentile(values, p), 3)
        entry["values"] = values
        stats[metric] = entry
    return stats


def print_report(stats: dict, num_runs: int, config: dict, runs: list[dict]):
    succeeded = sum(1 for r in runs if r.get("success", True))
    failed = num_runs - succeeded

    print(f"\n{'#' * 60}")
    print(f"  PERCENTILE REPORT  ({num_runs} run{'s' if num_runs != 1 else ''})")
    print(f"{'#' * 60}")
    print(f"  LLM model       : {config.get('llm_model', '?')}")
    print(
        f"  Embedding model  : {config.get('embedding_model', '?')} ({config.get('embedding_dimensions', '?')}d)"
    )
    print(f"  Success rate     : {succeeded}/{num_runs} ({failed} failed)")
    print()

    header = f"  {'Metric':<22} {'min':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8} {'mean':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for metric, entry in stats.items():
        label = LABELS.get(metric, metric)
        parts = [f"  {label:<22}"]
        for key in ["min", "p50", "p75", "p90", "p95", "p99", "max", "mean"]:
            parts.append(f"{entry[key]:>7.2f}s")
        print(" ".join(parts))

    print()
    print("  Individual runs:")
    for i, r in enumerate(runs, 1):
        ok = r.get("success", True)
        tag = "OK" if ok else "FAIL"
        detail = ""
        if not ok:
            status = r.get("status", {})
            failures = [f"{k}: {v}" for k, v in status.items() if v != "success"]
            detail = f"  ({'; '.join(failures)})"
        print(f"    Run {i}: {r['total_ingest_time_s']:.2f}s  [{tag}]{detail}")
    print(f"{'#' * 60}\n")


def generate_html(stats: dict, num_runs: int, config: dict, runs: list[dict], path: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pct_keys = ["min", "p50", "p75", "p90", "p95", "p99", "max", "mean"]

    # Build table rows
    table_rows = ""
    for metric, entry in stats.items():
        label = LABELS.get(metric, metric)
        cells = "".join(f"<td>{entry[k]:.3f}s</td>" for k in pct_keys)
        table_rows += f"<tr><td class='metric-name'>{label}</td>{cells}</tr>\n"

    # Metrics actually present in this report (build_report filters by what
    # the runs produced — cloud and local runs carry different sets).
    all_metrics = list(stats.keys())
    run_header_cells = "".join(f"<th>{LABELS.get(m, m)}</th>" for m in all_metrics)

    # Per-run breakdown rows
    succeeded = sum(1 for r in runs if r.get("success", True))
    failed = num_runs - succeeded
    run_rows = ""
    for i, r in enumerate(runs, 1):
        cells = "".join(f"<td>{r.get(m, 0):.3f}s</td>" for m in all_metrics)
        ok = r.get("success", True)
        status_style = "color: var(--green)" if ok else "color: var(--red)"
        status_label = "OK" if ok else "FAIL"
        if not ok:
            st = r.get("status", {})
            failures = [k for k, v in st.items() if v != "success"]
            status_label = f"FAIL ({', '.join(failures)})"
        cells += f"<td style='{status_style}'>{status_label}</td>"
        run_rows += f"<tr><td>Run {i}</td>{cells}</tr>\n"

    # Chart data
    chart_labels = json.dumps([LABELS.get(m, m) for m in all_metrics])
    chart_p50 = json.dumps([stats[m]["p50"] for m in all_metrics])
    chart_p90 = json.dumps([stats[m]["p90"] for m in all_metrics])
    chart_p99 = json.dumps([stats[m]["p99"] for m in all_metrics])
    chart_mean = json.dumps([stats[m]["mean"] for m in all_metrics])

    # Per-run line chart data for total_ingest_time_s
    run_indices = json.dumps(list(range(1, num_runs + 1)))
    run_totals = json.dumps([r["total_ingest_time_s"] for r in runs])
    run_add = json.dumps([r["add_time_s"] for r in runs])
    run_cognify = json.dumps([r["cognify_time_s"] for r in runs])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognee Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e;
    --accent: #58a6ff; --green: #3fb950; --orange: #d29922; --red: #f85149;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 0.9rem; }}
  .config-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .config-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }}
  .config-card .label {{ color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .config-card .value {{ font-size: 1.1rem; font-weight: 600; margin-top: 0.25rem; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }}
  .chart-box h2 {{ font-size: 1rem; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }}
  th, td {{ padding: 0.6rem 1rem; text-align: right; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
  th {{ background: rgba(255,255,255,0.03); color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em; }}
  td.metric-name, th:first-child, td:first-child {{ text-align: left; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .section-title {{ font-size: 1.1rem; margin-bottom: 0.75rem; }}
  @media (max-width: 900px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Cognee Benchmark Report</h1>
<p class="subtitle">{num_runs} run{"s" if num_runs != 1 else ""} &middot; {timestamp}</p>

<div class="config-grid">
  <div class="config-card"><div class="label">LLM Model</div><div class="value">{config.get("llm_model", "?")}</div></div>
  <div class="config-card"><div class="label">Embedding Model</div><div class="value">{config.get("embedding_model", "?")} ({config.get("embedding_dimensions", "?")}d)</div></div>
  <div class="config-card"><div class="label">Memories</div><div class="value">{runs[0].get("memories_count", "?")}</div></div>
  <div class="config-card"><div class="label">Runs</div><div class="value">{num_runs}</div></div>
  <div class="config-card"><div class="label">Success Rate</div><div class="value" style="color: {"var(--green)" if failed == 0 else "var(--red)"}">{succeeded}/{num_runs}</div></div>
</div>

<div class="charts">
  <div class="chart-box">
    <h2>Percentile Comparison</h2>
    <canvas id="barChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Per-Run Timings</h2>
    <canvas id="lineChart"></canvas>
  </div>
</div>

<h2 class="section-title">Percentile Summary</h2>
<table>
  <thead><tr><th>Metric</th><th>Min</th><th>p50</th><th>p75</th><th>p90</th><th>p95</th><th>p99</th><th>Max</th><th>Mean</th></tr></thead>
  <tbody>{table_rows}</tbody>
</table>

<h2 class="section-title">Individual Runs</h2>
<table>
  <thead><tr><th>Run</th>{run_header_cells}<th>Status</th></tr></thead>
  <tbody>{run_rows}</tbody>
</table>

<script>
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {{
  type: 'bar',
  data: {{
    labels: {chart_labels},
    datasets: [
      {{ label: 'p50', data: {chart_p50}, backgroundColor: 'rgba(88,166,255,0.8)' }},
      {{ label: 'p90', data: {chart_p90}, backgroundColor: 'rgba(63,185,80,0.8)' }},
      {{ label: 'p99', data: {chart_p99}, backgroundColor: 'rgba(210,153,34,0.8)' }},
      {{ label: 'Mean', data: {chart_mean}, backgroundColor: 'rgba(248,81,73,0.6)', borderColor: 'rgba(248,81,73,1)', borderWidth: 1 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v + 's' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }}
    }}
  }}
}});

const lineCtx = document.getElementById('lineChart').getContext('2d');
new Chart(lineCtx, {{
  type: 'line',
  data: {{
    labels: {run_indices},
    datasets: [
      {{ label: 'Total Ingest', data: {run_totals}, borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3 }},
      {{ label: 'cognee.add()', data: {run_add}, borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, tension: 0.3 }},
      {{ label: 'cognee.cognify()', data: {run_cognify}, borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.1)', fill: true, tension: 0.3 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Run #', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }},
      y: {{ title: {{ display: true, text: 'Seconds', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(48,54,61,0.5)' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    path.write_text(html)
    print(f"HTML report saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run bench_cognee.py N times and produce a percentile report."
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=5, help="Number of sequential runs (default: 5)"
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save full report as JSON")
    parser.add_argument(
        "--html",
        type=Path,
        default=RESULTS_DIR / "report.html",
        help="Save HTML report (default: results/report.html)",
    )
    parser.add_argument("--memories", type=Path, default=None, help="Forward to bench_cognee.py")
    parser.add_argument("--llm-model", default=None, help="Forward to bench_cognee.py")
    parser.add_argument("--llm-provider", default=None, help="Forward to bench_cognee.py")
    parser.add_argument("--embedding-model", default=None, help="Forward to bench_cognee.py")
    parser.add_argument("--embedding-provider", default=None, help="Forward to bench_cognee.py")
    parser.add_argument(
        "--embedding-dims", type=int, default=None, help="Forward to bench_cognee.py"
    )
    parser.add_argument(
        "--num-memories", type=int, default=None, help="Limit number of memories to load"
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=False,
        help="Use mock LLM/embedding (no API calls)",
    )
    parser.add_argument(
        "--mock-memories", type=Path, default=None, help="Forward to bench_cognee.py"
    )
    parser.add_argument(
        "--tenant-url",
        default=None,
        help="Cognee Cloud tenant URL; benchmark runs remotely via cognee.serve()",
    )
    parser.add_argument(
        "--tenant-api-key",
        default=None,
        help="API key for the cloud tenant (or set COGNEE_API_KEY)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Forward to bench_cognee.py: per-suite dataset name for cloud mode",
    )
    parser.add_argument(
        "--create-tenant",
        action="store_true",
        default=False,
        help="Forward to bench_cognee.py: create+measure+delete a tenant per run",
    )
    parser.add_argument(
        "--management-url",
        default=None,
        help="Forward to bench_cognee.py: tenant-controller API base URL",
    )
    args = parser.parse_args()

    extra_args = []
    if args.memories:
        extra_args += ["--memories", str(args.memories)]
    if args.llm_model:
        extra_args += ["--llm-model", args.llm_model]
    if args.llm_provider:
        extra_args += ["--llm-provider", args.llm_provider]
    if args.embedding_model:
        extra_args += ["--embedding-model", args.embedding_model]
    if args.embedding_provider:
        extra_args += ["--embedding-provider", args.embedding_provider]
    if args.embedding_dims:
        extra_args += ["--embedding-dims", str(args.embedding_dims)]
    if args.num_memories:
        extra_args += ["--num-memories", str(args.num_memories)]
    if args.mock_llm:
        extra_args += ["--mock-llm"]
    if args.mock_memories:
        extra_args += ["--mock-memories", str(args.mock_memories)]
    if args.tenant_url:
        extra_args += ["--tenant-url", args.tenant_url]
    if args.tenant_api_key:
        extra_args += ["--tenant-api-key", args.tenant_api_key]
    if args.dataset_name:
        extra_args += ["--dataset-name", args.dataset_name]
    if args.create_tenant:
        extra_args += ["--create-tenant"]
    if args.management_url:
        extra_args += ["--management-url", args.management_url]

    mode = (
        " [MOCK LLM]"
        if args.mock_llm
        else " [CLOUD]"
        if (args.tenant_url or args.create_tenant)
        else ""
    )
    print(f"Starting {args.runs} sequential run(s) of bench_cognee.py...{mode}")

    runs = []
    for i in range(1, args.runs + 1):
        import time

        if i != 1 and not args.mock_llm:
            time.sleep(60)

        data = run_single(i, args.runs, extra_args)
        runs.append(data)

    stats = build_report(runs)
    config = runs[0].get("config", {})
    print_report(stats, args.runs, config, runs)

    if args.output:
        succeeded = sum(1 for r in runs if r.get("success", True))
        report = {
            "num_runs": args.runs,
            "succeeded": succeeded,
            "failed": args.runs - succeeded,
            "config": config,
            "stats": {
                k: {sk: sv for sk, sv in v.items() if sk != "values"} for k, v in stats.items()
            },
            "raw_runs": runs,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Full report saved to {args.output}")

    generate_html(stats, args.runs, config, runs, args.html)

    # Propagate run failures via the exit code AFTER all artifacts are
    # written, so CI both keeps the report and fails the job.
    failed = args.runs - sum(1 for r in runs if r.get("success", True))
    if failed:
        sys.exit(f"{failed}/{args.runs} benchmark runs failed (report artifacts were written).")


if __name__ == "__main__":
    main()
