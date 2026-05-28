"""
Pytest wrapper for the performance analysis benchmark.

Runs statistics_percentile_report.py with mock LLM and validates:
- JSON and HTML report files are generated
- All benchmark runs completed without failures
"""

import json
import subprocess
import sys
from pathlib import Path

REPORT_SCRIPT = Path(__file__).parent / "performance" / "statistics_percentile_report.py"
COGNEE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).parent / ".data_storage" / "test_performance_analysis"


def test_performance_analysis():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "report.json"
    html_path = RESULTS_DIR / "report.html"

    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--runs",
            "5",
            "--output",
            str(json_path),
            "--html",
            str(html_path),
            "--mock-llm",
            "--num-memories",
            "20",
        ],
        cwd=str(COGNEE_ROOT),
        text=True,
        capture_output=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Performance report script failed (exit code {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    assert json_path.exists(), "JSON report was not generated"
    assert html_path.exists(), "HTML report was not generated"

    with open(json_path) as f:
        report = json.load(f)

    assert report["failed"] == 0, f"{report['failed']}/{report['num_runs']} runs failed"

    for i, run in enumerate(report["raw_runs"], 1):
        assert run.get("success") is True, f"Run {i} failed: {run.get('status', {})}"

    print(f"\nJSON report: {json_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    test_performance_analysis()
