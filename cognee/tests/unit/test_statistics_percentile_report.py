"""The percentile report must tolerate runs that skipped a phase.

build_report derived the metric list from runs[0] alone and then indexed every
run by it. With a skipped search phase (no search_time_* key on that run) the
outcome depended on run order: [measured, skipped] raised KeyError, and
[skipped, measured] silently dropped the metric.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "performance" / "statistics_percentile_report.py"
_SPEC = importlib.util.spec_from_file_location("statistics_percentile_report", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

M = "search_time_graph_completion"


def _measured(v=2.0):
    return {
        "add_time_s": 1.0,
        "cognify_time_s": 1.0,
        "total_ingest_time_s": 2.0,
        M: v,
        "wall_time_s": 5.0,
        "success": True,
    }


def _skipped():
    # elapsed-until-failure: add/cognify/total are always recorded, search is not
    return {
        "add_time_s": 1.0,
        "cognify_time_s": 0.5,
        "total_ingest_time_s": 1.5,
        "wall_time_s": 2.0,
        "success": False,
    }


@pytest.mark.parametrize(
    "runs, expect_measured",
    [
        ([_measured(), _skipped()], 1),
        ([_skipped(), _measured()], 1),
        ([_skipped(), _skipped()], None),
    ],
)
def test_build_report_tolerates_mixed_runs(runs, expect_measured):
    stats = report.build_report(runs)
    if expect_measured is None:
        assert M not in stats
    else:
        assert stats[M]["measured_runs"] == expect_measured
        assert stats[M]["total_runs"] == 2
        assert stats[M]["p50"] == 2.0
    # metrics every run has are unaffected
    assert stats["add_time_s"]["measured_runs"] == 2


def test_generate_html_renders_a_dash_for_a_skipped_metric(tmp_path):
    runs = [_measured(), _skipped()]
    stats = report.build_report(runs)
    out = tmp_path / "r.html"
    report.generate_html(stats, len(runs), {"x": 1}, runs, out)
    html = out.read_text()
    assert "&mdash;" in html
    assert "0.000s" not in html, "a skipped phase must not render as a 0.000s measurement"
