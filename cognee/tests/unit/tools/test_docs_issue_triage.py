"""Unit tests for the phase-1 (read-only) behaviour of tools/docs_issue_triage.py."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "tools" / "docs_issue_triage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("docs_issue_triage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def triage(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    return module


def _issue(
    number, title, body="", labels=(), state="open", created_at="2026-08-22T10:00:00Z", pr=False
):
    payload = {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": name} for name in labels],
        "state": state,
        "created_at": created_at,
        "html_url": f"https://github.com/topoteretes/cognee/issues/{number}",
    }
    if pr:
        payload["pull_request"] = {"url": f"https://api.github.com/repos/x/y/pulls/{number}"}
    return payload


def _install_fake_api(triage, monkeypatch, responses):
    """Replace github_api_json with a lookup on URL substring; record every call."""
    calls = []

    def fake(url):
        calls.append(url)
        for needle, payload in responses:
            if needle in url:
                return payload
        raise AssertionError(f"unexpected GitHub API call: {url}")

    monkeypatch.setattr(triage, "github_api_json", fake)
    return calls


# --- selector validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--issue-number", "", "--since", "", "--until", ""],
        ["--since", "2026-01-01"],
        ["--until", "2026-01-02"],
        ["--since", "2026-01-05", "--until", "2026-01-02"],
        ["--since", "01/01/2026", "--until", "2026-01-02"],
        ["--issue-number", "abc"],
    ],
)
def test_bad_selectors_exit_2_before_any_request(triage, monkeypatch, argv):
    calls = _install_fake_api(triage, monkeypatch, [])
    with pytest.raises(SystemExit) as excinfo:
        triage.main(argv)
    assert excinfo.value.code == 2
    assert calls == []


def test_issue_number_wins_over_date_range(triage, monkeypatch):
    calls = _install_fake_api(
        triage, monkeypatch, [("/issues/1", _issue(1, "[Docs]: quickstart typo"))]
    )
    assert (
        triage.main(["--issue-number", "1", "--since", "2026-01-01", "--until", "2026-01-02"]) == 0
    )
    assert len(calls) == 1
    assert calls[0].endswith("/repos/topoteretes/cognee/issues/1")


def test_repo_defaults_to_github_repository_env(triage, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/fork")
    module = _load_module()
    calls = _install_fake_api(module, monkeypatch, [("/issues/7", _issue(7, "docs"))])
    assert module.main(["--issue-number", "7"]) == 0
    assert "/repos/someone/fork/issues/7" in calls[0]


# --- single issue ---------------------------------------------------------------------------


def test_single_pull_request_is_skipped(triage, monkeypatch, tmp_path):
    _install_fake_api(triage, monkeypatch, [("/issues/5", _issue(5, "docs: fix", pr=True))])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "5", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "skipped_pr"


def test_single_closed_issue_is_skipped(triage, monkeypatch, tmp_path):
    _install_fake_api(
        triage, monkeypatch, [("/issues/6", _issue(6, "[Docs]: gone", state="closed"))]
    )
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "6", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "skipped_closed"


def test_single_docs_issue_is_pending_and_has_spec_shape(triage, monkeypatch, tmp_path):
    _install_fake_api(
        triage,
        monkeypatch,
        [("/issues/4656", _issue(4656, "[Docs]: TRIPLET_COMPLETION needs memify", body=None))],
    )
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--dry-run", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row == {
        "number": 4656,
        "title": "[Docs]: TRIPLET_COMPLETION needs memify",
        "html_url": "https://github.com/topoteretes/cognee/issues/4656",
        "verdict": "pending_docs_check",
        "reason": "title mentions docs",
        "doc_urls": [],
        "source_files": [],
        "docs_files": [],
        "commented": False,
    }


def test_single_non_docs_issue_is_filtered(triage, monkeypatch, tmp_path):
    _install_fake_api(
        triage, monkeypatch, [("/issues/9", _issue(9, "Docker build fails", body="docstring"))]
    )
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "9", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "skipped_filter"


# --- date range -----------------------------------------------------------------------------


def test_date_range_filters_prs_window_and_docs_words(triage, monkeypatch, tmp_path):
    page_one = [
        _issue(10, "docs: PR not issue", pr=True),
        _issue(11, "[Docs]: created too early", created_at="2026-08-20T23:59:59Z"),
        _issue(12, "Segfault on cognify", body="stack trace", labels=["bug"]),
        _issue(13, "[Docs]: recall page", created_at="2026-08-24T23:59:59Z"),
        _issue(14, "Unclear wording", labels=["documentation"], created_at="2026-08-21T00:00:00Z"),
    ]
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [("&page=1&", page_one), ("&page=2&", [])],
    )
    out = tmp_path / "r.json"
    assert (
        triage.main(["--since", "2026-08-21", "--until", "2026-08-24", "--results-json", str(out)])
        == 0
    )
    assert len(calls) == 2
    assert "state=open" in calls[0] and "per_page=100" in calls[0]

    rows = {row["number"]: row["verdict"] for row in json.loads(out.read_text())}
    assert rows == {
        12: "skipped_filter",
        13: "pending_docs_check",
        14: "pending_docs_check",
    }


# --- GitHub Actions outputs -----------------------------------------------------------------


def test_github_output_and_step_summary_are_written(triage, monkeypatch, tmp_path):
    output_file = tmp_path / "output.txt"
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    _install_fake_api(
        triage, monkeypatch, [("/issues/4604", _issue(4604, "Unclear", labels=["documentation"]))]
    )

    assert triage.main(["--issue-number", "4604", "--dry-run"]) == 0

    output = output_file.read_text()
    assert "has_gaps=false\n" in output
    assert "matrix=[]\n" in output

    summary = summary_file.read_text()
    assert "## Docs issue triage" in summary
    assert "issue #4604" in summary
    assert "| Issue | Verdict | Reason | Commented |" in summary
    assert "[#4604](https://github.com/topoteretes/cognee/issues/4604)" in summary
    assert "`pending_docs_check`" in summary
    assert "label: documentation" in summary


def test_http_error_returns_1(triage, monkeypatch):
    import io
    import urllib.error

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b"{}"))

    monkeypatch.setattr(triage, "github_api_json", boom)
    assert triage.main(["--issue-number", "1"]) == 1
