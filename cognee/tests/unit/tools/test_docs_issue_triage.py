"""Behavioural tests for tools/docs_issue_triage.py.

Everything here guards a public action or a hard rule: what may be posted, when, how
often, what must never leak, and what happens when a dependency fails. Nothing touches
the network: GitHub, the docs export, git, the LLM and SMTP are all stubbed.
"""

import base64
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "tools" / "docs_issue_triage.py"

PAGE_SEARCH = "https://docs.cognee.ai/python-api/search-type"
PAGE_CONFIG = "https://docs.cognee.ai/python-api/config"
FAKE_EXPORT = (
    f"# Search Types\nSource: {PAGE_SEARCH}\n\n"
    "TRIPLET_COMPLETION needs TRIPLET_EMBEDDING=true at cognify time.\n\n\n"
    f"# Config\nSource: {PAGE_CONFIG}\n\nset_graph_model(model): Set graph extraction model.\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("docs_issue_triage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def triage(monkeypatch):
    """The module with phase 2 offline and phase 3 switched off (see ``phase3``)."""
    module = _load_module()
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(module, "fetch_text", lambda url, max_chars=None: FAKE_EXPORT)
    monkeypatch.setattr(
        module,
        "run_site_check",
        lambda system_prompt, user_message: ("needs_source", "default site verdict", []),
    )
    module._real_run_source_checks = module.run_source_checks
    monkeypatch.setattr(module, "run_source_checks", lambda repo, results, index, root: True)
    return module


@pytest.fixture
def phase3(triage, monkeypatch):
    """Phase 3 on, offline: git grep, timeline and comment lookups, SMTP and the LLM stubbed."""
    monkeypatch.setattr(triage, "run_source_checks", triage._real_run_source_checks)
    monkeypatch.setattr(triage, "git_grep_hits", lambda tokens, repo_root: "cognee/x.py:1:hit")
    monkeypatch.setattr(triage, "linked_pull_requests", lambda repo, number: [])
    monkeypatch.setattr(triage, "list_issue_comments", lambda repo, number: [])
    monkeypatch.setattr(
        triage,
        "run_source_check",
        lambda system_prompt, user_message: ("uncertain", "default source verdict", [], []),
    )
    for name in triage.SMTP_ENV + ("SMTP_PORT", "SMTP_USE_TLS"):
        monkeypatch.delenv(name, raising=False)
    return triage


def _issue(number, title, body="", labels=(), state="open", created="2026-08-22T10:00:00Z"):
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [{"name": name} for name in labels],
        "state": state,
        "created_at": created,
        "html_url": f"https://github.com/topoteretes/cognee/issues/{number}",
    }


def _docs_issue(number=4656, title="[Docs]: TRIPLET_COMPLETION prerequisite", **kw):
    return _issue(number, title, body="The search-type documentation does not mention it.", **kw)


def _fake_api(triage, monkeypatch, responses):
    """github_api_json stub: first URL-substring match wins; every call is recorded."""
    calls = []

    def fake(url, method="GET", payload=None):
        calls.append(url if method == "GET" else (method, url, payload))
        listing = "/comments" in url or "/timeline" in url
        for needle, response in responses:
            # an "/issues/N" needle is for the issue itself, not its comment/timeline pages
            if needle in url and (
                not listing or any(k in needle for k in ("comments", "timeline", "page="))
            ):
                return response
        if method == "GET" and listing:
            return []  # no comments unless a test says otherwise; also ends pagination
        raise AssertionError(f"unexpected GitHub API call: {method} {url}")

    monkeypatch.setattr(triage, "github_api_json", fake)
    return calls


def _posts(calls):
    return [c for c in calls if isinstance(c, tuple)]


def _site_verdict(triage, monkeypatch, verdict, reason="r", pages=()):
    covering = [(url, "states the relevant fact") for url in pages]
    monkeypatch.setattr(triage, "run_site_check", lambda sp, um: (verdict, reason, covering))


def _source_verdict(phase3, monkeypatch, verdict, reason="r", source=(), docs=()):
    seen = []

    def fake(system_prompt, user_message):
        seen.append(user_message)
        return verdict, reason, list(source), list(docs)

    monkeypatch.setattr(phase3, "run_source_check", fake)
    return seen


def _run(triage, argv, tmp_path):
    out = tmp_path / "results.json"
    code = triage.main([*argv, "--results-json", str(out)])
    return code, json.loads(out.read_text())


# --- selection ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [[], ["--since", "2026-01-01"], ["--since", "2026-01-05", "--until", "2026-01-02"]],
)
def test_missing_or_invalid_selector_exits_2_before_any_request(triage, monkeypatch, argv):
    calls = _fake_api(triage, monkeypatch, [])
    with pytest.raises(SystemExit) as excinfo:
        triage.main(argv)
    assert excinfo.value.code == 2 and calls == []


def test_date_range_keeps_open_issues_created_in_window_and_drops_pull_requests(
    triage, monkeypatch, tmp_path
):
    pr = _issue(10, "[Docs]: a PR")
    pr["pull_request"] = {}
    page_one = [
        pr,
        _docs_issue(11, created="2026-08-20T23:59:59Z"),  # before the window
        _issue(12, "Segfault on cognify", body="stack trace", labels=["bug"]),
        _docs_issue(13, created="2026-08-24T23:59:59Z"),
    ]
    _fake_api(triage, monkeypatch, [("&page=1&", page_one), ("&page=2&", [])])
    code, rows = _run(triage, ["--since", "2026-08-21", "--until", "2026-08-24"], tmp_path)
    assert code == 0
    assert {r["number"]: r["verdict"] for r in rows} == {12: "skipped_filter", 13: "needs_source"}


def test_single_closed_issue_or_pull_request_is_skipped(triage, monkeypatch, tmp_path):
    pr = _docs_issue(5)
    pr["pull_request"] = {}
    _fake_api(
        triage, monkeypatch, [("/issues/5", pr), ("/issues/6", _docs_issue(6, state="closed"))]
    )
    assert _run(triage, ["--issue-number", "5"], tmp_path)[1][0]["verdict"] == "skipped_pr"
    assert _run(triage, ["--issue-number", "6"], tmp_path)[1][0]["verdict"] == "skipped_closed"


# --- phase 2: the documentation_covered comment ---------------------------------------------


def test_documentation_covered_posts_one_comment_linking_pages_by_title(
    triage, monkeypatch, tmp_path
):
    calls = _fake_api(triage, monkeypatch, [("/comments", []), ("/issues/4656", _docs_issue())])
    _site_verdict(triage, monkeypatch, "documentation_covered", "page says so", [PAGE_SEARCH])
    code, [row] = _run(triage, ["--issue-number", "4656"], tmp_path)
    assert code == 0
    [(method, url, payload)] = _posts(calls)
    assert method == "POST" and url.endswith("/issues/4656/comments")
    body = payload["body"]
    assert body.splitlines()[0] == triage.COMMENT_MARKER
    assert f"- [Search Types]({PAGE_SEARCH}): states the relevant fact" in body
    assert "please close this issue" in body and "will not auto-close" in body
    assert row["verdict"] == "documentation_covered" and row["commented"] is True
    assert row["comment"] == body and row["doc_urls"] == [PAGE_SEARCH]


def test_rerun_with_marker_present_does_not_comment_again(triage, monkeypatch, tmp_path):
    existing = [{"body": "human comment"}, {"body": f"{triage.COMMENT_MARKER}\nold bot comment"}]
    calls = _fake_api(
        triage,
        monkeypatch,
        [
            ("comments?per_page=100&page=1", existing),
            ("&page=2", []),
            ("/issues/4656", _docs_issue()),
        ],
    )
    _site_verdict(triage, monkeypatch, "documentation_covered", "still", [PAGE_SEARCH])
    code, [row] = _run(triage, ["--issue-number", "4656"], tmp_path)
    assert code == 0 and _posts(calls) == []
    assert row["commented"] is False and "already present" in row["reason"]
    # the marker only counts on a line of its own
    assert triage.has_triage_comment([{"body": f"x {triage.COMMENT_MARKER} y"}]) is False


def test_dry_run_never_posts_but_records_the_suggested_comment(triage, monkeypatch, tmp_path):
    calls = _fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _site_verdict(triage, monkeypatch, "documentation_covered", "covered", [PAGE_SEARCH])
    code, [row] = _run(triage, ["--issue-number", "4656", "--dry-run"], tmp_path)
    assert code == 0 and _posts(calls) == []
    assert row["commented"] is False and "dry run" in row["reason"]
    assert row["comment"].splitlines()[0] == triage.COMMENT_MARKER
    assert f"[Search Types]({PAGE_SEARCH})" in row["comment"]


def test_cited_page_not_shown_to_the_llm_downgrades_to_needs_source(triage, monkeypatch, tmp_path):
    calls = _fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _site_verdict(
        triage, monkeypatch, "documentation_covered", "cites", ["https://docs.cognee.ai/made-up"]
    )
    code, [row] = _run(triage, ["--issue-number", "4656"], tmp_path)
    assert code == 0 and _posts(calls) == []
    assert (
        row["verdict"] == "needs_source" and row["doc_urls"] == [] and "downgraded" in row["reason"]
    )


@pytest.mark.parametrize(
    ("title", "labels"),
    [("[Bug]: set_graph_model() is inert", []), ("prune wipes tables", ["bug"])],
)
def test_bug_reports_never_get_documentation_covered(triage, monkeypatch, tmp_path, title, labels):
    issue = _issue(
        4632, title, body="The config docs list set_graph_model as working.", labels=labels
    )
    calls = _fake_api(triage, monkeypatch, [("/issues/4632", issue)])
    _site_verdict(triage, monkeypatch, "documentation_covered", "docs describe it", [PAGE_CONFIG])
    code, [row] = _run(triage, ["--issue-number", "4632"], tmp_path)
    assert code == 0 and _posts(calls) == []
    assert row["verdict"] == "needs_source" and "filed as a bug report" in row["reason"]
    assert row["comment"] == ""


@pytest.mark.parametrize("verdict", ["not_docs", "too_vague", "needs_source"])
def test_silent_verdicts_never_post(triage, monkeypatch, tmp_path, verdict):
    calls = _fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _site_verdict(triage, monkeypatch, verdict, "why")
    code, [row] = _run(triage, ["--issue-number", "4656"], tmp_path)
    assert code == 0 and _posts(calls) == []
    assert row["verdict"] == verdict and row["comment"] == ""


def test_llm_failure_marks_row_uncertain_writes_results_and_exits_1(triage, monkeypatch, tmp_path):
    calls = _fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])

    def boom(system_prompt, user_message):
        raise RuntimeError("LLM site check failed: 429")

    monkeypatch.setattr(triage, "run_site_check", boom)
    code, [row] = _run(triage, ["--issue-number", "4656"], tmp_path)
    assert code == 1 and _posts(calls) == []
    assert row["verdict"] == "uncertain" and "429" in row["reason"]


def test_missing_llm_key_fails_only_when_an_issue_needs_the_llm(triage, monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY")
    _fake_api(triage, monkeypatch, [("/issues/9", _issue(9, "Docker build fails"))])
    assert triage.main(["--issue-number", "9"]) == 0
    _fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    assert triage.main(["--issue-number", "4656"]) == 1
    assert "LLM_API_KEY is not set" in capsys.readouterr().err


def test_docs_export_is_fetched_once_per_run_and_never_per_issue(triage, monkeypatch, tmp_path):
    fetched = []
    monkeypatch.setattr(
        triage, "fetch_text", lambda url, max_chars=None: fetched.append(url) or FAKE_EXPORT
    )
    _fake_api(
        triage, monkeypatch, [("&page=1&", [_docs_issue(13), _docs_issue(15)]), ("&page=2&", [])]
    )
    code, rows = _run(
        triage, ["--since", "2026-08-21", "--until", "2026-08-24", "--dry-run"], tmp_path
    )
    assert code == 0 and len(rows) == 2
    assert fetched == ["https://docs.cognee.ai/llms-full.txt"]


def test_summary_lists_only_issues_that_passed_the_filter(triage, monkeypatch, tmp_path):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    page_one = [
        _issue(12, "Segfault", body="trace"),
        _docs_issue(13),
        _issue(15, "Crash", body="x"),
    ]
    _fake_api(triage, monkeypatch, [("&page=1&", page_one), ("&page=2&", [])])
    _site_verdict(triage, monkeypatch, "documentation_covered", "covered", [PAGE_SEARCH])
    assert (
        _run(triage, ["--since", "2026-08-21", "--until", "2026-08-24", "--dry-run"], tmp_path)[0]
        == 0
    )
    summary = summary_file.read_text()
    assert "- Issues selected: 3" in summary
    assert "- Passed the cheap filter and listed below: 1" in summary
    assert "- No documentation signal, not listed: 2" in summary
    assert "#13" in summary and "#12" not in summary and "#15" not in summary
    assert "**suggested, not posted**" in summary and triage.COMMENT_MARKER not in summary


def test_maintainer_reply_leaves_the_thread_to_them_without_an_llm_call(
    triage, monkeypatch, tmp_path
):
    llm_calls = []
    monkeypatch.setattr(
        triage,
        "run_site_check",
        lambda sp, um: llm_calls.append(um) or ("documentation_covered", "x", [(PAGE_SEARCH, "n")]),
    )
    comments = [
        {"body": "Hello, thanks!", "user": {"login": "github-actions[bot]", "type": "Bot"}},
        {"body": "Same problem here.", "user": {"login": "someone"}, "author_association": "NONE"},
    ]
    calls = _fake_api(
        triage,
        monkeypatch,
        [("comments?per_page=100&page=1", comments), ("/issues/4604", _docs_issue(4604))],
    )
    code, [row] = _run(triage, ["--issue-number", "4604", "--dry-run"], tmp_path)
    assert code == 0 and len(llm_calls) == 1  # bot and contributor comments do not count

    comments.append(
        {
            "body": "Thanks for the questions, here are the answers.",
            "user": {"login": "lxobr"},
            "author_association": "MEMBER",
            "created_at": "2026-08-27T09:00:00Z",
        }
    )
    code, [row] = _run(triage, ["--issue-number", "4604"], tmp_path)
    assert code == 0 and len(llm_calls) == 1  # no second LLM call
    assert row["verdict"] == "maintainer_replied"
    assert row["reason"] == "maintainer @lxobr replied on 2026-08-27; left to them"
    assert row["comment"] == "" and _posts(calls) == []


# --- phase 3: source check, human review, draft-docs hand-off ------------------------------


def _bug_issue():
    return _issue(
        4632,
        "[Bug]: set_graph_model() is inert",
        body="`set_graph_model` is documented as setting the model but `GRAPH_MODEL` is unread.",
    )


def test_small_gap_becomes_a_flat_string_matrix_row_only_on_live_runs(
    phase3, monkeypatch, tmp_path
):
    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    _fake_api(phase3, monkeypatch, [("/issues/4632", _bug_issue())])
    _source_verdict(
        phase3,
        monkeypatch,
        "small_gap",
        "config page should say so",
        source=["cognee/api/v1/config/config.py"],
        docs=[
            "python-api/config.mdx",
            "https://docs.cognee.ai/not-a-path",
            "a.mdx",
            "b.mdx",
            "c.mdx",
        ],
    )
    code, [row] = _run(phase3, ["--issue-number", "4632"], tmp_path)
    assert code == 0
    assert row["verdict"] == "small_gap" and row["comment"] == ""
    assert row["docs_files"] == ["python-api/config.mdx", "a.mdx", "b.mdx"]  # URL dropped, cap 3
    output = output_file.read_text()
    assert "has_gaps=true\n" in output
    [matrix] = [
        json.loads(line.split("=", 1)[1])
        for line in output.splitlines()
        if line.startswith("matrix=")
    ]
    assert (
        matrix[0]["number"] == "4632"
        and matrix[0]["docs_files"] == "python-api/config.mdx a.mdx b.mdx"
    )
    assert base64.b64decode(matrix[0]["body_b64"]).decode().startswith("[Bug]: set_graph_model()")

    output_file.write_text("")
    code, [row] = _run(phase3, ["--issue-number", "4632", "--dry-run"], tmp_path)
    assert code == 0 and row["verdict"] == "small_gap"  # still visible in the summary
    assert (
        "has_gaps=false\n" in output_file.read_text() and "matrix=[]\n" in output_file.read_text()
    )


def test_small_gap_without_an_existing_docs_file_is_uncertain(phase3, monkeypatch, tmp_path):
    _fake_api(phase3, monkeypatch, [("/issues/4632", _bug_issue())])
    _source_verdict(phase3, monkeypatch, "small_gap", "gap", ["cognee/a.py"], [])
    code, [row] = _run(phase3, ["--issue-number", "4632", "--dry-run"], tmp_path)
    assert code == 0 and row["verdict"] == "uncertain" and row["docs_files"] == []


@pytest.mark.parametrize("verdict", ["not_in_source", "too_big", "uncertain"])
def test_human_review_verdicts_are_silent_and_listed_without_smtp(
    phase3, monkeypatch, tmp_path, verdict
):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    calls = _fake_api(phase3, monkeypatch, [("/issues/4632", _bug_issue())])
    _source_verdict(phase3, monkeypatch, verdict, "why")
    assert _run(phase3, ["--issue-number", "4632"], tmp_path)[0] == 0 and _posts(calls) == []
    summary = summary_file.read_text()
    assert "### Human review" in summary and f"`{verdict}`" in summary
    assert "email skipped, SMTP not configured" in summary


def test_open_linked_pull_request_means_fix_in_progress_without_an_llm_call(
    phase3, monkeypatch, tmp_path
):
    _fake_api(phase3, monkeypatch, [("/issues/4632", _bug_issue())])
    llm_calls = _source_verdict(phase3, monkeypatch, "small_gap", "gap", ["a.py"], ["b.mdx"])
    monkeypatch.setattr(
        phase3, "linked_pull_requests", lambda repo, number: ["https://x/pull/5000"]
    )
    _, [row] = _run(phase3, ["--issue-number", "4632"], tmp_path)
    assert row["verdict"] == "fix_in_progress" and "pull/5000" in row["reason"]
    assert llm_calls == []
    # a contributor's "I'd like to work on this" is not a fix in progress
    monkeypatch.setattr(phase3, "linked_pull_requests", lambda repo, number: [])
    comments = [
        {
            "body": "I'd like to work on this!",
            "user": {"login": "newcomer"},
            "author_association": "CONTRIBUTOR",
        }
    ]
    monkeypatch.setattr(phase3, "list_issue_comments", lambda repo, number: comments)
    _, [row] = _run(phase3, ["--issue-number", "4632"], tmp_path)
    assert row["verdict"] == "small_gap"


def test_closed_pull_requests_do_not_count_as_a_fix_in_progress(triage, monkeypatch):
    timeline = [
        {
            "event": "cross-referenced",
            "source": {
                "issue": {"html_url": "https://x/pull/10", "pull_request": {}, "state": "open"}
            },
        },
        {
            "event": "cross-referenced",
            "source": {
                "issue": {"html_url": "https://x/pull/11", "pull_request": {}, "state": "closed"}
            },
        },
    ]
    _fake_api(triage, monkeypatch, [("timeline?per_page=100&page=1", timeline), ("&page=2", [])])
    assert triage.linked_pull_requests("o/r", 4632) == ["https://x/pull/10"]


def test_human_review_email_goes_out_when_smtp_is_configured(phase3, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, server, port, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr(phase3.smtplib, "SMTP", FakeSMTP)
    for name in phase3.SMTP_ENV:
        monkeypatch.setenv(name, f"{name.lower()}@example.test")
    rows = [
        {
            "number": 4632,
            "title": "[Bug]: x",
            "html_url": "https://github.com/x/issues/4632",
            "verdict": "uncertain",
            "reason": "r",
        }
    ]
    assert phase3.send_human_review_email(rows, dry_run=True).startswith("emailed 1 human-review")
    body = sent["message"].get_content()
    assert sent["message"]["Subject"] == "docs issue triage: 1 item(s) need human review"
    assert "DRY RUN" in body and "https://github.com/x/issues/4632" in body
    assert phase3.send_human_review_email([], dry_run=False) == "no items need human review"


def test_post_gap_comment_posts_once_never_leaks_urls_and_respects_dry_run(
    triage, monkeypatch, capsys
):
    calls = _fake_api(triage, monkeypatch, [("/issues/4632/comments", [])])
    assert triage.main(["--post-gap-comment", "4632"]) == 0
    [(_method, _url, payload)] = _posts(calls)
    body = payload["body"]
    assert body.splitlines()[0] == triage.COMMENT_MARKER
    assert "a documentation change may be prepared" in body
    assert "github.com/topoteretes/cognee-docs" not in body and "docs.cognee.ai" not in body

    existing = [{"body": f"{triage.COMMENT_MARKER}\nold"}]
    calls = _fake_api(
        triage, monkeypatch, [("comments?per_page=100&page=1", existing), ("&page=2", [])]
    )
    assert triage.main(["--post-gap-comment", "4632"]) == 0 and _posts(calls) == []

    calls = _fake_api(triage, monkeypatch, [])
    assert triage.main(["--post-gap-comment", "4632", "--dry-run"]) == 0 and calls == []
    assert "not posted" in capsys.readouterr().out


def test_source_check_failure_marks_uncertain_and_exits_1(phase3, monkeypatch, tmp_path):
    def boom(system_prompt, user_message):
        raise RuntimeError("LLM source check failed: 500")

    monkeypatch.setattr(phase3, "run_source_check", boom)
    _fake_api(phase3, monkeypatch, [("/issues/4632", _bug_issue())])
    code, [row] = _run(phase3, ["--issue-number", "4632", "--dry-run"], tmp_path)
    assert code == 1 and row["verdict"] == "uncertain" and "500" in row["reason"]


def test_documentation_covered_rows_skip_the_source_check(phase3, monkeypatch, tmp_path):
    _fake_api(phase3, monkeypatch, [("/issues/4656", _docs_issue())])
    _site_verdict(phase3, monkeypatch, "documentation_covered", "covered", [PAGE_SEARCH])
    source_calls = _source_verdict(phase3, monkeypatch, "small_gap")
    assert _run(phase3, ["--issue-number", "4656", "--dry-run"], tmp_path)[0] == 0
    assert source_calls == []
