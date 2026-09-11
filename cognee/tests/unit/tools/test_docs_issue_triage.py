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
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    _offline(module, monkeypatch)
    return module


def _fake_fetch_text(url, max_chars=None):
    """Serve the fake full-site export; anything else fails like a real 404 would."""
    import urllib.error

    if url != "https://docs.cognee.ai/llms-full.txt":
        raise urllib.error.URLError(f"unexpected fetch in unit test: {url}")
    return FAKE_EXPORT[:max_chars] if max_chars else FAKE_EXPORT


def _offline(module, monkeypatch):
    """Neither the docs site, GitHub timelines, git, an LLM nor SMTP is touched from unit tests.

    Phase 3 is switched off as a whole so phase 2 tests see phase 2 output; the ``phase3``
    fixture switches it back on with its own stubs.
    """
    monkeypatch.setattr(module, "fetch_text", _fake_fetch_text)
    monkeypatch.setattr(
        module,
        "run_site_check",
        lambda system_prompt, user_message: ("needs_source", "default fake verdict", []),
    )
    module._real_run_source_checks = module.run_source_checks
    monkeypatch.setattr(module, "run_source_checks", lambda repo, results, index, repo_root: True)


PAGE_SEARCH = "https://docs.cognee.ai/python-api/search-type"
PAGE_PRUNE = "https://docs.cognee.ai/python-api/prune"
PAGE_CONFIG = "https://docs.cognee.ai/python-api/config"
PAGE_VECTOR = "https://docs.cognee.ai/setup-configuration/vector-stores"
PAGE_QUICKSTART = "https://docs.cognee.ai/getting-started/quickstart"
PAGE_CHANGELOG = "https://docs.cognee.ai/changelog"
PAGE_RUST = "https://docs.cognee.ai/rust/operations"


def _page(title, url, text):
    return f"# {title}\nSource: {url}\n\n{text}\n\n\n"


FAKE_EXPORT = "".join(
    [
        _page(
            "Search Types",
            PAGE_SEARCH,
            "TRIPLET_COMPLETION needs TRIPLET_EMBEDDING=true at cognify time.",
        ),
        _page("Prune", PAGE_PRUNE, "prune_system(metadata=False) keeps the relational database."),
        _page("Config", PAGE_CONFIG, "set_graph_model(model): Set graph extraction model."),
        _page(
            "Vector Stores",
            PAGE_VECTOR,
            "LanceDB is the default vector store. Compaction is manual.",
        ),
        _page("Python Quickstart", PAGE_QUICKSTART, "Run your first remember and recall."),
        _page(
            "Changelog",
            PAGE_CHANGELOG,
            "TRIPLET_COMPLETION prune_system set_graph_model LanceDB recall remember " * 20,
        ),
        _page("Rust Operations", PAGE_RUST, "TRIPLET_COMPLETION in the Rust port."),
    ]
)


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

    def fake(url, method="GET", payload=None):
        calls.append(url if method == "GET" else (method, url, payload))
        for needle, response in responses:
            if needle in url:
                return response
        raise AssertionError(f"unexpected GitHub API call: {method} {url}")

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
    _offline(module, monkeypatch)
    calls = _install_fake_api(module, monkeypatch, [("/issues/7", _issue(7, "[Docs]: typo"))])
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
        "verdict": "needs_source",
        "reason": "default fake verdict",
        "signals": ["title prefix"],
        "pages_shown": [PAGE_SEARCH],
        "doc_pages": [],
        "doc_urls": [],
        "source_files": [],
        "docs_files": [],
        "commented": False,
        "comment": "",
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
    assert rows == {12: "skipped_filter", 13: "needs_source", 14: "needs_source"}


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
    assert "| Issue | Signals | Verdict | Reason | Pages shown to the LLM | Comment |" in summary
    assert "[#4604](https://github.com/topoteretes/cognee/issues/4604)" in summary
    assert "`needs_source`" in summary
    assert "default fake verdict" in summary
    assert "label: documentation" in summary  # the cheap-filter signals stay visible
    assert "- Passed the cheap filter and listed below: 1" in summary
    assert "- No documentation signal, not listed: 0" in summary


def test_http_error_returns_1(triage, monkeypatch):
    import io
    import urllib.error

    def boom(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b"{}"))

    monkeypatch.setattr(triage, "github_api_json", boom)
    assert triage.main(["--issue-number", "1"]) == 1


# --- phase 2: docs check --------------------------------------------------------------------


def _docs_issue(number=4656, body="The search-type documentation does not mention it."):
    return _issue(number, "[Docs]: TRIPLET_COMPLETION prerequisite", body=body)


def _set_verdict(triage, monkeypatch, verdict, reason="r", urls=()):
    """Stub the LLM. ``urls`` may be plain URLs or (url, note) pairs."""
    seen = []
    covering = [(u, "states the relevant fact") if isinstance(u, str) else u for u in urls]

    def fake(system_prompt, user_message):
        seen.append(user_message)
        return verdict, reason, list(covering)

    monkeypatch.setattr(triage, "run_site_check", fake)
    return seen


def test_docs_index_splits_the_export_into_pages(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    assert len(index) == 7
    assert index.pages[PAGE_SEARCH][0] == "Search Types"
    assert index.pages[PAGE_SEARCH][1].startswith("TRIPLET_COMPLETION needs")


def test_tokenize_stems_plurals_and_splits_camel_case(triage):
    assert triage.tokenize("Search Types for SearchType embeddings") == [
        "search",
        "type",
        "search",
        "type",
        "embedding",
    ]
    assert triage.tokenize("cognee docs guide") == []  # stopwords


def test_query_terms_come_from_title_and_kept_sections_only(triage):
    body = (
        "## Summary\n\nCalling `prune_system(vector=True)` wipes tables. Tables gone.\n\n"
        "### Logs/Error Messages\n\nDROP TABLE datasets cascade lancedb ignored words\n"
    )
    terms = triage.issue_query_terms("[Bug]: prune drops schema on PGVector", body)
    assert terms["prune"] == 1.0 and terms["pgvector"] == 1.0
    assert terms["table"] == 2.0  # repeated use counts, after plural stemming
    assert "prune_system" not in terms and "vector" not in terms  # inline code is stripped
    assert "lancedb" not in terms  # dropped section is not a source of query terms


def test_rank_puts_the_matching_page_first_and_changelog_is_excluded(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    query = triage.issue_query_terms("[Docs]: TRIPLET_COMPLETION prerequisite", "")
    allowed = triage.candidate_urls(index, "triplet completion prerequisite")
    assert PAGE_CHANGELOG not in allowed and PAGE_RUST not in allowed
    ranked = index.rank(query, allowed)
    assert ranked[0][1] == PAGE_SEARCH


def test_port_pages_are_candidates_only_when_the_issue_names_the_port(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    assert PAGE_RUST not in triage.candidate_urls(index, "triplet completion")
    assert PAGE_RUST in triage.candidate_urls(index, "triplet completion in rust")


def test_pick_pages_prefers_pages_the_issue_links_and_trims_weak_tail(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    ranked = [(10.0, PAGE_SEARCH), (9.0, PAGE_CONFIG), (1.0, PAGE_QUICKSTART)]
    picked = triage.pick_pages(index, ranked, ["https://docs.cognee.ai/python-api/prune.md#params"])
    assert picked == [PAGE_PRUNE, PAGE_SEARCH, PAGE_CONFIG]  # linked first; 1.0 < 0.3 * 10 dropped
    assert triage.pick_pages(index, [], ["https://docs.cognee.ai"]) == []  # site root is no page


def test_documentation_covered_posts_one_marked_comment(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [("/issues/4656/comments", []), ("/issues/4656", _docs_issue())],
    )
    seen = _set_verdict(triage, monkeypatch, "documentation_covered", "page says so", [PAGE_SEARCH])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0

    posts = [c for c in calls if isinstance(c, tuple)]
    assert len(posts) == 1
    method, url, payload = posts[0]
    assert method == "POST" and url.endswith("/issues/4656/comments")
    body = payload["body"]
    assert body.splitlines()[0] == triage.COMMENT_MARKER
    assert "This comment is auto-generated." in body
    assert f"- [Search Types]({PAGE_SEARCH}): states the relevant fact" in body
    assert "please close this issue" in body and "will not auto-close" in body

    [row] = json.loads(out.read_text())
    assert row["verdict"] == "documentation_covered"
    assert row["doc_urls"] == [PAGE_SEARCH]
    assert row["doc_pages"] == [
        {"title": "Search Types", "url": PAGE_SEARCH, "note": "states the relevant fact"}
    ]
    assert row["commented"] is True
    assert row["comment"] == body  # the row records exactly what was posted
    assert "_body" not in row
    # the LLM saw the issue, the ranked candidate list and the page text
    assert "GitHub issue #4656" in seen[0]
    assert f"- Search Types: {PAGE_SEARCH}" in seen[0]
    assert "TRIPLET_EMBEDDING=true" in seen[0]


def test_rerun_with_marker_present_does_not_comment_again(triage, monkeypatch, tmp_path):
    existing = [{"body": "human comment"}, {"body": f"{triage.COMMENT_MARKER}\nold bot comment"}]
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [
            ("/issues/4656/comments?per_page=100&page=1", existing),
            ("/issues/4656/comments?per_page=100&page=2", []),
            ("/issues/4656", _docs_issue()),
        ],
    )
    _set_verdict(triage, monkeypatch, "documentation_covered", "still answered", [PAGE_SEARCH])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    assert not [c for c in calls if isinstance(c, tuple)]
    [row] = json.loads(out.read_text())
    assert row["commented"] is False and "already present" in row["reason"]


def test_marker_must_be_its_own_line(triage):
    assert triage.has_triage_comment([{"body": f"x {triage.COMMENT_MARKER} y"}]) is False
    assert triage.has_triage_comment([{"body": f"intro\n{triage.COMMENT_MARKER}\nmore"}]) is True


def test_dry_run_never_posts(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(triage, monkeypatch, "documentation_covered", "answered", [PAGE_SEARCH])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--dry-run", "--results-json", str(out)]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    assert not any("/comments" in c for c in calls)  # no need to list comments on dry run
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "documentation_covered" and row["commented"] is False
    assert "dry run" in row["reason"]
    # the comment a live run would have posted is recorded for review
    assert row["comment"].splitlines()[0] == triage.COMMENT_MARKER
    assert f"[Search Types]({PAGE_SEARCH})" in row["comment"]
    assert "please close this issue" in row["comment"]


def test_unprovided_url_downgrades_to_needs_source(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(
        triage, monkeypatch, "documentation_covered", "cites", ["https://docs.cognee.ai/made-up"]
    )
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "needs_source" and row["doc_urls"] == []
    assert "downgraded" in row["reason"]


def test_url_match_ignores_md_suffix_and_fragment(triage, monkeypatch, tmp_path):
    _install_fake_api(
        triage, monkeypatch, [("/issues/4656/comments", []), ("/issues/4656", _docs_issue())]
    )
    _set_verdict(triage, monkeypatch, "documentation_covered", "ok", [f"{PAGE_SEARCH}.md#triplet"])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "documentation_covered" and row["doc_urls"] == [PAGE_SEARCH]


@pytest.mark.parametrize("verdict", ["not_docs", "too_vague", "needs_source"])
def test_silent_verdicts_never_post(triage, monkeypatch, tmp_path, verdict):
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(triage, monkeypatch, verdict, "why")
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    [row] = json.loads(out.read_text())
    assert row["verdict"] == verdict and row["commented"] is False and row["doc_urls"] == []


def test_unexpected_llm_verdict_is_treated_as_needs_source(triage, monkeypatch, tmp_path):
    _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(triage, monkeypatch, "small_gap", "wrong phase")
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "needs_source" and "unexpected verdict" in row["reason"]


def test_llm_failure_marks_row_uncertain_and_exits_1(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])

    def boom(system_prompt, user_message):
        raise RuntimeError("LLM site check failed: 429")

    monkeypatch.setattr(triage, "run_site_check", boom)
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 1
    assert all(not isinstance(c, tuple) for c in calls)
    [row] = json.loads(out.read_text())  # results are still written
    assert row["verdict"] == "uncertain" and "429" in row["reason"]


def test_missing_llm_key_fails_only_when_something_is_pending(triage, monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY")
    _install_fake_api(triage, monkeypatch, [("/issues/9", _issue(9, "Docker build fails"))])
    assert triage.main(["--issue-number", "9"]) == 0  # skipped_filter needs no LLM

    _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    assert triage.main(["--issue-number", "4656"]) == 1
    assert "LLM_API_KEY is not set" in capsys.readouterr().err


def test_site_check_user_message_shape(triage):
    row = triage.make_result(_docs_issue(body="x" * 10_000), "pending_docs_check", "r")
    message = triage.build_site_check_user_message(
        row, [("Search Types", PAGE_SEARCH), ("Prune", PAGE_PRUNE)], {PAGE_SEARCH: "PAGE"}
    )
    assert message.startswith("GitHub issue #4656: [Docs]: TRIPLET_COMPLETION prerequisite")
    assert "x" * 8_000 in message and "x" * 8_001 not in message  # body truncated
    assert f"- Search Types: {PAGE_SEARCH}" in message and f"- Prune: {PAGE_PRUNE}" in message
    assert f"=== {PAGE_SEARCH}\n\nPAGE" in message


def test_export_is_fetched_once_per_run_and_pages_never_individually(triage, monkeypatch, tmp_path):
    fetched = []

    def spy(url, max_chars=None):
        fetched.append(url)
        return _fake_fetch_text(url, max_chars)

    monkeypatch.setattr(triage, "fetch_text", spy)
    page_one = [
        _issue(13, "[Docs]: recall page", body="The docs say recall works."),
        _issue(14, "Unclear wording", labels=["documentation"]),
        _issue(15, "[Docs]: prune", body="prune_system is documented as safe."),
    ]
    _install_fake_api(triage, monkeypatch, [("&page=1&", page_one), ("&page=2&", [])])
    seen = _set_verdict(triage, monkeypatch, "needs_source")
    assert triage.main(["--since", "2026-08-21", "--until", "2026-08-24", "--dry-run"]) == 0
    assert fetched == ["https://docs.cognee.ai/llms-full.txt"]
    assert len(seen) == 3  # one LLM call per pending issue
    with_pages = [message for message in seen if "Fetched pages:" in message]
    assert len(with_pages) == 2  # #13 and #15 name a topic; the label-only #14 matches no page
    assert any("No page bodies were fetched" in message for message in seen)


def test_empty_export_is_an_error_not_a_silent_run(triage, monkeypatch, capsys):
    monkeypatch.setattr(triage, "fetch_text", lambda url, max_chars=None: "no pages here")
    _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    assert triage.main(["--issue-number", "4656", "--dry-run"]) == 1
    assert "contained no pages" in capsys.readouterr().err


def test_summary_lists_only_rows_that_passed_the_filter(triage, monkeypatch, tmp_path):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    page_one = [
        _issue(12, "Segfault on cognify", body="stack trace", labels=["bug"]),
        _issue(
            13, "[Docs]: TRIPLET_COMPLETION", body="The docs say TRIPLET_COMPLETION just works."
        ),
        _issue(15, "Another crash", body="no docs words here"),
    ]
    _install_fake_api(triage, monkeypatch, [("&page=1&", page_one), ("&page=2&", [])])
    _set_verdict(triage, monkeypatch, "documentation_covered", "answered", [PAGE_SEARCH])
    assert triage.main(["--since", "2026-08-21", "--until", "2026-08-24", "--dry-run"]) == 0
    summary = summary_file.read_text()
    assert "- Issues selected: 3" in summary
    assert "- Passed the cheap filter and listed below: 1" in summary
    assert "- No documentation signal, not listed: 2" in summary
    assert "#13" in summary and "#12" not in summary and "#15" not in summary
    assert "**python-api/search-type (cited)**" in summary  # pages shown, cited one marked
    assert "**suggested, not posted**<br>This comment is auto-generated." in summary
    assert triage.COMMENT_MARKER not in summary  # the marker line is left out of the table


def test_pick_pages_caps_at_four(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    ranked = [(10.0, PAGE_SEARCH), (9.0, PAGE_CONFIG), (8.0, PAGE_VECTOR), (7.0, PAGE_QUICKSTART)]
    assert triage.pick_pages(index, ranked, [f"{PAGE_PRUNE}.md"]) == [
        PAGE_PRUNE,
        PAGE_SEARCH,
        PAGE_CONFIG,
        PAGE_VECTOR,
    ]


@pytest.mark.parametrize(
    ("title", "labels"),
    [
        ("[Bug]: set_graph_model() is inert", []),
        ("[bug] prune wipes tables", []),
        ("prune wipes tables", ["bug"]),
    ],
)
def test_bug_reports_never_get_documentation_covered(triage, monkeypatch, tmp_path, title, labels):
    issue = _issue(
        4632, title, body="The config docs list set_graph_model as working.", labels=labels
    )
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4632", issue)])
    _set_verdict(
        triage, monkeypatch, "documentation_covered", "docs describe the setter", [PAGE_CONFIG]
    )
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4632", "--results-json", str(out)]) == 0
    assert all(not isinstance(c, tuple) for c in calls)  # nothing posted
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "needs_source"
    assert "filed as a bug report" in row["reason"]
    assert row["doc_urls"] == [] and row["comment"] == ""


def test_docs_form_issue_is_not_treated_as_a_bug_report(triage):
    assert (
        triage.is_bug_report("[Docs]: TRIPLET_COMPLETION prerequisite", ["documentation"]) is False
    )
    assert triage.is_bug_report("Debugging tips", []) is False  # "bug" inside a word does not count
    assert triage.is_bug_report("[Bug]: x", []) is True
    assert triage.is_bug_report("x", ["Bug"]) is True


def test_comment_links_pages_by_title_with_the_note(triage):
    comment = triage.documentation_covered_comment(
        [
            {"title": "Search Types", "url": PAGE_SEARCH, "note": "states that X needs Y"},
            {"title": "Prune", "url": PAGE_PRUNE, "note": ""},
        ]
    )
    assert (
        f"- [Search Types]({PAGE_SEARCH}): states that X needs Y\n- [Prune]({PAGE_PRUNE})\n"
        in comment
    )


# --- phase 3: source check, human review, draft-docs hand-off ------------------------------


@pytest.fixture
def phase3(triage, monkeypatch):
    """Phase 3 on, offline: no git grep, no timeline/comment lookups, no email, LLM stubbed."""
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


def _set_source_verdict(phase3, monkeypatch, verdict, reason="r", source=(), docs=()):
    seen = []

    def fake(system_prompt, user_message):
        seen.append(user_message)
        return verdict, reason, list(source), list(docs)

    monkeypatch.setattr(phase3, "run_source_check", fake)
    return seen


def _needs_source_issue(number=4632, title="[Bug]: set_graph_model() is inert", labels=()):
    return _issue(
        number,
        title,
        body="`cognee.config.set_graph_model(MyModel)` is documented as setting the model. "
        "`GRAPH_DATABASE_PROVIDER` and SearchType.GRAPH_COMPLETION are unrelated.",
        labels=labels,
    )


def test_issue_identifiers_are_code_like_tokens_longest_first(phase3):
    tokens = phase3.issue_identifiers(
        "[Bug]: HUGGINGFACE_TOKENIZER ignored",
        "See `resolve_embedding_tokenizer` and `create_embedding_engine`; `x` is too short; "
        "SearchType.TRIPLET_COMPLETION works. `two words` is skipped.",
    )
    assert tokens[:2] == ["SearchType.TRIPLET_COMPLETION", "resolve_embedding_tokenizer"]
    assert tokens == sorted(tokens, key=len, reverse=True)
    assert "HUGGINGFACE_TOKENIZER" in tokens and "create_embedding_engine" in tokens
    assert "x" not in tokens and "two words" not in tokens
    assert len(tokens) == len(set(tokens)) <= phase3.GREP_MAX_TOKENS


def test_git_grep_hits_runs_one_fixed_string_grep(triage, monkeypatch, tmp_path):
    calls = []

    class Completed:
        returncode = 1  # no match
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        return Completed()

    monkeypatch.setattr(triage.subprocess, "run", fake_run)
    assert triage.git_grep_hits(["prune_system", "PGVector"], tmp_path) == ""
    [command] = calls
    assert command[:5] == ["git", "-C", str(tmp_path), "grep", "-I"]
    assert command.count("-e") == 2 and "-F" in command
    assert "--" in command and "cognee" in command and ":(exclude)cognee/tests" in command
    assert triage.git_grep_hits([], tmp_path) == "" and len(calls) == 1  # no tokens, no grep


def test_git_grep_failure_other_than_no_match_raises(triage, monkeypatch, tmp_path):
    class Completed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(triage.subprocess, "run", lambda *a, **k: Completed())
    with pytest.raises(RuntimeError, match="git grep failed"):
        triage.git_grep_hits(["x"], tmp_path)


def test_small_gap_becomes_matrix_row_and_needs_no_email(phase3, monkeypatch, tmp_path):
    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    seen = _set_source_verdict(
        phase3,
        monkeypatch,
        "small_gap",
        "setter is inert, config page should say so",
        source=["cognee/api/v1/config/config.py"],
        docs=[
            "python-api/config.mdx",
            "https://docs.cognee.ai/not-a-path",
            "a/b.mdx",
            "c.mdx",
            "d.mdx",
        ],
    )
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "small_gap"
    assert row["source_files"] == ["cognee/api/v1/config/config.py"]
    assert row["docs_files"] == ["python-api/config.mdx", "a/b.mdx", "c.mdx"]  # URL dropped, cap 3
    assert row["comment"] == ""  # the gap comment is posted by the draft-docs job, not here
    # the source LLM saw the issue, the docs pages already checked, and the grep hits
    assert "git grep hits" in seen[0] and "cognee/x.py:1:hit" in seen[0]

    output = output_file.read_text()
    assert "has_gaps=true\n" in output
    [matrix] = [
        json.loads(line.split("=", 1)[1])
        for line in output.splitlines()
        if line.startswith("matrix=")
    ]
    assert matrix == [
        {
            "number": "4632",
            "title": "[Bug]: set_graph_model() is inert",
            "body_b64": matrix[0]["body_b64"],
            "source_files": "cognee/api/v1/config/config.py",
            "docs_files": "python-api/config.mdx a/b.mdx c.mdx",
        }
    ]
    import base64

    excerpt = base64.b64decode(matrix[0]["body_b64"]).decode()
    assert excerpt.startswith("[Bug]: set_graph_model() is inert\n\n`cognee.config")


def test_dry_run_keeps_small_gap_out_of_the_matrix(phase3, monkeypatch, tmp_path):
    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    _set_source_verdict(phase3, monkeypatch, "small_gap", "gap", ["cognee/a.py"], ["guides/a.mdx"])
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--dry-run", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "small_gap"  # still visible in the summary and artifact
    assert (
        "has_gaps=false\n" in output_file.read_text() and "matrix=[]\n" in output_file.read_text()
    )


def test_small_gap_without_docs_file_is_uncertain(phase3, monkeypatch, tmp_path):
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    _set_source_verdict(phase3, monkeypatch, "small_gap", "gap", ["cognee/a.py"], [])
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--dry-run", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "uncertain" and "no existing docs file" in row["reason"]
    assert row["source_files"] == [] and row["docs_files"] == []


@pytest.mark.parametrize("verdict", ["not_in_source", "too_big", "uncertain"])
def test_human_review_verdicts_are_silent_and_listed(phase3, monkeypatch, tmp_path, verdict):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    calls = _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    _set_source_verdict(phase3, monkeypatch, verdict, "why")
    assert phase3.main(["--issue-number", "4632"]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    summary = summary_file.read_text()
    assert "### Human review" in summary and f"`{verdict}`" in summary
    assert "email skipped, SMTP not configured" in summary


def test_open_linked_pull_request_means_fix_in_progress(phase3, monkeypatch, tmp_path):
    monkeypatch.setattr(
        phase3,
        "linked_pull_requests",
        lambda repo, number: ["https://github.com/topoteretes/cognee/pull/5000"],
    )
    llm_calls = _set_source_verdict(phase3, monkeypatch, "small_gap", "gap", ["a.py"], ["b.mdx"])
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "fix_in_progress"
    assert "open linked pull request" in row["reason"] and "pull/5000" in row["reason"]
    assert llm_calls == []  # no second LLM call when a fix is already under way


def test_only_maintainer_volunteers_mean_fix_in_progress(phase3, monkeypatch, tmp_path):
    comments = [
        {"body": f"{phase3.COMMENT_MARKER}\nbot text: working on this", "user": {"login": "bot"}},
        {
            "body": "Hi, I'd like to work on this issue!",
            "user": {"login": "newcomer"},
            "author_association": "CONTRIBUTOR",
        },
    ]
    monkeypatch.setattr(phase3, "list_issue_comments", lambda repo, number: comments)
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "uncertain"  # the drive-by volunteer did not stop the source check

    comments.append(
        {
            "body": "Working on this.",
            "user": {"login": "coreperson"},
            "author_association": "MEMBER",
        }
    )
    assert phase3.main(["--issue-number", "4632", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "fix_in_progress" and "maintainer @coreperson" in row["reason"]


def test_core_team_file_marks_volunteers_as_maintainers(phase3, monkeypatch, tmp_path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "core-team.txt").write_text("# logins\n@Milenko\nlxobr\n")
    assert phase3.core_team_logins(tmp_path) == {"milenko", "lxobr"}
    assert phase3.core_team_logins(tmp_path / "nowhere") == set()
    comments = [
        {"body": "Linked the fix PR.", "user": {"login": "milenko"}, "author_association": "NONE"}
    ]
    monkeypatch.setattr(phase3, "list_issue_comments", lambda repo, number: comments)
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    out = tmp_path / "r.json"
    assert (
        phase3.main(
            ["--issue-number", "4632", "--repo-root", str(tmp_path), "--results-json", str(out)]
        )
        == 0
    )
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "fix_in_progress" and "@milenko" in row["reason"]


def test_linked_pull_requests_reads_cross_reference_events(triage, monkeypatch):
    timeline = [
        {"event": "labeled"},
        {"event": "cross-referenced", "source": {"issue": {"html_url": "https://x/issues/9"}}},
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
    _install_fake_api(
        triage, monkeypatch, [("timeline?per_page=100&page=1", timeline), ("&page=2", [])]
    )
    assert triage.linked_pull_requests("o/r", 4632) == ["https://x/pull/10"]  # closed #11 ignored


def test_human_review_email_is_sent_when_smtp_is_configured(phase3, monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, server, port, timeout):
            sent["server"] = (server, port)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = user

        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr(phase3.smtplib, "SMTP", FakeSMTP)
    for name in phase3.SMTP_ENV:
        monkeypatch.setenv(name, f"{name.lower()}@example.test")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "topoteretes/cognee")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    rows = [
        {
            "number": 4632,
            "title": "[Bug]: x | y",
            "html_url": "https://github.com/topoteretes/cognee/issues/4632",
            "verdict": "uncertain",
            "reason": "r",
        }
    ]
    status = phase3.send_human_review_email(rows, dry_run=True)
    assert status.startswith("emailed 1 human-review item(s)")
    assert sent["server"] == ("smtp_server@example.test", 587) and sent["tls"] is True
    message = sent["message"]
    assert message["Subject"] == "docs issue triage: 1 item(s) need human review"
    body = message.get_content()
    assert "DRY RUN" in body and "actions/runs/123" in body
    assert "| #4632 [Bug]: x \\| y | `uncertain` | r |" in body
    assert "https://github.com/topoteretes/cognee/issues/4632" in body


def test_email_is_skipped_without_smtp_and_never_raises(phase3):
    status = phase3.send_human_review_email([{"number": 1}], dry_run=False)
    assert status.startswith("1 item(s) need human review; email skipped")
    assert phase3.send_human_review_email([], dry_run=False) == "no items need human review"


def test_post_gap_comment_posts_once_and_never_leaks_urls(triage, monkeypatch):
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [("/issues/4632/comments?per_page=100&page=1", []), ("/issues/4632/comments", [])],
    )
    assert triage.main(["--post-gap-comment", "4632"]) == 0
    posts = [c for c in calls if isinstance(c, tuple)]
    assert len(posts) == 1
    body = posts[0][2]["body"]
    assert body.splitlines()[0] == triage.COMMENT_MARKER
    assert "a documentation change may be prepared" in body
    assert "github.com/topoteretes/cognee-docs" not in body  # never the private PR
    assert "docs.cognee.ai" not in body and "/pull/" not in body


def test_post_gap_comment_respects_marker_and_dry_run(triage, monkeypatch, capsys):
    existing = [{"body": f"{triage.COMMENT_MARKER}\nold"}]
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [
            ("/issues/4632/comments?per_page=100&page=1", existing),
            ("/issues/4632/comments?per_page=100&page=2", []),
        ],
    )
    assert triage.main(["--post-gap-comment", "4632"]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    assert "already carries the bot comment" in capsys.readouterr().out

    calls = _install_fake_api(triage, monkeypatch, [])
    assert triage.main(["--post-gap-comment", "4632", "--dry-run"]) == 0
    assert calls == []  # dry run does not even list comments
    assert "not posted" in capsys.readouterr().out


def test_post_gap_comment_ignores_selectors_and_llm(triage, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    _install_fake_api(triage, monkeypatch, [("/issues/7/comments", [])])
    assert triage.main(["--post-gap-comment", "7", "--since", "2020-01-01"]) == 0


def test_source_check_failure_marks_uncertain_and_exits_1(phase3, monkeypatch, tmp_path):
    def boom(system_prompt, user_message):
        raise RuntimeError("LLM source check failed: 500")

    monkeypatch.setattr(phase3, "run_source_check", boom)
    _install_fake_api(phase3, monkeypatch, [("/issues/4632", _needs_source_issue())])
    out = tmp_path / "r.json"
    assert phase3.main(["--issue-number", "4632", "--dry-run", "--results-json", str(out)]) == 1
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "uncertain" and "500" in row["reason"]


def test_documentation_covered_rows_skip_the_source_check(phase3, monkeypatch, tmp_path):
    _install_fake_api(phase3, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(phase3, monkeypatch, "documentation_covered", "covered", [PAGE_SEARCH])
    source_calls = _set_source_verdict(phase3, monkeypatch, "small_gap")
    assert phase3.main(["--issue-number", "4656", "--dry-run"]) == 0
    assert source_calls == []
