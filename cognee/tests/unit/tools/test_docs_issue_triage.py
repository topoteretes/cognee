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
    """Phase 2 network: neither the docs site nor an LLM is ever touched from unit tests."""
    monkeypatch.setattr(module, "fetch_text", _fake_fetch_text)
    monkeypatch.setattr(
        module,
        "run_site_check",
        lambda system_prompt, user_message: ("needs_source", "default fake verdict", []),
    )


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
    assert "| Issue | Signals | Verdict | Reason | Pages shown to the LLM | Commented |" in summary
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
    seen = []

    def fake(system_prompt, user_message):
        seen.append(user_message)
        return verdict, reason, list(urls)

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


def test_already_answered_posts_one_marked_comment(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(
        triage,
        monkeypatch,
        [("/issues/4656/comments", []), ("/issues/4656", _docs_issue())],
    )
    seen = _set_verdict(triage, monkeypatch, "already_answered", "page says so", [PAGE_SEARCH])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0

    posts = [c for c in calls if isinstance(c, tuple)]
    assert len(posts) == 1
    method, url, payload = posts[0]
    assert method == "POST" and url.endswith("/issues/4656/comments")
    body = payload["body"]
    assert body.splitlines()[0] == triage.COMMENT_MARKER
    assert "This comment is auto-generated." in body
    assert f"- {PAGE_SEARCH}" in body
    assert "please close this issue" in body and "will not auto-close" in body

    [row] = json.loads(out.read_text())
    assert row["verdict"] == "already_answered"
    assert row["doc_urls"] == [PAGE_SEARCH]
    assert row["commented"] is True
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
    _set_verdict(triage, monkeypatch, "already_answered", "still answered", [PAGE_SEARCH])
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
    _set_verdict(triage, monkeypatch, "already_answered", "answered", [PAGE_SEARCH])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--dry-run", "--results-json", str(out)]) == 0
    assert all(not isinstance(c, tuple) for c in calls)
    assert not any("/comments" in c for c in calls)  # no need to list comments on dry run
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "already_answered" and row["commented"] is False
    assert "dry run" in row["reason"]


def test_unprovided_url_downgrades_to_needs_source(triage, monkeypatch, tmp_path):
    calls = _install_fake_api(triage, monkeypatch, [("/issues/4656", _docs_issue())])
    _set_verdict(
        triage, monkeypatch, "already_answered", "cites", ["https://docs.cognee.ai/made-up"]
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
    _set_verdict(triage, monkeypatch, "already_answered", "ok", [f"{PAGE_SEARCH}.md#triplet"])
    out = tmp_path / "r.json"
    assert triage.main(["--issue-number", "4656", "--results-json", str(out)]) == 0
    [row] = json.loads(out.read_text())
    assert row["verdict"] == "already_answered" and row["doc_urls"] == [PAGE_SEARCH]


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
    _set_verdict(triage, monkeypatch, "already_answered", "answered", [PAGE_SEARCH])
    assert triage.main(["--since", "2026-08-21", "--until", "2026-08-24", "--dry-run"]) == 0
    summary = summary_file.read_text()
    assert "- Issues selected: 3" in summary
    assert "- Passed the cheap filter and listed below: 1" in summary
    assert "- No documentation signal, not listed: 2" in summary
    assert "#13" in summary and "#12" not in summary and "#15" not in summary
    assert "**python-api/search-type (cited)**" in summary  # pages shown, cited one marked


def test_pick_pages_caps_at_four(triage):
    index = triage.DocsIndex(FAKE_EXPORT)
    ranked = [(10.0, PAGE_SEARCH), (9.0, PAGE_CONFIG), (8.0, PAGE_VECTOR), (7.0, PAGE_QUICKSTART)]
    assert triage.pick_pages(index, ranked, [f"{PAGE_PRUNE}.md"]) == [
        PAGE_PRUNE,
        PAGE_SEARCH,
        PAGE_CONFIG,
        PAGE_VECTOR,
    ]
