"""Behavioural tests for the LLM-free docs-issue filter in tools/docs_issue_filter.py."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "tools" / "docs_issue_filter.py"


@pytest.fixture(scope="module")
def f():
    spec = importlib.util.spec_from_file_location("docs_issue_filter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_title_and_form_are_signals_but_lookalike_words_are_not(f):
    assert f.docs_signals([{"name": "Documentation"}], "Crash on start", "") == [
        "label: documentation"
    ]
    assert f.docs_signals([], "[Docs]: search page is wrong", "") == ["title prefix"]
    assert f.docs_signals([], "Unclear documentation about recall", "") == ["title mentions docs"]
    assert f.looks_like_docs_issue([{"name": "needs-triage"}], "Crash", "body") is False
    assert f.looks_like_docs_issue([], "Wrong docstring on cognify()", "") is False
    assert f.looks_like_docs_issue([], "Docker compose fails", "docker-compose up") is False
    assert f.looks_like_docs_issue(None, None, None) is False


def test_documentation_issue_form_is_detected_without_label(f):
    body = (
        "### Documentation Type\n\nUnclear documentation\n\n"
        "### Documentation Location\n\nhttps://github.com/topoteretes/cognee\n\n"
        "### Issue Description\n\nThe repo isn't very clear where to go to ask questions.\n"
    )
    assert "documentation issue form" in f.docs_signals([], "Request for comment", body)


def test_only_problem_sections_are_searched(f):
    body = (
        "### Bug Description\n\nCognify crashes on empty input.\n\n"
        "### Steps to Reproduce\n\n1. Read what the docs say about add()\n\n"
        "### Logs/Error Messages\n\nNoDataError: the documentation is unclear here\n"
    )
    assert f.looks_like_docs_issue([], "[Bug]: cognify crashes", body) is False


def test_hackathon_template_with_third_party_docs_link_does_not_match(f):
    body = (
        "### Problem Statement\n\ncognee has five data-source connectors today.\n\n"
        "### Proposed Solution\n\nAdd a `jira` connector.\n\n"
        "### Build it on dlt — use the REST API source\n\n"
        "The verified source docs describe the auth flow.\n"
        "https://dlthub.com/docs/dlt-ecosystem/verified-sources/jira\n\n"
        "### Acceptance Criteria\n\n- [ ] README and docs for the connector\n"
    )
    assert f.docs_signals([{"name": "hackathon"}], "Hackathon [Feature]: Add JIRA", body) == []
    third_party = "Caching is described at https://docs.anthropic.com/en/docs/build-with-claude."
    assert f.docs_signals([], "Anthropic prompt caching never enabled", third_party) == []


def test_docs_site_link_and_claim_phrase_are_signals(f):
    body = (
        "`set_graph_model` is documented as setting the graph extraction model, but it has "
        "no effect. The [config docs](https://docs.cognee.ai/python-api/config) list it.\n\n"
        "## Steps to Reproduce\n\n```python\ncognee.config.set_graph_model(M)\n```\n"
    )
    assert f.docs_signals([], "[Bug]: set_graph_model() is inert", body) == [
        "links https://docs.cognee.ai/python-api/config",
        'claim: "is documented as"',
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "The search-type documentation does not mention the prerequisite.",
        "Note the docs currently state the opposite of the PR.",
        "It is documented as setting the graph extraction model.",
        "This prerequisite is not documented anywhere.",
        "Nothing in the current docs mentions compaction.",
        "The documentation is unclear about which flag wins.",
        "I'm aware the docs already acknowledge that visualization is slow.",
    ],
)
def test_claim_phrases_match(f, sentence):
    assert f.find_docs_claim(sentence) is not None, sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "We should add docs for the new flag.",
        "Documentation improvements are welcome.",
        "The flag and behavior are documented in the integration README.",
        "My inclination is opt-in, documented as requiring calibration.",
        "Affects Swagger UI documentation generation.",
    ],
)
def test_deliverable_phrases_do_not_match(f, sentence):
    assert f.find_docs_claim(sentence) is None, sentence
