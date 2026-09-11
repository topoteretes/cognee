"""Unit tests for the LLM-free docs-issue filter in tools/docs_issue_filter.py."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO_ROOT / "tools" / "docs_issue_filter.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("docs_issue_filter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f():
    return _load_module()


# --- label and title ------------------------------------------------------------------------


def test_documentation_label_matches_as_dict_or_string(f):
    assert f.docs_signals([{"name": "Documentation"}], "Crash on start", "body") == [
        "label: documentation"
    ]
    assert f.looks_like_docs_issue(["documentation"], "x", "y") is True


def test_needs_triage_label_alone_does_not_match(f):
    assert f.looks_like_docs_issue([{"name": "needs-triage"}], "Crash", "body") is False


def test_docs_title_prefix_and_word(f):
    assert f.docs_signals([], "[Docs]: search page is wrong", "") == ["title prefix"]
    assert f.docs_signals([], "docs: fix quickstart", "") == ["title prefix"]
    assert f.docs_signals([], "Unclear documentation about recall", "") == ["title mentions docs"]


def test_docstring_and_docker_titles_do_not_match(f):
    assert f.looks_like_docs_issue([], "Wrong docstring on cognify()", "") is False
    assert f.looks_like_docs_issue([], "Docker compose fails", "docker-compose up") is False


# --- issue form structure -------------------------------------------------------------------

DOCS_FORM_BODY = """### Documentation Type

Unclear documentation

### Documentation Location

https://github.com/topoteretes/cognee

### Issue Description

The repo isn't very clear where to go to ask questions.

### Suggested Improvement

_No response_
"""


def test_documentation_issue_form_is_detected_without_label(f):
    signals = f.docs_signals([], "Request for comment", DOCS_FORM_BODY)
    assert "documentation issue form" in signals


def test_dropped_sections_are_not_searched(f):
    body = """### Bug Description

Cognify crashes on empty input.

### Steps to Reproduce

1. Read what the docs say about add()
2. Run cognify

### Logs/Error Messages

NoDataError: see the documentation is unclear here

### Environment

- Cognee 1.5.0
"""
    assert f.relevant_body_text(body).strip().startswith("### Bug Description")
    assert "docs say" not in f.relevant_body_text(body)
    assert f.looks_like_docs_issue([], "[Bug]: cognify crashes", body) is False


def test_hackathon_template_with_third_party_docs_link_does_not_match(f):
    body = """### Problem Statement

cognee has five data-source connectors today against roughly 25 planned.

**JIRA** is on the roadmap and has no connector.

### Proposed Solution

Add a `jira` connector following the shape of the existing connectors.

### Build it on dlt — use the REST API source

The verified source docs describe the auth flow.
https://dlthub.com/docs/dlt-ecosystem/verified-sources/jira

### Acceptance Criteria

- [ ] README and docs for the connector
- [ ] Incremental sync documented in the README

### Package layout to follow

```
packages/connector/jira/
├── README.md
```
"""
    assert (
        f.docs_signals([{"name": "hackathon"}], "Hackathon [Feature]: Add JIRA connector", body)
        == []
    )


def test_free_form_body_is_one_section(f):
    assert f.split_sections("just prose, no headings") == [(None, "just prose, no headings")]


# --- noise stripping ------------------------------------------------------------------------


def test_strip_noise_removes_code_links_quotes_and_checklists(f):
    text = (
        "The `docs` say X.\n"
        "```\nthe docs say nothing here\n```\n"
        "> quoted: the documentation is wrong\n"
        "- [ ] Documentation improvements\n"
        "See [the guide](https://docs.anthropic.com/en/docs/caching) and https://x.y/docs/z\n"
        "<!-- the docs say -->\n"
        "_No response_"
    )
    cleaned = f.strip_noise(text)
    assert "nothing here" not in cleaned
    assert "quoted" not in cleaned
    assert "improvements" not in cleaned
    assert "docs.anthropic.com" not in cleaned and "x.y/docs" not in cleaned
    assert "the guide" in cleaned  # link text survives, URL does not
    assert "<!--" not in cleaned and "No response" not in cleaned


def test_third_party_docs_url_alone_is_not_a_signal(f):
    body = "Prompt caching is described at https://docs.anthropic.com/en/docs/build-with-claude."
    assert f.docs_signals([], "Anthropic prompt caching never enabled", body) == []


# --- docs.cognee.ai link --------------------------------------------------------------------


def test_docs_site_link_is_a_positive_signal(f):
    body = (
        "The [config docs](https://docs.cognee.ai/python-api/config) list `set_graph_model`.\n"
        "Also https://docs.cognee.ai/python-api/config#setters, and https://docs.cognee.ai/x."
    )
    assert f.docs_site_urls(body) == [
        "https://docs.cognee.ai/python-api/config",
        "https://docs.cognee.ai/python-api/config#setters",
        "https://docs.cognee.ai/x",
    ]
    signals = f.docs_signals([], "[Bug]: set_graph_model is inert", body)
    assert "links https://docs.cognee.ai/python-api/config (+2)" in signals


# --- claim phrases --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The search-type documentation does not mention the prerequisite.",
        "Note the docs currently state the opposite of the PR.",
        "The config docs list set_graph_model as a working setter.",
        "It is documented as setting the graph extraction model.",
        "The behavior is completely undocumented.",
        "This prerequisite is not documented anywhere.",
        "Nothing in the current docs mentions compaction.",
        "According to the documentation, metadata=False protects the relational DB.",
        "The documentation is unclear about which flag wins.",
        "Outdated docs for the Ollama section.",
        "The docs present PGVector as supported.",
        "I'm aware the docs already acknowledge that visualization is slow.",
        "There is a typo in the docs for recall.",
        "The page has a broken link to the quickstart.",
    ],
)
def test_claim_phrases_match(f, sentence):
    assert f.find_docs_claim(sentence) is not None, sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "We should add docs for the new flag.",
        "Please update the documentation once this lands.",
        "Documentation improvements are welcome.",
        "The flag and behavior are documented in the integration README.",
        "My inclination is opt-in, documented as requiring calibration.",
        "Affects Swagger UI documentation generation.",
        "The docstring on cognify() is out of date.",
        "Docker docs are irrelevant here.",  # no claim verb, no cognee docs
    ],
)
def test_deliverable_and_unrelated_phrases_do_not_match(f, sentence):
    assert f.find_docs_claim(sentence) is None, sentence


def test_find_docs_claim_returns_earliest_match(f):
    text = "The docs are unclear. Later, the documentation says X."
    assert f.find_docs_claim(text) == "The docs are unclear"


# --- whole-issue behaviour ------------------------------------------------------------------


def test_real_docs_bug_report_matches_on_claim_and_link(f):
    body = """## Summary

`cognee.config.set_graph_model(MyModel)` is documented as setting the graph
extraction model, but it has no effect on extraction.

The [config docs](https://docs.cognee.ai) list `set_graph_model(model)` as "Set
graph extraction model".

## Steps to Reproduce

```python
cognee.config.set_graph_model(MyGraphModel)
```
"""
    signals = f.docs_signals([], "[Bug]: set_graph_model() is inert", body)
    assert signals == ["links https://docs.cognee.ai", 'claim: "is documented as"']
    assert f.docs_match_reason([], "[Bug]: set_graph_model() is inert", body) == "; ".join(signals)


def test_none_inputs_do_not_crash(f):
    assert f.looks_like_docs_issue(None, None, None) is False
    assert f.docs_match_reason(None, None, None) == "no documentation signal"
    assert f.looks_like_docs_issue([{"name": "documentation"}], None, None) is True


def test_label_objects_without_name_are_ignored(f):
    assert f.looks_like_docs_issue([{"color": "ff0000"}, 42], "Crash", "") is False
