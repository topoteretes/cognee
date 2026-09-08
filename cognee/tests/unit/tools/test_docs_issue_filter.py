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
def filter_module():
    return _load_module()


def test_documentation_label_as_dict_matches(filter_module):
    labels = [{"name": "Documentation"}, {"name": "needs-triage"}]
    assert filter_module.looks_like_docs_issue(labels, "Crash on start", "body") is True
    assert filter_module.docs_match_reason(labels, "Crash on start", "body") == (
        "label: documentation"
    )


def test_documentation_label_as_string_matches(filter_module):
    assert filter_module.looks_like_docs_issue(["documentation"], "x", "y") is True


def test_needs_triage_label_alone_does_not_match(filter_module):
    assert filter_module.looks_like_docs_issue([{"name": "needs-triage"}], "Crash", "body") is False


def test_docs_word_in_title_matches(filter_module):
    labels = [{"name": "bug"}]
    assert filter_module.looks_like_docs_issue(labels, "[Docs]: search page is wrong", "") is True
    assert filter_module.docs_match_reason(labels, "[Docs]: wrong", "") == "title mentions docs"


def test_documentation_word_in_body_matches(filter_module):
    body = "The documentation says X but the code does Y."
    assert filter_module.looks_like_docs_issue([], "Unexpected result", body) is True
    assert filter_module.docs_match_reason([], "Unexpected result", body) == "body mentions docs"


def test_docstring_does_not_match(filter_module):
    body = "The docstring on cognify() is out of date."
    assert filter_module.looks_like_docs_issue([], "Wrong docstring", body) is False


def test_docker_does_not_match(filter_module):
    assert (
        filter_module.looks_like_docs_issue([], "Docker compose fails", "docker-compose up")
        is False
    )


def test_none_body_and_title_do_not_crash(filter_module):
    assert filter_module.looks_like_docs_issue(None, None, None) is False
    assert filter_module.looks_like_docs_issue([{"name": "documentation"}], None, None) is True


def test_label_objects_without_name_are_ignored(filter_module):
    assert filter_module.looks_like_docs_issue([{"color": "ff0000"}, 42], "Crash", "") is False
