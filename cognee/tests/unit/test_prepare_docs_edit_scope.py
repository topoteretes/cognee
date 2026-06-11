from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "prepare_docs_edit_scope.py"
SPEC = importlib.util.spec_from_file_location("prepare_docs_edit_scope", MODULE_PATH)
prepare_docs_edit_scope = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(prepare_docs_edit_scope)


def test_get_pr_changed_files_reads_paginated_github_files(monkeypatch):
    calls = []

    def fake_github_api_json(url):
        calls.append(url)
        if url.endswith("page=1"):
            return [{"filename": f"cognee/file_{index}.py"} for index in range(100)]
        if url.endswith("page=2"):
            return [{"filename": "examples/python/references_example.py"}]
        return []

    monkeypatch.setattr(prepare_docs_edit_scope, "github_api_json", fake_github_api_json)

    changed_files = prepare_docs_edit_scope.get_pr_changed_files("topoteretes/cognee", 123)

    assert len(changed_files) == 101
    assert changed_files[0] == "cognee/file_0.py"
    assert changed_files[-1] == "examples/python/references_example.py"
    assert len(calls) == 2


def test_collect_changed_files_uses_pr_files(monkeypatch):
    monkeypatch.setattr(
        prepare_docs_edit_scope,
        "get_pr_changed_files",
        lambda repo, pr_number: ["cognee/modules/retrieval/references.py"],
    )

    changed_files = prepare_docs_edit_scope.collect_changed_files(
        Namespace(
            repo="topoteretes/cognee",
            pr_number=123,
        )
    )

    assert changed_files == ["cognee/modules/retrieval/references.py"]


def test_collect_changed_files_propagates_pr_file_errors(monkeypatch):
    def fake_get_pr_changed_files(repo, pr_number):
        raise RuntimeError("GitHub API unavailable")

    monkeypatch.setattr(prepare_docs_edit_scope, "get_pr_changed_files", fake_get_pr_changed_files)

    with pytest.raises(RuntimeError, match="GitHub API unavailable"):
        prepare_docs_edit_scope.collect_changed_files(
            Namespace(
                repo="topoteretes/cognee",
                pr_number=123,
            )
        )


def test_parse_args_requires_pr_inputs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_docs_edit_scope.py"])

    with pytest.raises(SystemExit):
        prepare_docs_edit_scope.parse_args()


def test_candidate_doc_files_are_ranked_from_pr_changed_files(tmp_path):
    docs_root = tmp_path / "docs-repo"
    (docs_root / "cognee-cloud/functionality").mkdir(parents=True)
    (docs_root / "core-concepts/search").mkdir(parents=True)
    (docs_root / "unrelated").mkdir(parents=True)
    (docs_root / "cognee-cloud/functionality/search-and-recall.mdx").write_text(
        "Search and recall",
        encoding="utf-8",
    )
    (docs_root / "core-concepts/search/retrieval.mdx").write_text(
        "Retrieval",
        encoding="utf-8",
    )
    (docs_root / "unrelated/overview.mdx").write_text("Overview", encoding="utf-8")

    keywords = prepare_docs_edit_scope.collect_keywords(
        [
            "cognee/api/v1/search/search.py",
            "cognee/api/v1/recall/recall.py",
        ]
    )
    candidates = prepare_docs_edit_scope.get_candidate_doc_files(docs_root, keywords)

    assert candidates[:2] == [
        "cognee-cloud/functionality/search-and-recall.mdx",
        "core-concepts/search/retrieval.mdx",
    ]


def test_build_scope_payload_prioritizes_public_files_and_excludes_tests():
    changed_files = [
        "cognee/tests/unit/test_search.py",
        "cognee/api/v1/search/search.py",
        "cognee/modules/retrieval/completion_retriever.py",
        "uv.lock",
        "examples/python/references_example.py",
    ]
    notes = {
        "summary": "Adds references to search responses.",
        "user_impact": "Users can parse optional references.",
        "documentation_signals": ["Search response documentation should mention references."],
    }
    assessment = {
        "needs_documentation_update": True,
        "reason": "Search API responses changed.",
        "candidate_areas": ["Search API docs"],
        "recommended_next_steps": ["Update search docs"],
    }

    payload = prepare_docs_edit_scope.build_scope_payload(
        123,
        changed_files,
        ["cognee-cloud/functionality/search-and-recall.mdx"],
        notes,
        assessment,
    )

    assert payload["source_files_to_inspect"] == [
        "cognee/api/v1/search/search.py",
        "examples/python/references_example.py",
        "cognee/modules/retrieval/completion_retriever.py",
    ]
    assert "cognee/tests/unit/test_search.py" in payload["out_of_scope_files"]
    assert "uv.lock" in payload["out_of_scope_files"]
    assert payload["candidate_doc_files"] == ["cognee-cloud/functionality/search-and-recall.mdx"]


def test_write_scope_files_writes_markdown_and_json(tmp_path):
    payload = {
        "pr_number": 123,
        "changed_files_count": 1,
        "needs_documentation_update": True,
        "assessment_reason": "Search response changed.",
        "notes_summary": "Adds references.",
        "notes_user_impact": "Users see references.",
        "source_file_classifications": [
            {"path": "cognee/api/v1/search/search.py", "classification": "public API surface"}
        ],
        "candidate_doc_files": ["cognee-cloud/functionality/search-and-recall.mdx"],
        "documentation_signals": ["Search docs"],
        "assessment_candidate_areas": ["Search API docs"],
        "assessment_recommended_next_steps": ["Update response shape"],
        "out_of_scope_files": [],
    }
    json_output = tmp_path / "scope.json"
    markdown_output = tmp_path / "scope.md"

    prepare_docs_edit_scope.write_scope_files(payload, json_output, markdown_output)

    assert json_output.exists()
    assert json.loads(json_output.read_text())["pr_number"] == 123
    markdown = markdown_output.read_text()
    assert "# Prepared Documentation Edit Scope" in markdown
    assert "`cognee/api/v1/search/search.py`" in markdown
