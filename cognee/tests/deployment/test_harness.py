"""Docker-free self-tests for the deployment harness mock LLM (T10, issue #3368).

These validate the in-process OpenAI-compatible mock without requiring Docker,
so the ``deployment`` marker always has fast, dependency-light coverage that runs
on every PR (including forks) regardless of container availability.
"""

from __future__ import annotations

import json

import pytest

from cognee.tests.deployment.mock_llm import (
    GOLDEN_ANSWER,
    GOLDEN_ENTITY,
    build_chat_completion_content,
    build_embeddings_response,
)

pytestmark = pytest.mark.deployment


def test_knowledge_graph_schema_returns_golden_entity():
    req = {"response_format": {"json_schema": {"name": "KnowledgeGraph"}}}
    content = json.loads(build_chat_completion_content(req))
    assert any(node["name"] == GOLDEN_ENTITY for node in content["nodes"])


def test_answer_schema_returns_golden_answer():
    req = {"response_format": {"json_schema": {"name": "Answer"}}}
    content = json.loads(build_chat_completion_content(req))
    assert content["answer"] == GOLDEN_ANSWER


def test_schema_name_resolves_from_tools_payload():
    req = {"tools": [{"function": {"name": "Answer"}}]}
    content = json.loads(build_chat_completion_content(req))
    assert content["answer"] == GOLDEN_ANSWER


def test_plain_completion_returns_golden_answer():
    assert build_chat_completion_content({}) == GOLDEN_ANSWER


def test_embeddings_dimensions_by_model():
    small = build_embeddings_response({"model": "text-embedding-3-small", "input": ["a", "b"]})
    assert len(small["data"]) == 2
    assert len(small["data"][0]["embedding"]) == 1536

    large = build_embeddings_response({"model": "text-embedding-3-large", "input": ["a"]})
    assert len(large["data"][0]["embedding"]) == 3072
