"""Lightweight harness self-tests (no Docker required)."""

import json

import httpx
import pytest

from cognee.tests.deployment.helpers import wait_for_health
from cognee.tests.deployment.mock_llm import (
    GOLDEN_ANSWER,
    GOLDEN_ENTITY,
    build_chat_completion_content,
    build_embeddings_response,
    start_mock_llm_server,
    stop_mock_llm_server,
)


@pytest.mark.deployment
def test_mock_llm_schema_responses():
    kg_content = build_chat_completion_content(
        {
            "response_format": {"json_schema": {"name": "KnowledgeGraph"}},
        }
    )
    kg = json.loads(kg_content)
    assert any(node["name"] == GOLDEN_ENTITY for node in kg["nodes"])

    summary_content = build_chat_completion_content(
        {
            "response_format": {"json_schema": {"name": "SummarizedContent"}},
        }
    )
    summary = json.loads(summary_content)
    assert "summary" in summary
    assert GOLDEN_ENTITY in summary["summary"]

    plain_answer = build_chat_completion_content({})
    assert plain_answer == GOLDEN_ANSWER


@pytest.mark.deployment
def test_mock_llm_embeddings():
    resp = build_embeddings_response({"model": "text-embedding-3-small", "input": ["hello"]})
    assert len(resp["data"]) == 1
    assert len(resp["data"][0]["embedding"]) == 1536


@pytest.mark.deployment
def test_mock_llm_http_server():
    server, _thread, port = start_mock_llm_server()
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "test"}],
                "response_format": {"json_schema": {"name": "KnowledgeGraph"}},
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        assert GOLDEN_ENTITY in content
    finally:
        stop_mock_llm_server(server)


@pytest.mark.deployment
def test_wait_for_health_times_out_on_dead_port():
    with pytest.raises(TimeoutError):
        wait_for_health("http://127.0.0.1:1/health", timeout=2.0)
