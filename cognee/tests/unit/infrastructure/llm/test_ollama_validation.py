import logging
import os
import pytest
from unittest.mock import patch, AsyncMock
from cognee.infrastructure.llm.ollama_support import (
    normalize_model_name,
    classify_model,
    emit_warning,
    _warned_models,
)
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.LLMGateway import LLMGateway


@pytest.mark.parametrize(
    "raw_input,expected_normalized,expected_classification,expected_warning",
    [
        ("ollama/llama3.1:8b", "llama3.1:8b", "recommended", False),
        ("llama3.2:latest", "llama3.2:latest", "recommended", False),
        ("mistral:7b", "mistral:7b", "problematic", True),
        ("qwen2.5:7b", "qwen2.5:7b", "problematic", True),
        ("qwen2.5:14b", "qwen2.5:14b", "recommended", False),
        ("random-model", "random-model", "unknown", True),
    ],
)
def test_ollama_model_validation_matrix(
    raw_input, expected_normalized, expected_classification, expected_warning, caplog
):
    # Test normalization function
    normalized = normalize_model_name(raw_input)
    assert normalized == expected_normalized

    # Test classification function
    classification = classify_model(normalized)
    assert classification == expected_classification

    # Clear warned list and caplog to avoid test pollution
    _warned_models.clear()
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        emit_warning(classification, raw_input)

    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]

    if expected_warning:
        assert len(warnings) == 1
        assert raw_input in warnings[0].message
    else:
        assert len(warnings) == 0


def test_ollama_warning_emitted_on_config_startup(caplog):
    """Test that instantiating LLMConfig with Ollama and a problematic model emits warning."""
    _warned_models.clear()
    caplog.clear()

    with (
        patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "mistral:7b",
                "LLM_ENDPOINT": "http://localhost:11434/v1",
                "LLM_API_KEY": "ollama",
            },
        ),
        caplog.at_level(logging.WARNING),
    ):
        LLMConfig()
        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "mistral:7b" in warnings[0].message
        assert "limitations" in warnings[0].message


def test_non_ollama_provider_does_not_warn(caplog):
    """Test that instantiating LLMConfig with another provider does not warn even if model matches."""
    _warned_models.clear()
    caplog.clear()

    with (
        patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "mistral:7b",
                "LLM_API_KEY": "openai-key",
            },
        ),
        caplog.at_level(logging.WARNING),
    ):
        LLMConfig()
        warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
        assert len(warnings) == 0


@pytest.mark.asyncio
async def test_local_ollama_example_executes_offline(monkeypatch):
    """Verify that local_ollama_example.py runs using the mock LLM/embedding harness."""
    monkeypatch.setenv("MOCK_EMBEDDING", "true")

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
        from cognee.shared.data_models import Node, Edge

        if response_model is KnowledgeGraph or (
            isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph)
        ):
            return KnowledgeGraph(
                nodes=[Node(id="cognee", name="cognee", type="Library", description="AI memory")],
                edges=[
                    Edge(source_node_id="cognee", target_node_id="cognee", relationship_name="is")
                ],
            )
        if response_model is SummarizedContent or (
            isinstance(response_model, type) and issubclass(response_model, SummarizedContent)
        ):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "Mocked answer"
        return response_model()

    with patch.object(LLMGateway, "acreate_structured_output", new=_mock_acreate):
        from examples.python.local_ollama_example import main

        await main()
