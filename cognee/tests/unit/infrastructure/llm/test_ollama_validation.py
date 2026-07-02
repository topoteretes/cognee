import logging
import pytest
from unittest.mock import patch, AsyncMock
from cognee.infrastructure.llm.config import (
    normalize_model_name,
    classify_model,
    emit_warning,
    LLMConfig,
)


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
    from cognee.infrastructure.llm.config import _warned_models

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


@pytest.mark.asyncio
async def test_local_ollama_example_executes_offline():
    # Mock all cognee operations used in the example
    with (
        patch("cognee.prune.prune_data", new_callable=AsyncMock) as mock_prune_data,
        patch("cognee.prune.prune_system", new_callable=AsyncMock) as mock_prune_system,
        patch("cognee.add", new_callable=AsyncMock) as mock_add,
        patch("cognee.cognify", new_callable=AsyncMock) as mock_cognify,
        patch(
            "cognee.search", new_callable=AsyncMock, return_value=["Mocked answer"]
        ) as mock_search,
    ):
        from examples.python.local_ollama_example import main

        await main()

        mock_prune_data.assert_called_once()
        mock_prune_system.assert_called_once_with(metadata=True)
        mock_add.assert_called_once()
        mock_cognify.assert_called_once_with(datasets=["ollama_local_demo"])
        mock_search.assert_called_once()
