import pytest
from unittest.mock import patch, MagicMock
from cognee.exceptions import CogneeConfigurationError
from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig


def test_embedding_config_defaults_to_openai_with_openai_llm():
    """Verify that if LLM provider is openai (or unset), embedding settings default to openai."""
    mock_llm_config = MagicMock()
    mock_llm_config.llm_provider = "openai"

    with patch(
        "cognee.infrastructure.llm.config.get_llm_context_config", return_value=mock_llm_config
    ):
        config = EmbeddingConfig()
        assert config.embedding_provider == "openai"
        assert config.embedding_model == "openai/text-embedding-3-large"
        assert config.embedding_dimensions == 3072


def test_embedding_config_raises_error_with_non_openai_llm_and_no_embedding_settings():
    """Verify that a CogneeConfigurationError is raised if LLM is non-OpenAI and no embedding settings are present."""
    mock_llm_config = MagicMock()
    mock_llm_config.llm_provider = "ollama"

    with patch(
        "cognee.infrastructure.llm.config.get_llm_context_config", return_value=mock_llm_config
    ):
        with pytest.raises(CogneeConfigurationError, match="no embedding provider is configured"):
            EmbeddingConfig()


def test_embedding_config_raises_error_on_unresolvable_dimensions():
    """Verify that a CogneeConfigurationError is raised if dimensions are unknown and not set explicitly."""
    mock_llm_config = MagicMock()
    mock_llm_config.llm_provider = "openai"

    with patch(
        "cognee.infrastructure.llm.config.get_llm_context_config", return_value=mock_llm_config
    ):
        with pytest.raises(
            CogneeConfigurationError, match="Could not auto-derive embedding dimensions"
        ):
            EmbeddingConfig(
                embedding_provider="custom_provider",
                embedding_model="custom_model_without_known_dimensions",
            )


def test_embedding_config_alias_resolution(monkeypatch):
    """Verify that EMBEDDING_MAX_TOKENS is resolved as an alias for embedding_max_completion_tokens."""
    monkeypatch.setenv("EMBEDDING_MAX_TOKENS", "4096")
    mock_llm_config = MagicMock()
    mock_llm_config.llm_provider = "openai"

    with patch(
        "cognee.infrastructure.llm.config.get_llm_context_config", return_value=mock_llm_config
    ):
        config = EmbeddingConfig()
        assert config.embedding_max_completion_tokens == 4096
