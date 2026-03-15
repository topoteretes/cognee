import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.minimax.adapter import (
    MiniMaxAdapter,
    MINIMAX_DEFAULT_BASE_URL,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    LLMProvider,
)


class TestMiniMaxAdapter:
    """Tests for the MiniMax LLM adapter."""

    def test_minimax_provider_enum_exists(self):
        """MiniMax should be registered in the LLMProvider enum."""
        assert LLMProvider.MINIMAX.value == "minimax"

    def test_adapter_creation_with_defaults(self):
        """MiniMaxAdapter should initialize with correct defaults."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5",
            max_completion_tokens=16384,
        )
        assert adapter.api_key == "test-key"
        # litellm requires the "openai/" prefix for OpenAI-compatible endpoints
        assert adapter.model == "openai/MiniMax-M2.5"
        assert adapter.endpoint == MINIMAX_DEFAULT_BASE_URL
        assert adapter.name == "MiniMax"
        assert adapter.max_completion_tokens == 16384

    def test_adapter_creation_with_custom_endpoint(self):
        """MiniMaxAdapter should use custom endpoint when provided."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5",
            max_completion_tokens=16384,
            endpoint="https://api.minimaxi.com/v1",
        )
        assert adapter.endpoint == "https://api.minimaxi.com/v1"

    def test_adapter_model_prefix(self):
        """MiniMaxAdapter should add openai/ prefix if not present."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5-highspeed",
            max_completion_tokens=16384,
        )
        assert adapter.model == "openai/MiniMax-M2.5-highspeed"

    def test_adapter_model_prefix_not_duplicated(self):
        """MiniMaxAdapter should not double-add openai/ prefix."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="openai/MiniMax-M2.5",
            max_completion_tokens=16384,
        )
        assert adapter.model == "openai/MiniMax-M2.5"

    def test_default_instructor_mode(self):
        """MiniMaxAdapter should use json_mode as default instructor mode."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5",
            max_completion_tokens=16384,
        )
        assert adapter.instructor_mode == "json_mode"

    def test_custom_instructor_mode(self):
        """MiniMaxAdapter should accept custom instructor mode."""
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5",
            max_completion_tokens=16384,
            instructor_mode="json_schema_mode",
        )
        assert adapter.instructor_mode == "json_schema_mode"

    def test_default_base_url_value(self):
        """Default base URL should point to MiniMax international endpoint."""
        assert MINIMAX_DEFAULT_BASE_URL == "https://api.minimax.io/v1"

    def test_llm_args_passed_through(self):
        """LLM args should be stored on the adapter."""
        llm_args = {"top_p": 0.9}
        adapter = MiniMaxAdapter(
            api_key="test-key",
            model="MiniMax-M2.5",
            max_completion_tokens=16384,
            llm_args=llm_args,
        )
        assert adapter.llm_args == {"top_p": 0.9}


class TestMiniMaxProviderRouting:
    """Tests for MiniMax provider routing in get_llm_client."""

    @patch("cognee.infrastructure.llm.config.LLMConfig")
    def test_minimax_provider_value(self, mock_config):
        """LLMProvider enum should include minimax."""
        provider = LLMProvider("minimax")
        assert provider == LLMProvider.MINIMAX
