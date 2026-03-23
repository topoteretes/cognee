from types import SimpleNamespace
from unittest.mock import patch

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    get_llm_client,
)


def test_custom_provider_passes_endpoint_and_api_version_to_generic_adapter():
    llm_config = SimpleNamespace(
        llm_provider="custom",
        llm_model="lm_studio/qwen/qwen3.5:9b",
        llm_api_key="sk-lm-test",
        llm_endpoint="http://127.0.0.1:1234/v1",
        llm_api_version="2024-01-01",
        llm_max_completion_tokens=8192,
        llm_instructor_mode="json_schema_mode",
        fallback_api_key="",
        fallback_endpoint="",
        fallback_model="",
        llm_args={},
    )

    with patch(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_config",
        return_value=llm_config,
    ), patch(
        "cognee.infrastructure.llm.utils.get_model_max_completion_tokens",
        return_value=None,
    ), patch(
        "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter.GenericAPIAdapter"
    ) as mock_generic_adapter:
        sentinel_adapter = object()
        mock_generic_adapter.return_value = sentinel_adapter

        adapter = get_llm_client()

        assert adapter is sentinel_adapter
        kwargs = mock_generic_adapter.call_args.kwargs
        assert kwargs["api_key"] == "sk-lm-test"
        assert kwargs["endpoint"] == "http://127.0.0.1:1234/v1"
        assert kwargs["api_version"] == "2024-01-01"
        assert kwargs["model"] == "lm_studio/qwen/qwen3.5:9b"
        assert kwargs["name"] == "Custom"
