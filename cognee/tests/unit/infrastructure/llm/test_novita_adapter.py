import pytest

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.novita.adapter import (
    NovitaAdapter,
    NOVITA_DEFAULT_ENDPOINT,
    NOVITA_DEFAULT_MODEL,
)


def test_novita_adapter_initialization():
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model="moonshotai/kimi-k2.5",
        max_completion_tokens=4096,
    )

    assert adapter.name == "Novita"
    assert adapter.model == "moonshotai/kimi-k2.5"
    assert adapter.api_key == "test_api_key"
    assert adapter.endpoint == NOVITA_DEFAULT_ENDPOINT
    assert adapter.max_completion_tokens == 4096


def test_novita_adapter_custom_endpoint():
    custom_endpoint = "https://custom.novita.endpoint/v1"
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model="moonshotai/kimi-k2.5",
        max_completion_tokens=4096,
        endpoint=custom_endpoint,
    )

    assert adapter.endpoint == custom_endpoint


def test_novita_adapter_default_model():
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model=NOVITA_DEFAULT_MODEL,
        max_completion_tokens=4096,
    )

    assert adapter.model == NOVITA_DEFAULT_MODEL


def test_novita_adapter_with_fallback():
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model="moonshotai/kimi-k2.5",
        max_completion_tokens=4096,
        fallback_api_key="fallback_key",
        fallback_endpoint="https://fallback.endpoint",
        fallback_model="zai-org/glm-5",
    )

    assert adapter.fallback_api_key == "fallback_key"
    assert adapter.fallback_endpoint == "https://fallback.endpoint"
    assert adapter.fallback_model == "zai-org/glm-5"


def test_novita_adapter_instructor_mode():
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model="moonshotai/kimi-k2.5",
        max_completion_tokens=4096,
        instructor_mode="json_mode",
    )

    assert adapter.instructor_mode == "json_mode"


def test_novita_adapter_default_instructor_mode():
    adapter = NovitaAdapter(
        api_key="test_api_key",
        model="moonshotai/kimi-k2.5",
        max_completion_tokens=4096,
    )

    assert adapter.instructor_mode == NovitaAdapter.default_instructor_mode
