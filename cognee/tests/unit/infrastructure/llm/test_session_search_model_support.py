import importlib
from types import SimpleNamespace

import pytest
from cognee.infrastructure.session.session_search_models import (
    get_session_search_completion_model,
)

gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")


@pytest.mark.parametrize("framework", ["instructor", "litellm_native"])
def test_supported_framework_accepts_session_completion_model(monkeypatch, framework):
    monkeypatch.setattr(
        gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework=framework),
    )

    assert gateway_module.LLMGateway.supports_structured_output_model(
        get_session_search_completion_model(str)
    )


@pytest.mark.parametrize("framework", ["baml", "unknown"])
def test_unsupported_framework_rejects_session_completion_model(monkeypatch, framework):
    monkeypatch.setattr(
        gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework=framework),
    )

    assert not gateway_module.LLMGateway.supports_structured_output_model(
        get_session_search_completion_model(str)
    )


def test_capability_check_rejects_non_pydantic_models(monkeypatch):
    monkeypatch.setattr(
        gateway_module,
        "get_llm_config",
        lambda: SimpleNamespace(structured_output_framework="instructor"),
    )

    assert not gateway_module.LLMGateway.supports_structured_output_model(str)
    assert not gateway_module.LLMGateway.supports_structured_output_model(object())
