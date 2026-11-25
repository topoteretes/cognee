"""Azure managed identity on the litellm_native path.

With ``LLM_PROVIDER=azure`` and ``LLM_AZURE_USE_MANAGED_IDENTITY=true`` the native client
authenticates with an Entra ID bearer token instead of an API key. The end-to-end tests
send real litellm requests to a local stand-in for an Azure OpenAI deployment and read the
credentials off the wire; only the token source is faked.
"""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from aiohttp import web
from pydantic import BaseModel

from cognee.infrastructure.llm.azure_identity import get_azure_ad_token_provider
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)

native_factory = importlib.import_module(
    "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
)

TOKEN = "entra-test-token"


class Person(BaseModel):
    name: str


def _config(**overrides):
    values = {
        "llm_api_key": "",
        "llm_provider": "azure",
        "llm_azure_use_managed_identity": True,
        "llm_model": "azure/my-deployment",
        "llm_max_completion_tokens": 1024,
        "llm_endpoint": "https://example.openai.azure.com",
        "llm_api_version": "2024-12-01-preview",
        "fallback_model": "",
        "fallback_api_key": "",
        "fallback_endpoint": "",
        "llm_args": {},
        "transcription_model": "",
        "image_transcribe_model": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def fresh_token_provider():
    get_azure_ad_token_provider.cache_clear()
    yield
    get_azure_ad_token_provider.cache_clear()


@pytest.fixture
def fake_azure_identity(monkeypatch):
    """Stand-in for azure.identity that records how often a credential is built."""
    built = []

    class DefaultAzureCredential:
        def __init__(self):
            built.append(self)

    def get_bearer_token_provider(credential, scope):
        assert scope == "https://cognitiveservices.azure.com/.default"
        return lambda: TOKEN

    module = ModuleType("azure.identity")
    module.DefaultAzureCredential = DefaultAzureCredential
    module.get_bearer_token_provider = get_bearer_token_provider
    monkeypatch.setitem(sys.modules, "azure.identity", module)
    return built


def test_managed_identity_builds_a_token_provider_and_sends_no_key(fake_azure_identity):
    with patch.object(native_factory, "get_llm_context_config", return_value=_config()):
        client = native_factory.get_native_client()

    assert client.api_key is None
    assert client.azure_ad_token_provider() == TOKEN


def test_managed_identity_ignores_a_configured_api_key(fake_azure_identity):
    config = _config(llm_api_key="stale-key")
    with patch.object(native_factory, "get_llm_context_config", return_value=config):
        client = native_factory.get_native_client()

    assert client.api_key is None


def test_token_provider_is_built_once_per_process(fake_azure_identity):
    with patch.object(native_factory, "get_llm_context_config", return_value=_config()):
        first = native_factory.get_native_client()
        second = native_factory.get_native_client()

    assert first.azure_ad_token_provider is second.azure_ad_token_provider
    assert len(fake_azure_identity) == 1


def test_managed_identity_requires_an_endpoint(fake_azure_identity):
    config = _config(llm_endpoint="")
    with (
        patch.object(native_factory, "get_llm_context_config", return_value=config),
        pytest.raises(ValueError, match="LLM_ENDPOINT"),
    ):
        native_factory.get_native_client()


def test_instructor_adapter_shares_the_native_token_provider(fake_azure_identity):
    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.azure_openai.adapter import (
        AzureOpenAIAdapter,
    )

    with patch.object(native_factory, "get_llm_context_config", return_value=_config()):
        native_factory.get_native_client()
    AzureOpenAIAdapter(
        api_key="",
        model="azure/my-deployment",
        max_completion_tokens=256,
        endpoint="https://example.openai.azure.com",
        use_managed_identity=True,
    )

    assert len(fake_azure_identity) == 1


def test_api_key_auth_is_unchanged_without_managed_identity():
    config = _config(llm_azure_use_managed_identity=False, llm_api_key="azure-key")
    with patch.object(native_factory, "get_llm_context_config", return_value=config):
        client = native_factory.get_native_client()

    assert client.api_key == "azure-key"
    assert client.azure_ad_token_provider is None


@pytest_asyncio.fixture
async def azure_stub():
    """A local Azure OpenAI chat-completions endpoint that records request headers."""
    seen = []

    async def chat_completions(request):
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        wants_json = body.get("response_format") is not None
        content = json.dumps({"name": "Ada"}) if wants_json else "plain answer"
        return web.json_response(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    app = web.Application()
    app.router.add_post("/openai/deployments/{deployment}/chat/completions", chat_completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", seen
    await runner.cleanup()


def _mi_adapter(endpoint):
    return NativeLiteLLMAdapter(
        api_key=None,
        model="azure/my-deployment",
        max_completion_tokens=256,
        endpoint=endpoint,
        api_version="2024-12-01-preview",
        azure_ad_token_provider=lambda: TOKEN,
    )


def _assert_bearer_only(headers):
    lowered = {key.lower(): value for key, value in headers.items()}
    assert lowered.get("authorization") == f"Bearer {TOKEN}"
    assert "api-key" not in lowered


@pytest.mark.asyncio
async def test_structured_output_authenticates_with_the_token(azure_stub, monkeypatch):
    endpoint, seen = azure_stub
    monkeypatch.delenv("AZURE_API_KEY", raising=False)

    result = await _mi_adapter(endpoint).acreate_structured_output(
        text_input="Who?", system_prompt="Extract the person.", response_model=Person
    )

    assert result.name == "Ada"
    assert seen
    _assert_bearer_only(seen[-1]["headers"])


@pytest.mark.asyncio
async def test_plain_text_output_authenticates_with_the_token(azure_stub, monkeypatch):
    endpoint, seen = azure_stub
    monkeypatch.delenv("AZURE_API_KEY", raising=False)

    result = await _mi_adapter(endpoint).acreate_structured_output(
        text_input="Hi", system_prompt="Answer.", response_model=str
    )

    assert result == "plain answer"
    _assert_bearer_only(seen[-1]["headers"])
