import asyncio
import importlib
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.context_global_variables import llm_config as llm_config_context
from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.exceptions import LLMCallTimeoutError
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    _get_llm_client_cached,
)

gateway_module = importlib.import_module("cognee.infrastructure.llm.LLMGateway")


class ResponseModel(BaseModel):
    value: str


class StallingLLMClient:
    async def acreate_structured_output(self, **kwargs):
        await asyncio.Event().wait()

    async def create_transcript(self, **kwargs):
        await asyncio.Event().wait()

    async def transcribe_image(self, **kwargs):
        await asyncio.Event().wait()


@pytest.fixture
def stalling_gateway(monkeypatch):
    config = SimpleNamespace(
        structured_output_framework="instructor",
        llm_call_timeout_seconds=0.02,
        llm_model="test-model",
    )
    monkeypatch.setattr(gateway_module, "get_llm_context_config", lambda: config)

    from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm import (
        get_llm_client as client_module,
    )

    monkeypatch.setattr(client_module, "get_llm_client", lambda: StallingLLMClient())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_name"),
    [
        (
            lambda: LLMGateway.acreate_structured_output("input", "system", ResponseModel),
            "structured output generation",
        ),
        (lambda: LLMGateway.create_transcript("audio.wav"), "audio transcription"),
        (lambda: LLMGateway.transcribe_image("image.png"), "image transcription"),
    ],
)
async def test_gateway_bounds_every_llm_operation(stalling_gateway, operation, expected_name):
    with pytest.raises(LLMCallTimeoutError) as raised:
        await operation()

    assert raised.value.operation == expected_name


@pytest.mark.asyncio
async def test_gateway_bounds_a_stalling_litellm_http_endpoint():
    request_received = asyncio.Event()
    release_connection = asyncio.Event()

    async def stall_response(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            request_received.set()
            await release_connection.wait()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(stall_response, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    config = LLMConfig(
        llm_provider="custom",
        llm_model="openai/test-model",
        llm_endpoint=f"http://127.0.0.1:{port}/v1",
        llm_api_key="test-key",
        llm_call_timeout_seconds=2.0,
        llm_args={"num_retries": 0},
    )
    token = llm_config_context.set(config)  # ty:ignore[invalid-argument-type]
    _get_llm_client_cached.cache_clear()

    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with pytest.raises(LLMCallTimeoutError):
            await LLMGateway.acreate_structured_output("input", "system", ResponseModel)

        assert request_received.is_set()
        assert loop.time() - started < 2.4
    finally:
        release_connection.set()
        server.close()
        await server.wait_closed()
        await asyncio.sleep(0)
        llm_config_context.reset(token)
        _get_llm_client_cached.cache_clear()
