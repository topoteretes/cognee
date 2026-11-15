import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.pipeline_stage import pipeline_stage
from cognee.infrastructure.llm.structured_output_framework.litellm_native import (
    get_native_client as native_client_module,
)
from cognee.modules.observability import llm_calls
from cognee.modules.observability.llm_calls import (
    flush_llm_call_capture,
    get_llm_call_capture_config,
)

llm_gateway_module = sys.modules["cognee.infrastructure.llm.LLMGateway"]


class Person(BaseModel):
    name: str


class ProviderDown(Exception):
    pass


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def acreate_structured_output(self, text_input, system_prompt, response_model, **kwargs):
        if self.error is not None:
            raise self.error
        return self.result


def _with_usage(model, prompt_tokens, completion_tokens):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    model._raw_response = SimpleNamespace(usage=usage)
    return model


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    """The real gateway on the litellm_native path, its client replaced by ``FakeClient``."""

    def configure(client, **capture_env):
        capture_env.setdefault("DIRECTORY", str(tmp_path))
        for key in ("ENABLED", "INCLUDE_CONTENT", "DIRECTORY"):
            monkeypatch.delenv(f"LLM_CALL_CAPTURE_{key}", raising=False)
        for key, value in capture_env.items():
            monkeypatch.setenv(f"LLM_CALL_CAPTURE_{key}", value)
        get_llm_call_capture_config.cache_clear()
        monkeypatch.setattr(
            llm_gateway_module, "resolve_structured_output_framework", lambda _: "litellm_native"
        )
        monkeypatch.setattr(native_client_module, "get_native_client", lambda: client)
        return tmp_path

    yield configure
    if llm_calls._writer is not None:
        llm_calls._writer.close()
    llm_calls._writer = None
    get_llm_call_capture_config.cache_clear()


def _records(directory):
    assert flush_llm_call_capture()
    return [
        json.loads(line)
        for path in sorted(directory.glob("llm_calls-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.mark.asyncio
async def test_a_call_through_the_gateway_is_recorded(gateway):
    answer = _with_usage(Person(name="Marie Curie"), prompt_tokens=31, completion_tokens=7)
    directory = gateway(FakeClient(result=answer), ENABLED="true", INCLUDE_CONTENT="true")

    with pipeline_stage("extraction"):
        result = await LLMGateway.acreate_structured_output(
            "Marie Curie discovered polonium.", "Extract the person.", Person
        )

    assert result is answer
    [record] = _records(directory)
    assert record["status"] == "ok"
    assert record["stage"] == "extraction"
    assert record["framework"] == "litellm_native"
    assert record["response_type"] == "Person"
    assert record["model"]
    assert (record["tokens_in"], record["tokens_out"]) == (31, 7)
    assert record["system_prompt"] == "Extract the person."
    assert record["text_input"] == "Marie Curie discovered polonium."
    assert json.loads(record["output"]) == {"name": "Marie Curie"}
    assert record["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_a_plain_text_call_is_recorded_with_its_text(gateway):
    directory = gateway(FakeClient(result="OK"), ENABLED="true", INCLUDE_CONTENT="true")

    await LLMGateway.acreate_structured_output("Say OK.", "Reply with one word.", str)

    [record] = _records(directory)
    assert record["response_type"] == "str"
    assert record["output"] == "OK"
    assert (record["tokens_in"], record["tokens_out"]) == (None, None)


@pytest.mark.asyncio
async def test_a_failed_call_is_recorded_and_its_error_reraised_unchanged(gateway):
    error = ProviderDown("provider is down")
    directory = gateway(FakeClient(error=error), ENABLED="true")

    with pytest.raises(ProviderDown) as raised:
        await LLMGateway.acreate_structured_output("text", "prompt", Person)

    assert raised.value is error
    [record] = _records(directory)
    assert record["status"] == "error"
    assert record["error_class"] == "ProviderDown"
    assert record["output_chars"] is None


@pytest.mark.asyncio
async def test_nothing_is_recorded_while_capture_is_off(gateway, monkeypatch):
    def fail(**kwargs):
        raise AssertionError("capture_llm_call must not run while capture is off")

    monkeypatch.setattr(llm_gateway_module, "capture_llm_call", fail)
    gateway(FakeClient(result=Person(name="Alan Turing")))

    result = await LLMGateway.acreate_structured_output("text", "prompt", Person)

    assert result == Person(name="Alan Turing")
    assert llm_calls._writer is None


@pytest.mark.asyncio
async def test_a_misconfigured_capture_never_breaks_the_call(gateway):
    gateway(FakeClient(result=Person(name="Ada Lovelace")), ENABLED="true", DIRECTORY="relative")

    result = await LLMGateway.acreate_structured_output("text", "prompt", Person)

    assert result == Person(name="Ada Lovelace")
    assert llm_calls._writer is None
