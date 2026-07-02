from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.modules.session_lifecycle import estimator


class _FakeTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


class _TinyGraph(BaseModel):
    entity: str


class _TinySummary(BaseModel):
    summary: str


class _AsyncUpload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload

    async def read(self):
        return self._payload


class _FakeChunker:
    def __init__(self, document, max_chunk_size, get_text):
        self.get_text = get_text

    async def read(self):
        async for text in self.get_text():
            yield SimpleNamespace(text=text)


def test_estimate_chunks_reports_stage_tokens_and_cost(monkeypatch):
    monkeypatch.setattr(estimator, "_llm_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(estimator, "_model_name", lambda: "gpt-4o-mini")
    monkeypatch.setattr(estimator, "_graph_prompt", lambda custom_prompt: "extract graph")
    monkeypatch.setattr(estimator, "read_query_prompt", lambda name: "summarize chunk")
    monkeypatch.setattr(
        estimator,
        "get_cognify_config",
        lambda: SimpleNamespace(summarization_model=_TinySummary),
    )
    monkeypatch.setattr(
        estimator,
        "estimate_cost_usd",
        lambda model, tokens_in, tokens_out: (tokens_in + tokens_out) / 1_000_000,
    )

    estimate = estimator.estimate_chunks(
        [SimpleNamespace(text="alpha beta gamma"), SimpleNamespace(text="delta epsilon")],
        operation="remember",
        graph_model=_TinyGraph,
    )

    payload = estimate.to_dict()
    assert payload["dry_run"] is True
    assert payload["operation"] == "remember"
    assert payload["chunks"] == 2
    assert payload["chunk_tokens"] == 5
    assert [stage["name"] for stage in payload["stages"]] == [
        "structured_graph_extraction",
        "chunk_summarization",
    ]
    assert all(stage["calls"] == 2 for stage in payload["stages"])
    assert payload["input_tokens"] > payload["chunk_tokens"]
    assert payload["output_tokens"] > 0
    assert payload["estimated_cost_usd"] > 0
    assert "Dry-run token estimate" in estimator.format_dry_run_estimate(payload)


def test_estimate_chunks_skips_dlt_chunks(monkeypatch):
    monkeypatch.setattr(estimator, "_llm_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(estimator, "_model_name", lambda: "gpt-4o-mini")
    monkeypatch.setattr(estimator, "_graph_prompt", lambda custom_prompt: "extract graph")
    monkeypatch.setattr(estimator, "read_query_prompt", lambda name: "summarize chunk")
    monkeypatch.setattr(
        estimator,
        "get_cognify_config",
        lambda: SimpleNamespace(summarization_model=_TinySummary),
    )

    dlt_document = estimator.DltRowDocument(
        name="table-row",
        raw_data_location="",
        external_metadata="{}",
    )
    estimate = estimator.estimate_chunks(
        [
            SimpleNamespace(text="normal text"),
            SimpleNamespace(text="schema row", is_part_of=dlt_document),
        ],
        operation="cognify",
        graph_model=_TinyGraph,
    )

    payload = estimate.to_dict()
    assert payload["chunks"] == 1
    assert payload["skipped_items"] == 1
    assert all(stage["calls"] == 1 for stage in payload["stages"])
    assert any("DLT row" in warning for warning in payload["warnings"])


@pytest.mark.asyncio
async def test_long_raw_text_is_not_treated_as_a_filesystem_path():
    raw_text = "hello " * 5000

    assert await estimator._input_to_texts(raw_text) == [raw_text]


@pytest.mark.asyncio
async def test_async_upload_is_read_without_llm_calls():
    upload = _AsyncUpload("notes.txt", b"uploaded text")

    assert await estimator._input_to_texts([upload]) == ["uploaded text"]


@pytest.mark.asyncio
async def test_unsupported_upload_extension_is_rejected():
    upload = _AsyncUpload("image.png", b"not text")

    with pytest.raises(ValueError, match="text-like"):
        await estimator._input_to_texts(upload)


def test_unsupported_file_path_extension_is_rejected(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not text")

    with pytest.raises(ValueError, match="text-like"):
        estimator._read_supported_text_path(str(image_path))


@pytest.mark.asyncio
async def test_estimate_remember_dry_run_uses_chunker_without_llm_calls(monkeypatch):
    monkeypatch.setattr(estimator, "_llm_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(estimator, "_model_name", lambda: "gpt-4o-mini")
    monkeypatch.setattr(estimator, "_graph_prompt", lambda custom_prompt: "extract graph")
    monkeypatch.setattr(estimator, "read_query_prompt", lambda name: "summarize chunk")
    monkeypatch.setattr(
        estimator,
        "get_cognify_config",
        lambda: SimpleNamespace(summarization_model=_TinySummary),
    )

    estimate = await estimator.estimate_remember_dry_run(
        "alpha beta",
        chunker=_FakeChunker,
        chunk_size=128,
        graph_model=_TinyGraph,
    )

    assert estimate.operation == "remember"
    assert estimate.chunks == 1
    assert estimate.input_tokens > 0
