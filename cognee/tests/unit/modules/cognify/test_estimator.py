"""Dry-run estimator: stage math, pipeline-faithful skips, and input routing.

The estimator must never make LLM calls, so everything here runs offline; the
tokenizer/prompt/config seams are monkeypatched where the arithmetic is under
test, and input routing (paths vs raw text vs rejected inputs) is exercised
against the real functions.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cognee.modules.cognify import estimator


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
    def __init__(self, document, get_text, max_chunk_size):
        self.get_text = get_text

    async def read(self):
        async for text in self.get_text():
            yield SimpleNamespace(text=text)


@pytest.fixture
def offline_estimator(monkeypatch):
    monkeypatch.setattr(estimator, "_llm_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        estimator,
        "get_llm_config",
        lambda: SimpleNamespace(llm_model="gpt-4o-mini", graph_prompt_path="unused.txt"),
    )
    monkeypatch.setattr(estimator, "_graph_prompt", lambda custom_prompt: "extract graph")
    monkeypatch.setattr(estimator, "read_query_prompt", lambda name: "summarize chunk")
    monkeypatch.setattr(
        estimator,
        "get_cognify_config",
        lambda: SimpleNamespace(summarization_model=_TinySummary),
    )


# --------------------------------------------------------------------------- #
# Stage math
# --------------------------------------------------------------------------- #
def test_estimate_chunks_reports_stage_tokens_and_cost(offline_estimator, monkeypatch):
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
    assert "Dry-run token estimate" in str(estimate)


def test_estimate_chunks_warns_when_model_has_no_pricing(offline_estimator, monkeypatch):
    monkeypatch.setattr(estimator, "estimate_cost_usd", lambda model, tokens_in, tokens_out: 0.0)

    estimate = estimator.estimate_chunks(
        [SimpleNamespace(text="alpha beta")], operation="remember", graph_model=_TinyGraph
    )

    assert estimate.estimated_cost_usd == 0.0
    assert any("no pricing entry" in warning for warning in estimate.warnings)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5-mini", True),
        ("gpt-5", True),
        ("gpt-5-nano", True),
        ("gpt-5.4", True),
        ("o1", True),
        ("o3-mini", True),
        ("openai/o4-mini", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("gpt-4.1", False),
        ("anthropic/claude-opus-4-5", False),
        ("gemini-2.5-pro", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert estimator._is_reasoning_model(model) is expected


def test_reasoning_model_scales_output_and_warns(offline_estimator, monkeypatch):
    # gpt-5/o-series bill hidden reasoning tokens as output; the estimator scales
    # the visible-output heuristics by REASONING_OUTPUT_MULTIPLIER and warns.
    chunks = [SimpleNamespace(text="alpha beta gamma delta")]

    baseline = estimator.estimate_chunks(chunks, operation="cognify", graph_model=_TinyGraph)

    monkeypatch.setattr(
        estimator,
        "get_llm_config",
        lambda: SimpleNamespace(llm_model="openai/gpt-5-mini", graph_prompt_path="unused.txt"),
    )
    reasoning = estimator.estimate_chunks(chunks, operation="cognify", graph_model=_TinyGraph)

    assert reasoning.input_tokens == baseline.input_tokens
    assert reasoning.output_tokens == baseline.output_tokens * estimator.REASONING_OUTPUT_MULTIPLIER
    assert any("reasoning model" in w for w in reasoning.warnings)
    assert not any("reasoning model" in w for w in baseline.warnings)


def test_estimate_chunks_skips_dlt_chunks(offline_estimator):
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


# --------------------------------------------------------------------------- #
# Input routing: raw text vs local files vs loudly rejected inputs
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_long_raw_text_is_not_treated_as_a_filesystem_path():
    raw_text = "hello " * 5000

    assert await estimator._input_to_texts(raw_text) == [raw_text]


@pytest.mark.asyncio
async def test_text_file_path_is_read(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("stored text")

    assert await estimator._input_to_texts(str(file_path)) == ["stored text"]


@pytest.mark.asyncio
async def test_file_uri_resolves_to_the_local_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("stored text")

    assert await estimator._input_to_texts(file_path.as_uri()) == ["stored text"]


@pytest.mark.asyncio
async def test_code_file_counts_as_text_like(tmp_path):
    file_path = tmp_path / "script.py"
    file_path.write_text("print('hi')")

    assert await estimator._input_to_texts(str(file_path)) == ["print('hi')"]


@pytest.mark.asyncio
async def test_async_upload_is_read_without_llm_calls():
    upload = _AsyncUpload("notes.txt", b"uploaded text")

    assert await estimator._input_to_texts([upload]) == ["uploaded text"]


@pytest.mark.asyncio
async def test_data_item_recurses_into_wrapped_payload():
    # DataItem is a documented remember() input; a wrapped URL must be rejected,
    # not estimated as the repr string.
    from cognee.tasks.ingestion.data_item import DataItem

    assert await estimator._input_to_texts(DataItem(data="plain text")) == ["plain text"]
    with pytest.raises(ValueError, match="local text inputs only"):
        await estimator._input_to_texts(DataItem(data="https://example.com/page"))


@pytest.mark.asyncio
async def test_trailing_newline_url_is_still_rejected():
    with pytest.raises(ValueError, match="local text inputs only"):
        await estimator._input_to_texts("https://example.com/page\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["https://example.com/doc.html", "s3://bucket/key.txt"])
async def test_remote_urls_are_rejected_not_silently_mispriced(url):
    with pytest.raises(ValueError, match="local text inputs only"):
        await estimator._input_to_texts(url)


@pytest.mark.asyncio
async def test_directories_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="directory inputs"):
        await estimator._input_to_texts(str(tmp_path))


@pytest.mark.asyncio
async def test_missing_absolute_path_is_raw_text():
    # A "/"-prefixed string that is not an existing file is ingested as text by
    # a real run (see #3887), so dry_run must price it as text, not reject it.
    assert await estimator._input_to_texts("/no/such/file.txt") == ["/no/such/file.txt"]


@pytest.mark.asyncio
async def test_local_paths_are_rejected_when_gate_disabled(tmp_path, monkeypatch):
    # ACCEPT_LOCAL_FILE_PATH=false must gate dry_run exactly like real ingestion.
    from cognee.tasks.ingestion.save_data_item_to_storage import settings

    monkeypatch.setattr(settings, "accept_local_file_path", False)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("stored text")

    # An existing local file is rejected, whether named by file:// URI or by an
    # absolute path — the gate must reject exactly what a real run rejects.
    with pytest.raises(ValueError, match="not accepted"):
        await estimator._input_to_texts(file_path.as_uri())
    with pytest.raises(ValueError, match="not accepted"):
        await estimator._input_to_texts(str(file_path))
    # A non-existent absolute path is raw text even with the gate off: a real run
    # saves it as text because it is not an existing local file (see #3887).
    assert await estimator._input_to_texts("/no/such/file.txt") == ["/no/such/file.txt"]
    # A real run treats a relative path to an existing file as raw text when
    # the gate is off.
    monkeypatch.chdir(tmp_path)
    assert await estimator._input_to_texts("notes.txt") == ["notes.txt"]


@pytest.mark.asyncio
async def test_unsupported_upload_extension_is_rejected():
    upload = _AsyncUpload("image.png", b"not text")

    with pytest.raises(ValueError, match="text-like"):
        await estimator._input_to_texts(upload)


def test_unsupported_file_path_extension_is_rejected(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not text")

    with pytest.raises(ValueError, match="text-like"):
        estimator._read_text_path(image_path, str(image_path))


# --------------------------------------------------------------------------- #
# End-to-end remember estimate through the chunker, no LLM calls
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_cognify_dry_run_does_not_create_missing_datasets(monkeypatch):
    # A dry run resolves datasets read-only: a name with no existing dataset
    # must fail loudly rather than create one (contrast the real run).
    created = []

    async def _fake_get_default_user():
        return SimpleNamespace(id="user-1")

    async def _fake_get_authorized_existing_datasets(datasets, permission, user):
        assert permission == "read"
        return []

    monkeypatch.setattr(estimator, "get_default_user", _fake_get_default_user)
    monkeypatch.setattr(
        estimator, "get_authorized_existing_datasets", _fake_get_authorized_existing_datasets
    )

    with pytest.raises(estimator.DatasetNotFoundError):
        await estimator.estimate_cognify_dry_run(["typo-dataset"], chunk_size=128)

    assert created == []


@pytest.mark.asyncio
async def test_estimate_remember_dry_run_uses_chunker_without_llm_calls(offline_estimator):
    estimate = await estimator.estimate_remember_dry_run(
        "alpha beta",
        chunker=_FakeChunker,
        chunk_size=128,
        graph_model=_TinyGraph,
    )

    assert estimate.operation == "remember"
    assert estimate.chunks == 1
    assert estimate.input_tokens > 0
    assert estimate.total_tokens == estimate.input_tokens + estimate.output_tokens
