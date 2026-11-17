import asyncio
import json
import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_pipeline_stage
from cognee.modules.observability import llm_calls
from cognee.modules.observability.llm_calls import (
    capture_llm_call,
    chunk_scope,
    flush_llm_call_capture,
    get_llm_call_capture_config,
)
from cognee.modules.observability.llm_calls import writer as writer_module
from cognee.modules.operations.usage_accumulator import parent_run_scope

SYSTEM_PROMPT = "Extract a knowledge graph from the text."
TEXT_INPUT = "Marie Curie discovered polonium in Paris."
OUTPUT = '{"nodes": [{"id": "Marie Curie"}]}'


@pytest.fixture
def capture_env(monkeypatch, tmp_path):
    """Capture configured through the environment, with a fresh config and writer per test."""

    def configure(**env):
        for key in ("ENABLED", "INCLUDE_CONTENT", "DIRECTORY"):
            monkeypatch.delenv(f"LLM_CALL_CAPTURE_{key}", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(f"LLM_CALL_CAPTURE_{key}", value)
        get_llm_call_capture_config.cache_clear()
        return tmp_path

    yield configure
    if llm_calls._writer is not None:
        llm_calls._writer.close()
    llm_calls._writer = None
    get_llm_call_capture_config.cache_clear()


def _capture(**overrides):
    fields = {
        "started_at": datetime.now(timezone.utc),
        "duration_ms": 12.5,
        "model": "openai/gpt-5-mini",
        "framework": "litellm_native",
        "response_type": "KnowledgeGraph",
        "system_prompt": SYSTEM_PROMPT,
        "text_input": TEXT_INPUT,
        "output": OUTPUT,
        "status": "ok",
        "tokens_in": 40,
        "tokens_out": 12,
    }
    fields.update(overrides)
    capture_llm_call(**fields)


def _records(directory):
    assert flush_llm_call_capture()
    lines = [
        line
        for path in sorted(directory.glob("llm_calls-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    return [json.loads(line) for line in lines]


def test_capture_is_off_by_default_and_writes_nothing(capture_env):
    calls_dir = capture_env() / "calls"
    capture_env(DIRECTORY=str(calls_dir))

    _capture()

    assert llm_calls._writer is None
    assert not calls_dir.exists()


def test_scrubbed_record_keeps_metadata_and_no_text(capture_env):
    directory = capture_env(ENABLED="true", DIRECTORY=str(capture_env()))

    _capture()

    [record] = _records(directory)
    assert record["content_included"] is False
    assert record["system_prompt"] is None
    assert record["text_input"] is None
    assert record["output"] is None
    assert record["system_prompt_chars"] == len(SYSTEM_PROMPT)
    assert record["text_input_chars"] == len(TEXT_INPUT)
    assert record["output_chars"] == len(OUTPUT)
    assert len(record["system_prompt_sha256"]) == 64
    assert record["response_type"] == "KnowledgeGraph"
    assert record["tokens_in"] == 40 and record["tokens_out"] == 12
    raw = json.dumps(record)
    for text in ("Marie Curie", "polonium", "Extract a knowledge graph"):
        assert text not in raw


def test_record_with_content_included_keeps_the_text(capture_env):
    directory = capture_env(ENABLED="true", INCLUDE_CONTENT="true", DIRECTORY=str(capture_env()))

    _capture()

    [record] = _records(directory)
    assert record["content_included"] is True
    assert record["system_prompt"] == SYSTEM_PROMPT
    assert record["text_input"] == TEXT_INPUT
    assert record["output"] == OUTPUT


def test_stage_and_run_come_from_the_current_context(capture_env):
    directory = capture_env(ENABLED="true", DIRECTORY=str(capture_env()))
    run_id = uuid4()

    token = current_pipeline_stage.set("extraction")
    try:
        with parent_run_scope(run_id):
            _capture()
    finally:
        current_pipeline_stage.reset(token)
    _capture()

    first, second = _records(directory)
    assert first["stage"] == "extraction"
    assert first["run_id"] == str(run_id)
    assert second["stage"] is None
    assert second["run_id"] is None


def test_chunk_id_comes_from_the_enclosing_chunk_scope(capture_env):
    directory = capture_env(ENABLED="true", INCLUDE_CONTENT="true", DIRECTORY=str(capture_env()))
    first_chunk, second_chunk = uuid4(), uuid4()

    async def extract(chunk_id):
        with chunk_scope(chunk_id):
            await asyncio.sleep(0)
            _capture(text_input=str(chunk_id))

    async def run():
        await asyncio.gather(extract(first_chunk), extract(second_chunk))

    asyncio.run(run())
    _capture(text_input="outside")

    chunk_by_input = {r["text_input"]: r["chunk_id"] for r in _records(directory)}
    assert chunk_by_input == {
        str(first_chunk): str(first_chunk),
        str(second_chunk): str(second_chunk),
        "outside": None,
    }


def test_failed_call_is_recorded_with_its_error_class(capture_env):
    directory = capture_env(ENABLED="true", DIRECTORY=str(capture_env()))

    _capture(output=None, status="error", error_class="RateLimitError", tokens_in=None)

    [record] = _records(directory)
    assert record["status"] == "error"
    assert record["error_class"] == "RateLimitError"
    assert record["output_chars"] is None


def test_every_record_has_a_version_and_its_own_id(capture_env):
    directory = capture_env(ENABLED="true", DIRECTORY=str(capture_env()))

    _capture()
    _capture()

    first, second = _records(directory)
    assert first["schema_version"] == llm_calls.SCHEMA_VERSION
    assert first["event_id"] != second["event_id"]


@pytest.mark.parametrize("directory", ["relative/llm_calls", "s3://bucket/llm_calls"])
def test_a_directory_that_is_not_an_absolute_local_path_is_rejected(capture_env, directory):
    capture_env(ENABLED="true", DIRECTORY=directory)

    with pytest.raises(ValueError, match="absolute local path"):
        get_llm_call_capture_config()


def test_capture_never_raises_into_the_caller(capture_env):
    capture_env(ENABLED="true", DIRECTORY="relative/llm_calls")

    _capture()

    assert llm_calls._writer is None


def test_a_write_error_stops_the_writer_without_raising(capture_env):
    blocker = capture_env() / "not_a_directory"
    blocker.write_text("")
    capture_env(ENABLED="true", DIRECTORY=str(blocker / "calls"))

    _capture()
    assert flush_llm_call_capture()
    _capture()

    assert llm_calls._writer._failed is True
    assert llm_calls._writer._discarded >= 1


def test_capture_returns_while_the_writer_is_stuck_on_disk(capture_env, monkeypatch):
    """A disk write that never finishes must not hold up the LLM calls being captured."""
    directory = capture_env(ENABLED="true", DIRECTORY=str(capture_env()))
    write_started, release_disk = threading.Event(), threading.Event()
    real_open = open

    class StuckFile:
        """A file whose write() blocks until the test releases it, like a hung disk."""

        def __init__(self, path):
            self._file = real_open(path, "a", encoding="utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._file.close()

        def write(self, text):
            write_started.set()
            release_disk.wait()
            self._file.write(text)

        def flush(self):
            self._file.flush()

    monkeypatch.setattr(writer_module, "open", lambda path, *a, **k: StuckFile(path), raising=False)

    _capture()
    assert write_started.wait(5), "the writer thread never reached the disk"

    caller = threading.Thread(target=lambda: [_capture() for _ in range(100)])
    caller.start()
    caller.join(timeout=5)
    stuck = caller.is_alive()

    release_disk.set()
    caller.join()
    assert not stuck, "capture_llm_call waited on the disk write"
    assert len(_records(directory)) == 101
