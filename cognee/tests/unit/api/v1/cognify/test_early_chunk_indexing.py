"""The early chunk-indexing step of the default cognify pipeline (SDK-397)."""

import importlib

import pytest

from cognee.modules.cognify.config import get_cognify_config

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")


def _config(early_chunk_indexing: bool):
    return get_cognify_config().model_copy(update={"early_chunk_indexing": early_chunk_indexing})


@pytest.mark.asyncio
async def test_default_tasks_index_chunks_before_extraction(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(True))

    tasks = await cognify_module.get_default_tasks(chunk_size=512, chunks_per_batch=7)

    names = [task.executable.__name__ for task in tasks]
    assert names == [
        "classify_documents",
        "extract_chunks_from_documents",
        "index_chunks_for_search",
        "extract_graph_and_summarize",
        "add_data_points",
    ]
    early = tasks[2]
    assert early.needs_llm is False, "the early write never calls the LLM"
    assert early.task_config["batch_size"] == 7, "batched like the extraction it precedes"


@pytest.mark.asyncio
async def test_the_setting_turns_the_step_off(monkeypatch):
    monkeypatch.setattr(cognify_module, "get_cognify_config", lambda: _config(False))

    tasks = await cognify_module.get_default_tasks(chunk_size=512)

    names = [task.executable.__name__ for task in tasks]
    assert "index_chunks_for_search" not in names
    assert names[:2] == ["classify_documents", "extract_chunks_from_documents"]
    assert names[2] == "extract_graph_and_summarize"


def test_the_setting_is_on_by_default_and_reported():
    config = get_cognify_config()

    assert config.early_chunk_indexing is True
    assert config.to_dict()["early_chunk_indexing"] is True
