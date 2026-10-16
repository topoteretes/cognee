"""cognify_code_docs_pipeline hands memify the right tasks; the registry runs them by name."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.memify.memify import _task_factories, resolve_memify_tasks
from cognee.memify_pipelines.cognify_code_docs import (
    cognify_chunks_per_batch,
    cognify_code_doc_chunks,
    cognify_code_docs_pipeline,
    get_code_docs_memify_tasks,
)
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.pipelines.tasks.task import pipeline_needs_llm
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.code_docs.models import CodeDocument
from cognee.tasks.summarization.models import TextSummary

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

EMBED_ONLY_NAMES = ["extract_code_docs", "store_code_docs", "add_code_doc_edges"]
ENTITY_ENRICHMENT_NAMES = ["cognify_code_doc_chunks", "store_code_docs", "add_code_doc_edges"]
EXTRACT_TARGET = "cognee.tasks.code_docs.cognify_code_doc_chunks.extract_graph_and_summarize"
CONFIG_TARGET = "cognee.tasks.code_docs.cognify_code_doc_chunks.get_cognify_config"
SUMMARY_READ = [{"type": ["TextSummary", "DocumentChunk"]}]


def _names(tasks):
    return [task.executable.__name__ for task in tasks]


def _document():
    return CodeDocument(
        name="app/db.py",
        raw_data_location="app/db.py",
        external_metadata=None,
        repo="demo_repo",
        file_path="app/db.py",
    )


def _chunk(document, index):
    return DocumentChunk(
        text=f"app/db.py:{index + 1}\napp/db.f{index} (function)\n\nDocstring {index}.",
        chunk_size=6,
        chunk_index=index,
        cut_type="code_docstring",
        is_part_of=document,
    )


@pytest.fixture
def graph(monkeypatch):
    """The graph engine cognify_code_doc_chunks reads stored summaries from (none by default)."""
    engine = AsyncMock()
    engine.get_filtered_graph_data.return_value = ([], [])
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    return engine


@pytest.mark.asyncio
async def test_pipeline_runs_memify_with_the_embed_only_chain():
    user = SimpleNamespace(id="u1")

    with patch(
        "cognee.memify_pipelines.cognify_code_docs.run_memify_tasks",
        new=AsyncMock(return_value={"status": "ok"}),
    ) as memify_mock:
        result = await cognify_code_docs_pipeline(
            user=user, dataset="code", repo="demo_repo", include_modules=False
        )

    assert result == {"status": "ok"}
    kwargs = memify_mock.call_args.kwargs
    assert kwargs["dataset"] == "code"
    assert kwargs["user"] is user
    # The placeholder seed: the extraction reads the code graph itself, so the
    # dataset's graph is not projected into a memory fragment first.
    assert kwargs["data"] == [{}]
    assert kwargs["run_in_background"] is False

    extraction, enrichment = kwargs["extraction_tasks"], kwargs["enrichment_tasks"]
    assert _names(extraction) + _names(enrichment) == EMBED_ONLY_NAMES
    assert extraction[0].default_params["kwargs"] == {"repo": "demo_repo", "include_modules": False}
    assert enrichment[0].task_config["batch_size"] == cognify_chunks_per_batch()
    # The sweep is scoped the way the extraction was filtered.
    assert enrichment[1].default_params["kwargs"] == {"repo": "demo_repo"}
    # Embed-only makes no LLM call: the setup layer must not run the LLM probe.
    assert all(task.needs_llm is False for task in extraction + enrichment)
    assert pipeline_needs_llm(extraction + enrichment) is False


@pytest.mark.asyncio
async def test_pipeline_forwards_background_and_entity_mode():
    with patch(
        "cognee.memify_pipelines.cognify_code_docs.run_memify_tasks", new=AsyncMock()
    ) as memify_mock:
        await cognify_code_docs_pipeline(
            user=None, run_in_background=True, extract_entities=True, repo=None
        )

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["dataset"] == "main_dataset"
    assert kwargs["run_in_background"] is True
    assert _names(kwargs["enrichment_tasks"]) == ENTITY_ENRICHMENT_NAMES
    assert kwargs["enrichment_tasks"][2].default_params["kwargs"] == {"repo": None}


def test_entity_mode_puts_cognify_extraction_over_the_chunks():
    extraction, enrichment = get_code_docs_memify_tasks(extract_entities=True)

    assert _names(enrichment) == ENTITY_ENRICHMENT_NAMES
    assert enrichment[0].default_params["kwargs"] == {
        "graph_model": KnowledgeGraph,
        "chunks_per_batch": cognify_chunks_per_batch(),
    }
    # Only the extraction talks to the LLM; everything around it stays local.
    assert [task.needs_llm for task in extraction + enrichment] == [False, True, False, False]
    assert pipeline_needs_llm(extraction + enrichment) is True


@pytest.mark.parametrize("extract_entities", [False, True])
def test_every_task_is_a_coroutine_so_the_sweep_sees_the_whole_run(extract_entities):
    # The pipeline hands a coroutine's result to the next task in one call; a
    # generator's output would reach store_code_docs and add_code_doc_edges in
    # batch_size pieces, and a sweep per piece would delete the run's other
    # pieces as stale.
    extraction, enrichment = get_code_docs_memify_tasks(extract_entities=extract_entities)

    assert {task.task_type for task in extraction + enrichment} == {"Coroutine"}


def test_batch_size_follows_the_cognify_config():
    with patch(CONFIG_TARGET, return_value=SimpleNamespace(chunks_per_batch=None)):
        assert cognify_chunks_per_batch() == 2000
    with patch(CONFIG_TARGET, return_value=SimpleNamespace(chunks_per_batch=7)):
        assert cognify_chunks_per_batch() == 7
        _extraction, enrichment = get_code_docs_memify_tasks(extract_entities=True)
        assert enrichment[0].default_params["kwargs"]["chunks_per_batch"] == 7
        assert [task.task_config["batch_size"] for task in enrichment] == [1, 7, 1]


@pytest.mark.asyncio
async def test_cognify_code_doc_chunks_runs_extraction_one_cognify_batch_at_a_time(graph):
    document = _document()
    chunks = [_chunk(document, index) for index in range(5)]
    ctx = SimpleNamespace(pipeline_name="memify_pipeline")
    pieces, active = [], []

    async def fake_extract(data_chunks, graph_model, ctx):
        # Never two pieces in flight: the LLM sees one cognify batch at a time.
        active.append(data_chunks)
        assert len(active) == 1
        await asyncio.sleep(0)
        active.pop()
        pieces.append([chunk.chunk_index for chunk in data_chunks])
        return [TextSummary(text=f"summary {c.chunk_index}", made_from=c) for c in data_chunks]

    with patch(EXTRACT_TARGET, new=AsyncMock(side_effect=fake_extract)) as extract:
        result = await cognify_code_doc_chunks([document, *chunks], ctx=ctx, chunks_per_batch=2)

    # The documents are dropped (extraction reads chunk text); the chunks are
    # cut into cognify batches and the results come back in chunk order.
    assert pieces == [[0, 1], [2, 3], [4]]
    assert all(
        call.kwargs == {"graph_model": KnowledgeGraph, "ctx": ctx}
        for call in extract.await_args_list
    )
    assert [summary.made_from for summary in result] == chunks
    # Stored summaries were looked up once, through the summary/chunk read.
    graph.get_filtered_graph_data.assert_awaited_once_with(SUMMARY_READ)

    # Without an explicit size the pieces are cognify's chunks_per_batch.
    pieces.clear()
    with (
        patch(CONFIG_TARGET, return_value=SimpleNamespace(chunks_per_batch=3)),
        patch(EXTRACT_TARGET, new=AsyncMock(side_effect=fake_extract)),
    ):
        await cognify_code_doc_chunks(chunks)
    assert pieces == [[0, 1, 2], [3, 4]]


@pytest.mark.asyncio
async def test_cognify_code_doc_chunks_sends_only_unsummarized_chunks_to_the_llm(graph):
    document = _document()
    chunks = [_chunk(document, index) for index in range(5)]
    # Chunks 1 and 3 were extracted in an earlier run: a TextSummary was made
    # from each. Another document's summary and a stray edge are ignored.
    graph.get_filtered_graph_data.return_value = (
        [],
        [
            (str(uuid4()), str(chunks[1].id), "made_from", {}),
            (str(uuid4()), str(chunks[3].id), "made_from", {}),
            (str(uuid4()), str(uuid4()), "made_from", {}),
            (str(chunks[2].id), str(document.id), "is_part_of", {}),
        ],
    )

    async def fake_extract(data_chunks, graph_model, ctx):
        return [TextSummary(text=f"summary {c.chunk_index}", made_from=c) for c in data_chunks]

    with patch(EXTRACT_TARGET, new=AsyncMock(side_effect=fake_extract)) as extract:
        result = await cognify_code_doc_chunks([document, *chunks], chunks_per_batch=10)

    # Only the new chunks reached the LLM, in one piece.
    assert [[c.chunk_index for c in call.args[0]] for call in extract.await_args_list] == [
        [0, 2, 4]
    ]
    # Their summaries come first, then the unchanged chunks as themselves, so
    # the whole run still reaches storage and the sweep.
    assert [getattr(point, "made_from", point) for point in result] == [
        chunks[0],
        chunks[2],
        chunks[4],
        chunks[1],
        chunks[3],
    ]
    assert result[3] is chunks[1] and result[4] is chunks[3]


@pytest.mark.asyncio
async def test_cognify_code_doc_chunks_passes_a_chunkless_batch_through(graph):
    # Nothing to extract from: the batch goes on as it is (its documents reach
    # storage, an empty run still reaches the sweep), without a graph read.
    documents = [_document()]
    with patch(EXTRACT_TARGET, new=AsyncMock()) as extract:
        assert await cognify_code_doc_chunks(documents) is documents
        assert await cognify_code_doc_chunks([]) == []
    extract.assert_not_awaited()
    graph.get_filtered_graph_data.assert_not_awaited()


def test_registry_runs_the_pipeline_by_name():
    factories = _task_factories()
    assert set(EMBED_ONLY_NAMES + ENTITY_ENRICHMENT_NAMES) <= set(factories)
    # add_data_points stays registered for generic chains; the LLM stage is
    # reachable by name only in its bounded form.
    assert "add_data_points" in factories
    assert "extract_graph_and_summarize" not in factories

    embed_only = resolve_memify_tasks(EMBED_ONLY_NAMES)
    assert _names(embed_only) == EMBED_ONLY_NAMES
    assert embed_only[0].default_params["kwargs"] == {}
    assert embed_only[1].task_config["batch_size"] == cognify_chunks_per_batch()
    # By name there is no repo filter: the sweep covers every repository.
    assert embed_only[2].default_params["kwargs"] == {}
    assert pipeline_needs_llm(embed_only) is False

    entity = resolve_memify_tasks(ENTITY_ENRICHMENT_NAMES)
    assert _names(entity) == ENTITY_ENRICHMENT_NAMES
    assert entity[0].default_params["kwargs"] == {
        "graph_model": KnowledgeGraph,
        "chunks_per_batch": cognify_chunks_per_batch(),
    }
    assert [task.needs_llm for task in entity] == [True, False, False]

    [generic] = resolve_memify_tasks(["add_data_points"])
    assert generic.executable.__name__ == "add_data_points"
    assert generic.task_config["batch_size"] == cognify_chunks_per_batch()
