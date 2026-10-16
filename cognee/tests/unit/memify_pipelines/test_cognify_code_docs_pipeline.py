"""cognify_code_docs_pipeline hands memify the right tasks; the registry runs them by name."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.api.v1.memify.memify import _task_factories, resolve_memify_tasks
from cognee.memify_pipelines.cognify_code_docs import (
    cognify_chunks_per_batch,
    cognify_code_docs_pipeline,
    get_code_docs_memify_tasks,
    select_code_doc_chunks,
)
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.pipelines.tasks.task import pipeline_needs_llm
from cognee.pipelines.types import Drop
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.code_docs.models import CodeDocument

EMBED_ONLY_NAMES = ["extract_code_docs", "add_data_points", "add_code_doc_edges"]
ENTITY_ENRICHMENT_NAMES = [
    "select_code_doc_chunks",
    "extract_graph_and_summarize",
    "add_data_points",
    "add_code_doc_edges",
]


def _names(tasks):
    return [task.executable.__name__ for task in tasks]


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


def test_entity_mode_puts_cognify_extraction_over_the_chunks():
    extraction, enrichment = get_code_docs_memify_tasks(extract_entities=True)

    assert _names(enrichment) == ENTITY_ENRICHMENT_NAMES
    extract_graph = enrichment[1]
    assert extract_graph.default_params["kwargs"] == {"graph_model": KnowledgeGraph}
    assert extract_graph.task_config["batch_size"] == cognify_chunks_per_batch()
    # Only the extraction talks to the LLM; everything around it stays local.
    assert [task.needs_llm for task in extraction + enrichment] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert pipeline_needs_llm(extraction + enrichment) is True


@pytest.mark.parametrize("extract_entities", [False, True])
def test_every_task_is_a_coroutine_so_the_sweep_sees_the_whole_run(extract_entities):
    # The pipeline hands a coroutine's result to the next task in one call; a
    # generator's output would reach add_data_points and add_code_doc_edges in
    # batch_size pieces, and a sweep per piece would delete the run's other
    # pieces as stale.
    extraction, enrichment = get_code_docs_memify_tasks(extract_entities=extract_entities)

    assert {task.task_type for task in extraction + enrichment} == {"Coroutine"}


def test_batch_size_follows_the_cognify_config():
    config_target = "cognee.memify_pipelines.cognify_code_docs.get_cognify_config"
    with patch(config_target, return_value=SimpleNamespace(chunks_per_batch=None)):
        assert cognify_chunks_per_batch() == 2000
    with patch(config_target, return_value=SimpleNamespace(chunks_per_batch=7)):
        assert cognify_chunks_per_batch() == 7
        _extraction, enrichment = get_code_docs_memify_tasks(extract_entities=True)
        assert [task.task_config["batch_size"] for task in enrichment] == [1, 7, 7, 1]


@pytest.mark.asyncio
async def test_select_code_doc_chunks_keeps_chunks_and_drops_an_empty_batch():
    document = CodeDocument(
        name="app/db.py",
        raw_data_location="app/db.py",
        external_metadata=None,
        repo="demo_repo",
        file_path="app/db.py",
    )
    chunk = DocumentChunk(
        text="app/db.py:1\napp/db (module)\n\nDatabase helpers.",
        chunk_size=6,
        chunk_index=0,
        cut_type="code_docstring",
        is_part_of=document,
    )

    assert await select_code_doc_chunks([document, chunk]) == [chunk]
    # extract_graph_and_summarize rejects an empty list, so a chunk-less batch
    # leaves the pipeline here instead.
    assert await select_code_doc_chunks([document]) is Drop
    assert await select_code_doc_chunks([]) is Drop


def test_registry_runs_the_pipeline_by_name():
    assert set(EMBED_ONLY_NAMES + ENTITY_ENRICHMENT_NAMES) <= set(_task_factories())

    embed_only = resolve_memify_tasks(EMBED_ONLY_NAMES)
    assert _names(embed_only) == EMBED_ONLY_NAMES
    assert embed_only[0].default_params["kwargs"] == {}
    assert embed_only[1].task_config["batch_size"] == cognify_chunks_per_batch()
    assert pipeline_needs_llm(embed_only) is False

    entity = resolve_memify_tasks(ENTITY_ENRICHMENT_NAMES)
    assert _names(entity) == ENTITY_ENRICHMENT_NAMES
    assert entity[1].default_params["kwargs"] == {"graph_model": KnowledgeGraph}
    assert entity[1].task_config["batch_size"] == cognify_chunks_per_batch()
    assert [task.needs_llm for task in entity] == [False, True, False, False]
