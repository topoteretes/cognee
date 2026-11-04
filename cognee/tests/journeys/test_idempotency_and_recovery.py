"""Journey 5: idempotency and recovery from an interrupted build.

Remembering the same thing twice must not grow the graph. Re-running cognify
must not duplicate anything. When the LLM fails mid-build the pipeline must
report the failure honestly, leave no half-written graph behind, and succeed
on the next attempt.
"""

from __future__ import annotations

from uuid import UUID

import pytest

import cognee
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.search.types import SearchType
from cognee.tests.journeys import _support, mock_ai

DATASET = "journey_idempotency"

DOC_A = (
    "Title: Fenwick Tidal Mill\n\nThe Fenwick Tidal Mill was restored by millwright Beatriz Anand "
    "and grinds spelt on spring tides."
)
DOC_B = (
    "Title: Cormorant Rowing Club\n\nThe Cormorant Rowing Club captain is Declan Obuya and its "
    "boathouse stands below the tidal mill."
)


async def _entity_names(dataset_id: UUID, user) -> list[str]:
    """Lower-cased names of the dataset's Entity nodes (chunk text is not an entity)."""
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    async with set_database_global_context_variables(dataset_id, user.id):
        nodes, _ = await (await get_graph_engine()).get_filtered_graph_data([{"type": ["Entity"]}])
    return [str(props.get("name", "")).lower() for _, props in nodes]


async def _status(dataset_id: UUID) -> str:
    statuses = await cognee.datasets.get_status([dataset_id], ["cognify_pipeline"])
    value = statuses.get(str(dataset_id)) or statuses.get(dataset_id)
    if isinstance(value, dict):
        value = value.get("cognify_pipeline")
    return str(getattr(value, "value", value))


@pytest.mark.journey
@pytest.mark.asyncio
async def test_remembering_the_same_content_twice_is_idempotent(clean_env, default_user):
    first = await cognee.remember(DOC_A, dataset_name=DATASET)
    assert first.status == "completed"
    dataset_id = UUID(str(first.dataset_id))
    base = await _support.snapshot_dataset(dataset_id, default_user)
    assert base.data_rows == 1 and base.graph_nodes > 0, base

    second = await cognee.remember(DOC_A, dataset_name=DATASET)
    assert second.status == "completed"
    repeat = await _support.snapshot_dataset(dataset_id, default_user)

    assert repeat.data_rows == 1, f"duplicate data row after re-remember: {base} -> {repeat}"
    assert repeat.graph_nodes == base.graph_nodes, (
        f"graph nodes grew on re-remember: {base} -> {repeat}"
    )
    assert repeat.graph_edges == base.graph_edges, (
        f"graph edges grew on re-remember: {base} -> {repeat}"
    )
    if base.vector_rows is not None:
        assert repeat.vector_rows == base.vector_rows, (
            f"vector rows changed on re-remember: {base.vector_rows} -> {repeat.vector_rows}"
        )

    # Re-running the build step explicitly is equally a no-op.
    await cognee.cognify([DATASET])
    rebuilt = await _support.snapshot_dataset(dataset_id, default_user)
    assert (rebuilt.graph_nodes, rebuilt.graph_edges) == (base.graph_nodes, base.graph_edges), (
        f"explicit cognify re-run changed the graph: {base} -> {rebuilt}"
    )

    text = _support.content_text(
        await cognee.search(
            query_text="Who restored the Fenwick Tidal Mill?",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
        )
    )
    assert "anand" in text


@pytest.mark.journey
@pytest.mark.asyncio
async def test_interrupted_build_is_reported_and_recovers_on_retry(
    clean_env, default_user, monkeypatch
):
    first = await cognee.remember(DOC_A, dataset_name=DATASET)
    assert first.status == "completed"
    dataset_id = UUID(str(first.dataset_id))
    before = await _support.snapshot_dataset(dataset_id, default_user)
    assert await _status(dataset_id) == PipelineRunStatus.DATASET_PROCESSING_COMPLETED.value

    injector = mock_ai.inject_llm_failure(RuntimeError("injected LLM outage"))
    try:
        injector.armed = True
        with pytest.raises(Exception) as excinfo:
            await cognee.remember(DOC_B, dataset_name=DATASET)
        assert "injected LLM outage" in str(excinfo.value) or "injected LLM outage" in repr(
            excinfo.value.__cause__
        ), f"original error was swallowed: {excinfo.value!r}"
        assert injector.trips >= 1, (
            "failure injector never fired; the test did not exercise an outage"
        )

        status = await _status(dataset_id)
        assert status == PipelineRunStatus.DATASET_PROCESSING_ERRORED.value, (
            f"pipeline status after a failed build should be ERRORED, got {status}"
        )

        during = await _support.snapshot_dataset(dataset_id, default_user)
        # The extraction's own writes are rolled back: no entity of the failed
        # document reaches the graph. The indexing run completed before it
        # (SDK-397), so the document's chunks STAY: the dataset answers
        # chunk-level searches for it, and the memory state says extraction
        # failed while the chunks still answer.
        entity_names = await _entity_names(dataset_id, default_user)
        assert not any("obuya" in n for n in entity_names), (
            f"entities from the failed document reached the graph: {entity_names}"
        )
        memory = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
        assert memory.state == "extraction_failed", memory
        assert memory.search_level == "graph", memory  # the first document's graph still answers
        failed_text = _support.content_text(
            await cognee.search(
                query_text="Who is the Cormorant Rowing Club captain?",
                query_type=SearchType.CHUNKS,
                datasets=[DATASET],
            )
        )
        assert "obuya" in failed_text, "the failed document's chunks should stay searchable"
        if before.vector_rows is not None and during.vector_rows is not None:
            chunk_rows_during = during.vector_rows.get("DocumentChunk_text", 0)
            assert chunk_rows_during > before.vector_rows.get("DocumentChunk_text", 0), (
                "the indexing run should have stored the failed document's chunks"
            )

        # The failed run does not raise for reads: the existing knowledge is still served.
        text = _support.content_text(
            await cognee.search(
                query_text="Who restored the Fenwick Tidal Mill?",
                query_type=SearchType.CHUNKS,
                datasets=[DATASET],
            )
        )
        assert "anand" in text
    finally:
        injector.armed = False
        injector.uninstall()

    # --- the outage clears; the same call now succeeds ----------------------------
    # Record what the retry embeds: an upsert of the same ids would leave the
    # row counts unchanged, so the counts cannot tell a re-embed apart.
    import importlib

    adp_module = importlib.import_module("cognee.tasks.storage.add_data_points")
    original_index = adp_module.index_data_points
    embedded_on_retry: list[str] = []

    async def recording_index(data_points, *args, **kwargs):
        embedded_on_retry.extend(
            str(point.id) for point in data_points if type(point).__name__ == "DocumentChunk"
        )
        return await original_index(data_points, *args, **kwargs)

    monkeypatch.setattr(adp_module, "index_data_points", recording_index)
    retry = await cognee.remember(DOC_B, dataset_name=DATASET)
    monkeypatch.setattr(adp_module, "index_data_points", original_index)
    assert retry.status == "completed", f"retry after outage did not complete: {retry!r}"
    assert await _status(dataset_id) == PipelineRunStatus.DATASET_PROCESSING_COMPLETED.value

    after = await _support.snapshot_dataset(dataset_id, default_user)
    assert after.data_rows == 2, after
    assert after.graph_nodes > before.graph_nodes, (
        f"retry did not add the new document's graph: {after}"
    )
    # The failed attempt's indexing run embedded the new document's chunks; the
    # retry skipped its indexing run and found them in the store, and the first
    # document was extracted before: the retry embedded no chunk at all.
    assert embedded_on_retry == [], f"the retry re-embedded chunks: {embedded_on_retry}"
    if during.vector_rows is not None and after.vector_rows is not None:
        assert after.vector_rows.get("DocumentChunk_text") == during.vector_rows.get(
            "DocumentChunk_text"
        ), f"the retry changed the chunk rows: {during.vector_rows} -> {after.vector_rows}"
    memory = (await cognee.datasets.get_memory_status([dataset_id]))[str(dataset_id)]
    assert memory.state == "ready", memory

    text = _support.content_text(
        await cognee.search(
            query_text="Who is the Cormorant Rowing Club captain?",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
        )
    )
    assert "obuya" in text, f"recovered document is not searchable: {text[:300]}"
