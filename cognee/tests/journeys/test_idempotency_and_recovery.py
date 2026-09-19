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

    text = _support.result_text(
        await cognee.search(
            query_text="Who restored the Fenwick Tidal Mill?",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
        )
    )
    assert "anand" in text


@pytest.mark.journey
@pytest.mark.asyncio
async def test_interrupted_build_is_reported_and_recovers_on_retry(clean_env, default_user):
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
        assert (
            during.graph_nodes == before.graph_nodes and during.graph_edges == before.graph_edges
        ), f"failed build left partial graph writes behind: {before} -> {during}"
        names = await _support.graph_node_names(dataset_id, default_user)
        assert not any("obuya" in n for n in names), (
            "entities from the failed document reached the graph"
        )

        # The failed run does not raise for reads: the existing knowledge is still served.
        text = _support.result_text(
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
    retry = await cognee.remember(DOC_B, dataset_name=DATASET)
    assert retry.status == "completed", f"retry after outage did not complete: {retry!r}"
    assert await _status(dataset_id) == PipelineRunStatus.DATASET_PROCESSING_COMPLETED.value

    after = await _support.snapshot_dataset(dataset_id, default_user)
    assert after.data_rows == 2, after
    assert after.graph_nodes > before.graph_nodes, (
        f"retry did not add the new document's graph: {after}"
    )

    text = _support.result_text(
        await cognee.search(
            query_text="Who is the Cormorant Rowing Club captain?",
            query_type=SearchType.CHUNKS,
            datasets=[DATASET],
        )
    )
    assert "obuya" in text, f"recovered document is not searchable: {text[:300]}"
