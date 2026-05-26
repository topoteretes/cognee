from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from cognee.modules.graph.methods.get_global_context_graph_inputs import (
    DatasetEntityCounts,
    DatasetGraphEntityInput,
    SummaryEntityLoadResult,
    _chunk_entity_statement,
    _summary_chunk_statement,
    get_dataset_chunk_entity_counts,
    load_dataset_graph_entity_input,
    load_summary_entities_for_dataset,
)
from cognee.tasks.memify.global_context_index.bucketing.graph.inputs import (
    load_graph_bucketing_inputs,
)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


def _session_with_results(*results):
    session = AsyncMock()
    session.execute.side_effect = [FakeResult(result) for result in results]
    return session


@pytest.mark.asyncio
async def test_load_summary_entities_preserves_missing_and_no_entity_diagnostics():
    dataset_id = uuid4()
    summary_with_entity = uuid4()
    summary_without_entity = uuid4()
    missing_summary = uuid4()
    chunk_with_entity = uuid4()
    chunk_without_entity = uuid4()
    entity_id = uuid4()
    session = _session_with_results(
        [
            (summary_with_entity, chunk_with_entity),
            (summary_without_entity, chunk_without_entity),
        ],
        [(chunk_with_entity, entity_id)],
    )

    result = await load_summary_entities_for_dataset(
        dataset_id,
        [summary_with_entity, summary_without_entity, missing_summary],
        session=session,
    )

    assert result.entities_by_summary_id[str(summary_with_entity)] == {str(entity_id)}
    assert result.entities_by_summary_id[str(summary_without_entity)] == set()
    assert result.entities_by_summary_id[str(missing_summary)] == set()
    assert result.summarized_chunk_count == 2
    assert result.summary_ids_with_made_from == {
        str(summary_with_entity),
        str(summary_without_entity),
    }
    assert result.missing_made_from_summary_ids == {str(missing_summary)}
    assert result.entity_link_count == 1


@pytest.mark.asyncio
async def test_load_summary_entities_all_no_entity_dataset_is_valid_diagnostic():
    dataset_id = uuid4()
    first_summary = uuid4()
    second_summary = uuid4()
    first_chunk = uuid4()
    second_chunk = uuid4()
    session = _session_with_results(
        [(first_summary, first_chunk), (second_summary, second_chunk)],
        [],
    )

    result = await load_summary_entities_for_dataset(
        dataset_id,
        [first_summary, second_summary],
        session=session,
    )

    assert result.entities_by_summary_id == {
        str(first_summary): set(),
        str(second_summary): set(),
    }
    assert result.summarized_chunk_count == 2
    assert result.missing_made_from_summary_ids == set()
    assert result.entity_link_count == 0


@pytest.mark.asyncio
async def test_dataset_chunk_entity_counts_use_distinct_chunks_per_entity():
    dataset_id = uuid4()
    first_summary = uuid4()
    second_summary = uuid4()
    third_summary = uuid4()
    first_chunk = uuid4()
    second_chunk = uuid4()
    shared_entity = uuid4()
    other_entity = uuid4()
    session = _session_with_results(
        [
            (first_summary, first_chunk),
            (second_summary, first_chunk),
            (third_summary, second_chunk),
        ],
        [
            (first_chunk, shared_entity),
            (first_chunk, shared_entity),
            (second_chunk, shared_entity),
            (second_chunk, other_entity),
        ],
    )

    counts = await get_dataset_chunk_entity_counts(
        dataset_id,
        [first_summary, second_summary, third_summary],
        session=session,
    )

    assert counts.chunk_count == 2
    assert counts.entity_chunk_counts == {
        str(shared_entity): 2,
        str(other_entity): 1,
    }


@pytest.mark.asyncio
async def test_combined_graph_entity_input_loads_rows_once_for_entities_and_counts():
    dataset_id = uuid4()
    first_summary = uuid4()
    second_summary = uuid4()
    first_chunk = uuid4()
    second_chunk = uuid4()
    shared_entity = uuid4()
    session = _session_with_results(
        [(first_summary, first_chunk), (second_summary, second_chunk)],
        [(first_chunk, shared_entity), (second_chunk, shared_entity)],
    )

    result = await load_dataset_graph_entity_input(
        dataset_id,
        [first_summary, second_summary],
        session=session,
    )

    assert result.summary_entities.entities_by_summary_id == {
        str(first_summary): {str(shared_entity)},
        str(second_summary): {str(shared_entity)},
    }
    assert result.entity_counts == DatasetEntityCounts(
        chunk_count=2,
        entity_chunk_counts={str(shared_entity): 2},
    )
    assert session.execute.await_count == 2


@pytest.mark.asyncio
async def test_graph_bucketing_input_provider_computes_idf_weights(monkeypatch):
    dataset_id = uuid4()
    summary_id = uuid4()
    entity_id = uuid4()

    async def load_graph_entity_input(dataset_id, expected_summary_ids, session=None):
        return DatasetGraphEntityInput(
            summary_entities=SummaryEntityLoadResult(
                entities_by_summary_id={str(summary_id): {str(entity_id)}},
                summarized_chunk_count=4,
                summary_ids_with_made_from={str(summary_id)},
                missing_made_from_summary_ids=set(),
                entity_link_count=1,
            ),
            entity_counts=DatasetEntityCounts(
                chunk_count=4,
                entity_chunk_counts={str(entity_id): 1},
            ),
        )

    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.bucketing.graph.inputs."
        "load_dataset_graph_entity_input",
        load_graph_entity_input,
    )

    entities_by_summary_id, idf_weights = await load_graph_bucketing_inputs(
        dataset_id, [summary_id]
    )

    assert entities_by_summary_id == {str(summary_id): {str(entity_id)}}
    assert idf_weights[str(entity_id)] == pytest.approx(1.3862943611)


@pytest.mark.asyncio
async def test_graph_bucketing_input_provider_omits_none_session(monkeypatch):
    dataset_id = uuid4()
    calls = []

    async def load_graph_entity_input(*args, **kwargs):
        calls.append((args, kwargs))
        return DatasetGraphEntityInput(
            summary_entities=SummaryEntityLoadResult(
                entities_by_summary_id={},
                summarized_chunk_count=0,
                summary_ids_with_made_from=set(),
                missing_made_from_summary_ids=set(),
                entity_link_count=0,
            ),
            entity_counts=DatasetEntityCounts(chunk_count=0, entity_chunk_counts={}),
        )

    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.bucketing.graph.inputs."
        "load_dataset_graph_entity_input",
        load_graph_entity_input,
    )

    await load_graph_bucketing_inputs(dataset_id, [])

    assert calls == [((dataset_id, []), {})]


@pytest.mark.asyncio
async def test_graph_bucketing_input_provider_rejects_missing_made_from(monkeypatch):
    dataset_id = uuid4()
    summary_id = uuid4()

    async def load_graph_entity_input(dataset_id, expected_summary_ids, session=None):
        return DatasetGraphEntityInput(
            summary_entities=SummaryEntityLoadResult(
                entities_by_summary_id={str(summary_id): set()},
                summarized_chunk_count=0,
                summary_ids_with_made_from=set(),
                missing_made_from_summary_ids={str(summary_id)},
                entity_link_count=0,
            ),
            entity_counts=DatasetEntityCounts(chunk_count=0, entity_chunk_counts={}),
        )

    monkeypatch.setattr(
        "cognee.tasks.memify.global_context_index.bucketing.graph.inputs."
        "load_dataset_graph_entity_input",
        load_graph_entity_input,
    )

    with pytest.raises(ValueError, match="made_from"):
        await load_graph_bucketing_inputs(dataset_id, [summary_id])


def test_summary_chunk_query_joins_through_slug_and_scopes_dataset():
    dataset_id = uuid4()
    summary_id = uuid4()
    statement = _summary_chunk_statement(dataset_id, {summary_id})

    sql = str(statement)
    params = list(statement.compile().params.values())

    assert "slug" in sql
    assert "source_node_id" in sql
    assert "destination_node_id" in sql
    assert "relationship_name" in sql
    assert "dataset_id" in sql
    assert "made_from" in params
    assert "TextSummary" in params
    assert "DocumentChunk" in params


def test_chunk_entity_query_filters_contains_label_and_entity_type():
    dataset_id = uuid4()
    chunk_id = uuid4()
    statement = _chunk_entity_statement(dataset_id, {chunk_id})

    sql = str(statement)
    params = list(statement.compile().params.values())

    assert "slug" in sql
    assert "source_node_id" in sql
    assert "destination_node_id" in sql
    assert "label" in sql
    assert "dataset_id" in sql
    assert "contains" in params
    assert "DocumentChunk" in params
    assert "Entity" in params
