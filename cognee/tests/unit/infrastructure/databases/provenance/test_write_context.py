from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    data_item_id,
    graph_provenance_write_kwargs,
    make_source_ref_key,
    source_ref_from_context,
)


def test_data_item_id_resolves_data_dot_id_and_dataitem_dot_data_id():
    # Relational Data exposes `.id`.
    data_id = uuid4()
    assert data_item_id(SimpleNamespace(id=data_id)) == data_id

    # Ingestion DataItem exposes `.data_id` (no `.id`).
    item_data_id = uuid4()
    data_item = SimpleNamespace(data=object(), data_id=item_data_id)
    assert data_item_id(data_item) == item_data_id

    # Items with neither (raw file/text, or the memify CogneeGraph) -> None.
    assert data_item_id(SimpleNamespace(data=object(), data_id=None)) is None
    assert data_item_id(object()) is None
    assert data_item_id(None) is None


def test_source_ref_from_context_uses_dataitem_data_id():
    dataset_id = uuid4()
    item_data_id = uuid4()
    ctx = SimpleNamespace(
        dataset=SimpleNamespace(id=dataset_id),
        data_item=SimpleNamespace(data=object(), data_id=item_data_id),
        pipeline_run_id=None,
    )
    assert source_ref_from_context(ctx) == (
        make_source_ref_key(dataset_id, item_data_id),
        None,
    )


def test_source_ref_from_context_uses_data_item_id():
    dataset_id = uuid4()
    data_id = uuid4()
    run_id = uuid4()
    ctx = SimpleNamespace(
        dataset=SimpleNamespace(id=dataset_id),
        data_item=SimpleNamespace(id=data_id),
        pipeline_run_id=run_id,
    )

    assert source_ref_from_context(ctx) == (make_source_ref_key(dataset_id, data_id), str(run_id))


def test_source_ref_from_context_uses_fallback_data_id_only_when_missing():
    dataset_id = uuid4()
    fallback_data_id = uuid4()
    explicit_data_id = uuid4()

    missing_data_ctx = SimpleNamespace(dataset=SimpleNamespace(id=dataset_id), data_item=None)
    assert source_ref_from_context(missing_data_ctx, fallback_data_id=fallback_data_id) == (
        make_source_ref_key(dataset_id, fallback_data_id),
        None,
    )

    data_ctx = SimpleNamespace(
        dataset=SimpleNamespace(id=dataset_id),
        data_item=SimpleNamespace(id=explicit_data_id),
    )
    assert source_ref_from_context(data_ctx, fallback_data_id=fallback_data_id) == (
        make_source_ref_key(dataset_id, explicit_data_id),
        None,
    )


@pytest.mark.asyncio
async def test_graph_provenance_write_kwargs_unstamped_on_ledger_graph(monkeypatch):
    import cognee.infrastructure.databases.provenance.write_context as write_context

    async def _false(_graph):
        return False

    # A ledger graph (not graph-provenance) -> no stamping, and no mutation:
    # the helper must not mark the graph.
    monkeypatch.setattr(write_context, "stores_provenance_in_graph", _false)

    kwargs = await graph_provenance_write_kwargs(object(), dataset_id=uuid4(), data_id=uuid4())

    assert kwargs == {"source_ref_key": None, "pipeline_run_id": None}


@pytest.mark.asyncio
async def test_graph_provenance_write_kwargs_stamps_when_graph_stores_provenance(monkeypatch):
    import cognee.infrastructure.databases.provenance.write_context as write_context

    dataset_id = uuid4()
    data_id = uuid4()

    async def _true(_graph):
        return True

    monkeypatch.setattr(write_context, "stores_provenance_in_graph", _true)

    kwargs = await graph_provenance_write_kwargs(object(), dataset_id=dataset_id, data_id=data_id)

    assert kwargs == {
        "source_ref_key": make_source_ref_key(dataset_id, data_id),
        "pipeline_run_id": None,
    }
