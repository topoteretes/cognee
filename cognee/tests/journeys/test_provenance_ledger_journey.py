"""Journey: the audit ledger across the life of a data item.

With ``PROVENANCE_TRACKING`` on, each data item gets one ledger row and the
nodes and edges it produced get none. The row gets a new version when the
item's content changes, and a tombstone when the document is deleted: forgetting
one document or deleting the dataset. The tombstone is written last, once the
document row is gone, so a delete that failed never left one. Whatever removes
only an item's memory keeps the document and writes none: a memory-only forget,
and the rollback of a failed run. A ledger that cannot be written fails the
operation, and the retry completes it.
"""

from __future__ import annotations

from uuid import UUID

import pytest

import cognee
import cognee.modules.data.methods as data_methods
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.cognify.rollback import cognify_rollback_handler
from cognee.modules.provenance import get_provenance_manager, storage
from cognee.modules.provenance.tombstones import ledger_data_item_key
from cognee.tests.journeys import _support

DATASET = "journey_ledger"

DOC_KITES = (
    "Title: Meridian Kite Club\n\nThe Meridian Kite Club is chaired by Olamide Farquhar and flies "
    "from Gannet Hill every Sunday."
)
DOC_ORCHARD = (
    "Title: Bramblefield Orchard\n\nBramblefield Orchard grows the Winter Pippin apple and is "
    "run by the Okonkwo family."
)
DOC_CHOIR = (
    "Title: Saltmarsh Choir\n\nThe Saltmarsh Choir rehearses in the Corn Exchange under conductor "
    "Yusuf Haldane."
)
DOC_CHOIR_V2 = (
    "Title: Saltmarsh Choir\n\nThe Saltmarsh Choir rehearses in the Corn Exchange under conductor "
    "Ingrid Solvay."
)


@pytest.fixture
def ledger_on(monkeypatch):
    monkeypatch.setenv("PROVENANCE_TRACKING", "true")
    get_cognify_config.cache_clear()
    get_provenance_manager.cache_clear()
    yield get_provenance_manager()
    monkeypatch.delenv("PROVENANCE_TRACKING")
    get_cognify_config.cache_clear()
    get_provenance_manager.cache_clear()


async def _data_id(dataset_id, user, needle: str):
    """The id of the single data item whose stored text contains ``needle`` (lowercase)."""
    rows = await cognee.datasets.list_data(dataset_id, user)
    [match] = [row for row in rows if needle in _support.stored_text(row)]
    return match.id


async def _item(manager, dataset_id, data_id) -> dict:
    row = await manager.get_provenance(ledger_data_item_key(dataset_id, data_id))
    assert row is not None, f"no ledger row for data item {data_id}"
    return row


@pytest.mark.journey
@pytest.mark.asyncio
async def test_ledger_follows_the_data_item_through_update_and_every_delete_path(
    clean_env, default_user, ledger_on
):
    manager = ledger_on
    result = await cognee.remember([DOC_KITES, DOC_ORCHARD, DOC_CHOIR], dataset_name=DATASET)
    assert result.status == "completed"
    dataset_id = _support.as_uuid(result.dataset_id)
    kites = await _data_id(dataset_id, default_user, "meridian kite club")
    orchard = await _data_id(dataset_id, default_user, "bramblefield")
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")

    # --- one row per data item, attributed by id, and none for what it produced ---
    for data_id in (kites, orchard, choir):
        row = await _item(manager, dataset_id, data_id)
        assert row["invalidated"] is False
        assert row["agent_id"] == str(default_user.id)
    snapshot = await _support.snapshot_dataset(dataset_id, default_user)
    assert snapshot.graph_nodes > 3 and snapshot.graph_edges > 0
    kinds = {row["entity_type"] async for row in manager.export(dataset_id=dataset_id)}
    assert kinds == {"data_item"}

    # --- an update is a new version that names the old and the new content -------
    before = (await _item(manager, dataset_id, choir))["metadata"]["snapshot"]["hash"]
    await cognee.update(data_id=choir, data=DOC_CHOIR_V2, dataset_id=dataset_id, user=default_user)
    after = (await _item(manager, dataset_id, choir))["metadata"]["snapshot"]["hash"]
    assert after != before
    history = await manager.revision_history(ledger_data_item_key(dataset_id, choir))
    assert len(history) == 2
    assert history[-1]["delta"] == {"content_hash": [before, after]}

    # --- forget one document --------------------------------------------------------
    await cognee.forget(data_id=kites, dataset_id=dataset_id, user=default_user)
    tombstone = await _item(manager, dataset_id, kites)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "data_deleted"
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is False

    # --- drop only the memory of another, then build it again: the document never
    # left, so there is no tombstone, and the same content is not a new version ---
    await cognee.forget(data_id=orchard, dataset_id=dataset_id, memory_only=True, user=default_user)
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is False
    await cognee.cognify([DATASET], user=default_user)
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is False
    assert len(await manager.revision_history(ledger_data_item_key(dataset_id, orchard))) == 1

    # --- delete the dataset ---------------------------------------------------------
    await cognee.forget(dataset=DATASET, user=default_user)
    for data_id in (orchard, choir):
        tombstone = await _item(manager, dataset_id, data_id)
        assert tombstone["invalidated"] is True
        assert tombstone["invalidation_reason"] == "dataset_deleted"

    # --- nothing was removed from the ledger, and the chain holds -------------------
    assert (await manager.verify_chain())["valid"] is True
    assert (await manager.check(dataset_id=dataset_id))["valid"] is True


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_delete_that_fails_before_the_document_is_gone_writes_no_tombstone(
    clean_env, default_user, ledger_on, monkeypatch
):
    manager = ledger_on
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")

    async def rows_stay(*args, **kwargs):
        raise RuntimeError("relational delete failed")

    with monkeypatch.context() as patched:
        patched.setattr(data_methods, "delete_data", rows_stay)
        with pytest.raises(RuntimeError, match="relational delete failed"):
            await cognee.forget(data_id=choir, dataset_id=dataset_id, user=default_user)

    # The document is still stored, and the ledger does not say otherwise.
    assert [row.id for row in await cognee.datasets.list_data(dataset_id, default_user)] == [choir]
    assert (await _item(manager, dataset_id, choir))["invalidated"] is False

    await cognee.forget(data_id=choir, dataset_id=dataset_id, user=default_user)
    assert await cognee.datasets.list_data(dataset_id, default_user) == []
    tombstone = await _item(manager, dataset_id, choir)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "data_deleted"


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_tombstone_that_cannot_be_written_fails_the_delete_and_deleting_again_writes_it(
    clean_env, default_user, ledger_on, monkeypatch
):
    manager = ledger_on
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")

    async def ledger_down(write_fns):
        raise RuntimeError("ledger down")

    with monkeypatch.context() as patched:
        patched.setattr(storage, "append_chained_many", ledger_down)
        with pytest.raises(RuntimeError, match="ledger down"):
            await cognee.forget(data_id=choir, dataset_id=dataset_id, user=default_user)

    # The document is gone and the caller was told the delete failed: the
    # ledger still holds the item live, never a tombstone it could not write.
    assert await cognee.datasets.list_data(dataset_id, default_user) == []
    assert (await _item(manager, dataset_id, choir))["invalidated"] is False

    # Deleting the same id again finds no document and retracts the ledger row.
    receipt = await cognee.datasets.delete_data(dataset_id, choir, default_user)
    assert receipt["status"] == "not_found"
    tombstone = await _item(manager, dataset_id, choir)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "data_deleted"
    assert (await manager.verify_chain())["valid"] is True


@pytest.mark.journey
@pytest.mark.asyncio
async def test_deleting_the_last_document_with_its_dataset_tombstones_every_row_of_the_dataset(
    clean_env, default_user, ledger_on
):
    manager = ledger_on
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")

    receipt = await cognee.datasets.delete_data(
        dataset_id, choir, default_user, delete_dataset_if_empty=True
    )
    assert receipt["status"] == "success" and receipt["dataset_deleted"] is True

    assert (await _item(manager, dataset_id, choir))["invalidation_reason"] == "data_deleted"
    current = [row async for row in manager.export(dataset_id=dataset_id, include_archived=False)]
    assert current and all(row["invalidated"] for row in current)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_an_update_the_ledger_cannot_record_fails_and_a_retry_records_it(
    clean_env, default_user, ledger_on, monkeypatch
):
    manager = ledger_on
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")
    key = ledger_data_item_key(dataset_id, choir)

    async def ledger_down(write_fns):
        raise RuntimeError("ledger down")

    with monkeypatch.context() as patched:
        patched.setattr(storage, "append_chained_many", ledger_down)
        with pytest.raises(RuntimeError, match="ledger down"):
            await cognee.update(
                data_id=choir, data=DOC_CHOIR_V2, dataset_id=dataset_id, user=default_user
            )
    [original] = await manager.revision_history(key)

    # The retry may rebuild the item from scratch. The document never left, so
    # either way the ledger shows one new version and no tombstone.
    await cognee.update(data_id=choir, data=DOC_CHOIR_V2, dataset_id=dataset_id, user=default_user)
    history = await manager.revision_history(key)
    latest = history[-1]
    assert len(history) == 2
    assert latest.get("invalidated") is not True
    assert latest["content_hash"] != original["content_hash"]
    assert latest["delta"] == {"content_hash": [original["content_hash"], latest["content_hash"]]}
    [row] = await cognee.datasets.list_data(dataset_id, default_user)
    assert "ingrid solvay" in _support.stored_text(row)
    chunks = await _support.vector_texts(dataset_id, default_user)
    assert any("solvay" in text for text in chunks) and not any(
        "haldane" in text for text in chunks
    )


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_ledger_that_cannot_be_written_fails_the_ingestion(
    clean_env, default_user, ledger_on, monkeypatch
):
    manager = ledger_on

    async def ledger_down(write_fns):
        raise RuntimeError("ledger down")

    with monkeypatch.context() as patched:
        patched.setattr(storage, "append_chained_many", ledger_down)
        with pytest.raises(Exception, match="ledger down"):
            await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    assert (await manager.get_statistics())["total_entries"] == 0

    # Nothing was left half-recorded: the same call now stores and records it.
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET)
    assert result.status == "completed"
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")
    assert (await _item(manager, dataset_id, choir))["invalidated"] is False


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_rolled_back_run_keeps_the_row_live_so_a_later_delete_is_recorded(
    clean_env, default_user, ledger_on, monkeypatch
):
    # With early chunk indexing off, the cognify run is the one that first
    # stores the item, so it is the run the ledger row names.
    monkeypatch.setenv("EARLY_CHUNK_INDEXING", "false")
    get_cognify_config.cache_clear()
    manager = ledger_on
    result = await cognee.remember([DOC_CHOIR], dataset_name=DATASET, self_improvement=False)
    dataset_id = _support.as_uuid(result.dataset_id)
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")
    run_id = (await _item(manager, dataset_id, choir))["bundle_id"]
    before = await _support.snapshot_dataset(dataset_id, default_user)

    # Roll that run back the way recovery rolls back a run that never finished the item.
    dataset = await data_methods.get_authorized_dataset(default_user, dataset_id, "write")
    await data_methods.reset_data_pipeline_status(choir, dataset_id)
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        await cognify_rollback_handler(
            pipeline_run_id=UUID(run_id), dataset=dataset, user=default_user
        )

    # The run's memory is gone and the document is not, so the row stays live.
    after = await _support.snapshot_dataset(dataset_id, default_user)
    assert after.graph_nodes < before.graph_nodes
    assert [row.id for row in await cognee.datasets.list_data(dataset_id, default_user)] == [choir]
    assert (await _item(manager, dataset_id, choir))["invalidated"] is False

    # Deleting the document is then what the ledger records, with who did it.
    await cognee.forget(data_id=choir, dataset_id=dataset_id, user=default_user)
    tombstone = await _item(manager, dataset_id, choir)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "data_deleted"
    assert tombstone["invalidated_by"] == str(default_user.id)
