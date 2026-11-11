"""Journey: the audit ledger across every way a data item's memory is removed.

With ``PROVENANCE_TRACKING`` on, each data item gets one ledger row, and that
row is what a delete retracts: forgetting one document, dropping only its
memory, deleting the dataset. A ledger that cannot be written fails the
operation, and the retry completes it.
"""

from __future__ import annotations

import pytest

import cognee
from cognee.modules.cognify.config import get_cognify_config
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
async def test_ledger_retracts_the_data_item_on_every_delete_path(
    clean_env, default_user, ledger_on
):
    manager = ledger_on
    result = await cognee.remember([DOC_KITES, DOC_ORCHARD, DOC_CHOIR], dataset_name=DATASET)
    assert result.status == "completed"
    dataset_id = _support.as_uuid(result.dataset_id)
    kites = await _data_id(dataset_id, default_user, "meridian kite club")
    orchard = await _data_id(dataset_id, default_user, "bramblefield")
    choir = await _data_id(dataset_id, default_user, "saltmarsh choir")

    # --- every data item is recorded, and a fresh dataset shows no drift ----------
    for data_id in (kites, orchard, choir):
        assert (await _item(manager, dataset_id, data_id))["invalidated"] is False
    drift = await manager.check_drift(dataset_id, default_user.id)
    assert drift["valid"] is True, drift
    assert drift["checked"] > 0

    # --- forget one document --------------------------------------------------------
    await cognee.forget(data_id=kites, dataset_id=dataset_id, user=default_user)
    tombstone = await _item(manager, dataset_id, kites)
    assert tombstone["invalidated"] is True
    assert tombstone["invalidation_reason"] == "data_deleted"
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is False

    # --- drop only the memory of another, then build it again ----------------------
    await cognee.forget(data_id=orchard, dataset_id=dataset_id, memory_only=True, user=default_user)
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is True
    await cognee.cognify([DATASET], user=default_user)
    assert (await _item(manager, dataset_id, orchard))["invalidated"] is False
    history = await manager.revision_history(ledger_data_item_key(dataset_id, orchard))
    assert [bool(version.get("invalidated")) for version in history] == [False, True, False]

    # --- delete the dataset ---------------------------------------------------------
    await cognee.forget(dataset=DATASET, user=default_user)
    for data_id in (orchard, choir):
        tombstone = await _item(manager, dataset_id, data_id)
        assert tombstone["invalidated"] is True
        assert tombstone["invalidation_reason"] == "dataset_deleted"

    # --- only data items are ever retracted, and the chain holds -------------------
    retracted = {
        row["entity_type"]
        async for row in manager.export(dataset_id=dataset_id, include_archived=False)
        if row["invalidated"]
    }
    assert retracted == {"data_item"}
    assert (await manager.verify_chain())["valid"] is True
    assert (await manager.check(dataset_id=dataset_id))["valid"] is True


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_tombstone_that_cannot_be_written_fails_the_delete_and_a_retry_writes_it(
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

    # The delete did not finish: the data item is still there to be retried.
    assert (await _item(manager, dataset_id, choir))["invalidated"] is False
    assert [row.id for row in await cognee.datasets.list_data(dataset_id, default_user)] == [choir]

    await cognee.forget(data_id=choir, dataset_id=dataset_id, user=default_user)
    assert (await _item(manager, dataset_id, choir))["invalidated"] is True
    assert await cognee.datasets.list_data(dataset_id, default_user) == []


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
