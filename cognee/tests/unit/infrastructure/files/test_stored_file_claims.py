"""A stored file an in-flight operation is about to commit a row for is never removed.

Stored files are content-addressed and shared across a tenant's datasets, and a
removal in one dataset does not wait on another dataset's lock. These tests
stand in for that: one task stores a file and has not committed its row yet,
another removes the same location because no row points at it.
"""

import asyncio
import importlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee.modules.data.methods.remove_unreferenced_files as removal
from cognee.infrastructure.files.stored_file_claims import (
    holding_stored_files,
    removing_stored_files,
    store_claimed,
)
from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path

# The package re-exports a function under the module's name, so import the module itself.
saving = importlib.import_module("cognee.modules.ingestion.save_data_to_file")

OUTCOME = "The update was applied: the document holds its new content."


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Store under ``tmp_path``; the engine sees no row for any file and really deletes it."""
    monkeypatch.setattr(
        saving, "get_storage_config", lambda: {"data_root_directory": str(tmp_path)}
    )

    async def remove_data_file(location):
        os.remove(get_data_file_path(location))

    engine = SimpleNamespace(
        is_managed_data_file=lambda location: True,
        referenced_data_files=AsyncMock(return_value=set()),
        remove_data_file=remove_data_file,
    )
    monkeypatch.setattr(removal, "get_relational_engine", lambda: engine)
    return tmp_path


@pytest.mark.asyncio
async def test_a_file_another_operation_stored_stays_until_that_operation_is_over(data_root):
    stored = asyncio.Event()
    finish = asyncio.Event()
    location = None

    async def add_in_another_dataset():
        nonlocal location
        async with holding_stored_files():
            location = await saving.save_data_to_file("The same text in two datasets.")
            stored.set()
            await finish.wait()  # its row commits here

    adding = asyncio.create_task(add_in_another_dataset())
    await stored.wait()

    # An update in this dataset releases the same file: no row points at it yet.
    await removal.remove_unreferenced_files([location], OUTCOME)
    assert os.path.exists(get_data_file_path(location)), (
        "A file an add was about to commit was removed"
    )

    finish.set()
    await adding
    # Once that operation is over the claim is gone, and an unused file is removed.
    await removal.remove_unreferenced_files([location], OUTCOME)
    assert not os.path.exists(get_data_file_path(location))


@pytest.mark.asyncio
async def test_an_operation_removes_the_files_it_stored_and_did_not_use(data_root):
    async with holding_stored_files():
        location = await saving.save_data_to_file("Staged by an update that did not publish.")
        await removal.remove_unreferenced_files([location], OUTCOME)

    assert not os.path.exists(get_data_file_path(location))


@pytest.mark.asyncio
async def test_a_nested_scope_keeps_its_claims_until_the_outer_one_ends(data_root):
    claimed = asyncio.Event()
    finish = asyncio.Event()
    location = None

    async def item_run():
        nonlocal location
        async with holding_stored_files():
            async with holding_stored_files():  # ingest_data inside the item's run
                location = await saving.save_data_to_file("Stored by a nested ingest.")
            claimed.set()
            await finish.wait()

    running = asyncio.create_task(item_run())
    await claimed.wait()
    await removal.remove_unreferenced_files([location], OUTCOME)
    assert os.path.exists(get_data_file_path(location))

    finish.set()
    await running


def _counting_store(path):
    """A store that writes ``path`` and returns its location; counts its writes."""
    writes = []

    async def store():
        writes.append(path)
        path.write_text("content")
        return str(path)

    return store, writes


@pytest.mark.asyncio
async def test_a_store_its_location_was_removed_under_is_stored_again(tmp_path):
    path = tmp_path / "text_abc.txt"
    write, writes = _counting_store(path)

    async def store_then_lose_it():
        location = await write()
        # Another dataset's removal takes the location before the claim.
        async with removing_stored_files([location]) as taken:
            assert taken == [location]
            path.unlink()
        return location

    async with holding_stored_files():
        await store_claimed(store_then_lose_it)

    assert path.read_text() == "content"
    assert len(writes) == 2


@pytest.mark.asyncio
async def test_a_removal_of_another_location_does_not_store_it_again(tmp_path):
    path = tmp_path / "text_abc.txt"
    write, writes = _counting_store(path)

    async def store_while_another_file_is_removed():
        async with removing_stored_files([str(tmp_path / "text_other.txt")]):
            pass
        return await write()

    async with holding_stored_files():
        await store_claimed(store_while_another_file_is_removed)

    assert len(writes) == 1


@pytest.mark.asyncio
async def test_a_claim_on_a_location_being_removed_waits_and_stores_again(tmp_path):
    path = tmp_path / "text_abc.txt"
    path.write_text("content")
    write, writes = _counting_store(path)
    taken = asyncio.Event()
    delete = asyncio.Event()

    async def removal_in_another_dataset():
        async with removing_stored_files([str(path)]):
            taken.set()
            await delete.wait()
            path.unlink()

    removing = asyncio.create_task(removal_in_another_dataset())
    await taken.wait()

    async def add():
        async with holding_stored_files():
            await store_claimed(write)

    adding = asyncio.create_task(add())
    await asyncio.sleep(0.05)
    assert not adding.done(), "The claim did not wait for the removal of its location"

    delete.set()
    await removing
    await adding
    # Written once before the claim and once after the removal it waited for.
    assert path.read_text() == "content"
    assert len(writes) == 2


@pytest.mark.asyncio
async def test_a_removal_in_progress_does_not_hold_up_a_store_of_another_location(tmp_path):
    taken = asyncio.Event()
    finish = asyncio.Event()

    async def slow_removal():
        async with removing_stored_files([str(tmp_path / "text_other.txt")]):
            taken.set()
            await finish.wait()  # its reference check and deletes

    removing = asyncio.create_task(slow_removal())
    await taken.wait()

    write, writes = _counting_store(tmp_path / "text_abc.txt")
    async with holding_stored_files():
        await asyncio.wait_for(store_claimed(write), timeout=1)

    assert len(writes) == 1
    finish.set()
    await removing
