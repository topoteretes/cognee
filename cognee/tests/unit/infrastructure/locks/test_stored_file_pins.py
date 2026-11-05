"""Tests for pins on stored files (cognee.infrastructure.locks.stored_file_pins).

A writer pins every file it stores until its rows are committed, and cleanup
skips a pinned file, so a delete in one dataset cannot reclaim a shared file
another dataset's in-flight add is about to reference.
"""

import asyncio

import pytest

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.infrastructure.locks import (
    pin_stored_file,
    stored_file_pin_scope,
    stored_file_pins,
    stored_file_removal,
)


async def _removable(key: str) -> bool:
    async with stored_file_removal(key) as removable:
        return removable


@pytest.mark.asyncio
async def test_pin_outside_a_scope_is_a_noop():
    await pin_stored_file("/data/no-scope.txt")

    assert await _removable("/data/no-scope.txt")
    assert "/data/no-scope.txt" not in stored_file_pins._entries


@pytest.mark.asyncio
async def test_removal_skips_a_pinned_file_until_the_scope_exits():
    key = "/data/pinned.txt"
    async with stored_file_pin_scope():
        await pin_stored_file(key)
        assert not await _removable(key)

    assert await _removable(key)
    # Nothing pins or uses the entry any more, so the registry does not grow.
    assert key not in stored_file_pins._entries


@pytest.mark.asyncio
async def test_scope_exit_unpins_on_exception():
    key = "/data/failed-ingest.txt"
    with pytest.raises(RuntimeError):
        async with stored_file_pin_scope():
            await pin_stored_file(key)
            raise RuntimeError("ingest failed")

    assert await _removable(key)
    assert key not in stored_file_pins._entries


@pytest.mark.asyncio
async def test_pin_waits_for_an_in_flight_removal():
    key = "/data/being-removed.txt"
    pinned = asyncio.Event()

    async def writer():
        async with stored_file_pin_scope():
            await pin_stored_file(key)
            pinned.set()
            await asyncio.sleep(0.05)

    async with stored_file_removal(key) as removable:
        assert removable
        task = asyncio.create_task(writer())
        await asyncio.sleep(0.05)
        # The removal still holds the file's lock: the pin has not landed.
        assert not pinned.is_set()

    await asyncio.wait_for(pinned.wait(), timeout=1)
    assert not await _removable(key)
    await task
    assert await _removable(key)


@pytest.mark.asyncio
async def test_nested_scope_keeps_pins_until_the_outer_scope_exits():
    key = "/data/nested.txt"
    async with stored_file_pin_scope():
        async with stored_file_pin_scope():
            await pin_stored_file(key)
        assert not await _removable(key)

    assert await _removable(key)


@pytest.mark.asyncio
async def test_child_task_pins_into_the_parent_scope():
    key = "/data/child-task.txt"
    async with stored_file_pin_scope():
        await asyncio.create_task(pin_stored_file(key))
        assert not await _removable(key)

    assert await _removable(key)


@pytest.mark.asyncio
async def test_pin_after_the_scope_closed_is_a_noop():
    # A task spawned in a scope that outlives it must not leave a pin nothing
    # would release.
    key = "/data/late-pin.txt"
    release = asyncio.Event()

    async def late_pin():
        await release.wait()
        await pin_stored_file(key)

    async with stored_file_pin_scope():
        task = asyncio.create_task(late_pin())

    release.set()
    await task
    assert await _removable(key)
    assert key not in stored_file_pins._entries


@pytest.mark.asyncio
async def test_local_store_pins_the_key_cleanup_computes(tmp_path):
    # The writer and the cleanup reach one file through different storage
    # roots (a tenant subdirectory vs the data root); both must derive one key.
    data_root = LocalFileStorage(str(tmp_path))
    tenant_root = LocalFileStorage(str(tmp_path / "tenant"))

    async with stored_file_pin_scope():
        await tenant_root.store("text_abc.txt", "body")
        assert not await _removable(data_root.stored_file_key("tenant/text_abc.txt"))

    assert await _removable(data_root.stored_file_key("tenant/text_abc.txt"))
