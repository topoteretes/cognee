"""Incremental chunk cleanup delegates ownership decisions to the shared planner."""

import asyncio
from uuid import uuid4

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.modules.graph.methods.delete_chunks_incremental import delete_chunks_incremental


class _RecordingUnified:
    def __init__(self):
        self.calls = []

    async def delete_by_source_refs(self, source_refs):
        self.calls.append(list(source_refs))


def test_all_deleted_chunks_go_through_the_planner_in_one_pass(monkeypatch):
    import cognee.modules.graph.methods.delete_chunks_incremental as deletion

    dataset_id, data_id = uuid4(), uuid4()
    chunk_ids = [uuid4(), uuid4()]
    unified = _RecordingUnified()

    async def _get_unified_engine():
        return unified

    monkeypatch.setattr(deletion, "get_unified_engine", _get_unified_engine)
    asyncio.run(delete_chunks_incremental(chunk_ids, dataset_id, data_id))

    assert unified.calls == [
        [make_chunk_source_ref_key(dataset_id, data_id, chunk_id) for chunk_id in chunk_ids]
    ], "one planner pass carrying every retired chunk key, not one pass per chunk"


def test_duplicate_chunk_ids_are_retired_once(monkeypatch):
    import cognee.modules.graph.methods.delete_chunks_incremental as deletion

    dataset_id, data_id, chunk_id = uuid4(), uuid4(), uuid4()
    unified = _RecordingUnified()

    async def _get_unified_engine():
        return unified

    monkeypatch.setattr(deletion, "get_unified_engine", _get_unified_engine)
    asyncio.run(delete_chunks_incremental([chunk_id, chunk_id], dataset_id, data_id))

    assert unified.calls == [[make_chunk_source_ref_key(dataset_id, data_id, chunk_id)]]


def test_nothing_to_delete_never_touches_the_engine(monkeypatch):
    import cognee.modules.graph.methods.delete_chunks_incremental as deletion

    async def _get_unified_engine():
        raise AssertionError("no chunks to retire, the engine must not be resolved")

    monkeypatch.setattr(deletion, "get_unified_engine", _get_unified_engine)
    assert asyncio.run(delete_chunks_incremental([], uuid4(), uuid4())) is None
