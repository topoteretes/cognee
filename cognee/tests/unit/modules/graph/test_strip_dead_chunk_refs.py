"""Incremental chunk cleanup delegates ownership decisions to the shared planner."""

import asyncio
from uuid import uuid4

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.modules.graph.methods.delete_chunks_incremental import delete_chunks_incremental


class _RecordingUnified:
    def __init__(self):
        self.deleted_refs = []

    async def delete_by_source_ref(self, source_ref):
        self.deleted_refs.append(source_ref)


def test_each_deleted_chunk_uses_the_provenance_planner(monkeypatch):
    import cognee.modules.graph.methods.delete_chunks_incremental as deletion

    dataset_id, data_id = uuid4(), uuid4()
    chunk_ids = [uuid4(), uuid4()]
    unified = _RecordingUnified()

    async def _get_unified_engine():
        return unified

    monkeypatch.setattr(deletion, "get_unified_engine", _get_unified_engine)
    asyncio.run(delete_chunks_incremental(chunk_ids, dataset_id, data_id))

    assert unified.deleted_refs == [
        make_chunk_source_ref_key(dataset_id, data_id, chunk_id) for chunk_id in chunk_ids
    ]


def test_duplicate_chunk_ids_are_deleted_once(monkeypatch):
    import cognee.modules.graph.methods.delete_chunks_incremental as deletion

    dataset_id, data_id, chunk_id = uuid4(), uuid4(), uuid4()
    unified = _RecordingUnified()

    async def _get_unified_engine():
        return unified

    monkeypatch.setattr(deletion, "get_unified_engine", _get_unified_engine)
    asyncio.run(delete_chunks_incremental([chunk_id, chunk_id], dataset_id, data_id))

    assert unified.deleted_refs == [make_chunk_source_ref_key(dataset_id, data_id, chunk_id)]
