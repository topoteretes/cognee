"""Refusals are declared, not guessed.

Every reason a chunk-level update falls back used to surface as one free-text
message and one log line, so an unsupported backend, an incompatible chunker
and a first ingestion were indistinguishable in the logs. Each raise site now
carries a typed reason.

The backend gate is a declared capability rather than a provider-name list, so
a graph adapter registered at runtime through ``use_graph_adapter()`` can take
the incremental path by satisfying the contract.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.update.incremental import (
    IncrementalUpdateNotPossible,
    RefusalReason,
    incremental_update,
)
from cognee.modules.chunking.chunk_policy import DEFAULT_CHUNK_POLICY
from cognee.modules.chunking.TextChunker import TextChunker


def test_every_reason_is_distinguishable():
    """The whole point of the enum: no two reasons share a value."""
    values = [reason.value for reason in RefusalReason]
    assert len(values) == len(set(values))


def test_refusal_carries_its_reason():
    error = IncrementalUpdateNotPossible("nope", RefusalReason.UNSUPPORTED_CHUNKER)
    assert error.reason is RefusalReason.UNSUPPORTED_CHUNKER
    assert str(error) == "nope"


def test_refusal_defaults_to_no_baseline():
    """Untagged raise sites must still produce a usable structured field."""
    assert IncrementalUpdateNotPossible("nope").reason is RefusalReason.NO_BASELINE


@pytest.mark.asyncio
async def test_unsupported_backend_refuses_before_touching_anything(monkeypatch):
    """A backend that does not declare the capability refuses immediately.

    Refusing here — before permissions, before staging — is what keeps the
    fallback cheap for backends that can never take this path.
    """
    import cognee.api.v1.update.incremental as incremental

    class _StubAdapter:
        supports_incremental_chunk_updates = False

    async def _get_graph_engine():
        return _StubAdapter()

    monkeypatch.setattr(incremental, "get_graph_engine", _get_graph_engine)

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await incremental_update(uuid4(), "text", uuid4(), user=None)

    assert raised.value.reason is RefusalReason.UNSUPPORTED_BACKEND


@pytest.mark.asyncio
async def test_declaring_the_capability_is_enough_to_pass_the_gate(monkeypatch):
    """A non-in-tree adapter clears the gate by declaring support.

    Impossible before: the gate was a hardcoded {kuzu, ladybug, neo4j,
    postgres} name set, so a community adapter satisfying the contract could
    never participate. The call still fails afterwards (no user), but it fails
    PAST the backend check — which is what this pins.
    """
    import cognee.api.v1.update.incremental as incremental

    class _CommunityAdapter:
        supports_incremental_chunk_updates = True

    async def _get_graph_engine():
        return _CommunityAdapter()

    monkeypatch.setattr(incremental, "get_graph_engine", _get_graph_engine)

    with pytest.raises(Exception) as raised:
        await incremental_update(uuid4(), "text", uuid4(), user=None)

    assert not (
        isinstance(raised.value, IncrementalUpdateNotPossible)
        and raised.value.reason is RefusalReason.UNSUPPORTED_BACKEND
    )


def test_the_capability_is_off_by_default():
    """An adapter that has not been verified cannot opt in by accident."""
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    assert GraphDBInterface.supports_incremental_chunk_updates is False


def test_ladybug_declares_the_capability():
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    assert LadybugAdapter.supports_incremental_chunk_updates is True


def test_postgres_declares_the_capability():
    pytest.importorskip("asyncpg")
    from cognee.infrastructure.databases.graph.postgres.adapter import PostgresDemoAdapter

    assert PostgresDemoAdapter.supports_incremental_chunk_updates is True


def test_neo4j_declares_the_capability():
    pytest.importorskip("neo4j")
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    assert Neo4jAdapter.supports_incremental_chunk_updates is True


def test_the_provider_name_gate_is_gone():
    """The name set could not see a runtime-registered adapter; it must not return."""
    import cognee.api.v1.update.incremental as incremental

    assert not hasattr(incremental, "SUPPORTED_GRAPH_PROVIDERS")


@pytest.mark.asyncio
async def test_a_document_built_by_another_chunker_is_refused_by_name(monkeypatch):
    """The point of chunker_id: say WHY, instead of failing the tiling check.

    Chunkers disagree on boundaries, so a document can only be updated by the
    one that built it. Without recorded identity the mismatch surfaced as
    "stored chunk 0 does not tile the stored document text" — the same error a
    never-cognified document produces.
    """
    import cognee.api.v1.update.incremental as incremental

    stored = [{"id": str(uuid4()), "text": "para\n", "chunk_index": 0, "chunker_id": "other_v1"}]

    async def _get_data(*_args, **_kwargs):
        return SimpleNamespace(
            id=uuid4(), raw_data_location="/tmp/old.txt", content_hash="OLD", external_metadata={}
        )

    async def _read(location):
        return "old\n" if location == "/tmp/old.txt" else "new\n"

    async def _stored_chunks(*_args, **_kwargs):
        return stored

    async def _stage(*_args, **_kwargs):
        return SimpleNamespace(raw_data_location="/tmp/new.txt", content_hash="NEW")

    monkeypatch.setattr(incremental, "get_data", _get_data)
    monkeypatch.setattr(incremental, "_read_processed_text", _read)
    monkeypatch.setattr(incremental, "_get_stored_chunks", _stored_chunks)
    monkeypatch.setattr(incremental, "_stage_new_content", _stage)
    monkeypatch.setattr(incremental, "_build_document", lambda *a, **k: SimpleNamespace(id=uuid4()))

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await incremental._stage_and_plan(
            uuid4(),
            "new",
            SimpleNamespace(id=uuid4()),
            SimpleNamespace(id=uuid4()),
            None,
            None,
            chunker=TextChunker,  # declares text_chunker_v1, stored says other_v1
            policy=DEFAULT_CHUNK_POLICY,
        )

    assert raised.value.reason is RefusalReason.UNSUPPORTED_CHUNKER
    assert "other_v1" in str(raised.value)


@pytest.mark.asyncio
async def test_write_without_delete_permission_is_denied(monkeypatch):
    """This path destroys graph state, so it needs the same permission the
    full fallback needs — and must deny with the same exception type.

    Without the delete check, the permission update() demanded depended on
    which branch it happened to take, and the faster branch was the weaker one.
    """
    import cognee.api.v1.update.incremental as incremental
    from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
    from cognee.modules.users.exceptions import PermissionDeniedError

    class _Adapter:
        supports_incremental_chunk_updates = True

    async def _get_graph_engine():
        return _Adapter()

    granted = []

    async def _authorize(_user, dataset_id, permission):
        granted.append(permission)
        if permission == "delete":
            raise PermissionDeniedError("no delete permission")
        return SimpleNamespace(id=dataset_id)

    monkeypatch.setattr(incremental, "get_graph_engine", _get_graph_engine)
    monkeypatch.setattr(incremental, "get_authorized_dataset", _authorize)

    with pytest.raises(UnauthorizedDataAccessError):
        await incremental_update(uuid4(), "text", uuid4(), user=SimpleNamespace(id=uuid4()))

    assert "delete" in granted, "the delete permission was never checked"


@pytest.mark.asyncio
async def test_fresh_chunks_are_extracted_in_bounded_batches(monkeypatch):
    """A big edit must not become one oversized, all-or-nothing extraction.

    The cognify pipeline bounds this through the task machinery
    (task_config={"batch_size": ...}); this path does not run through it, so
    the slicing is explicit and needs its own guard. Unbounded, a rewrite of
    most of a large document sends every replacement chunk into a single call
    with no intermediate progress.
    """
    import cognee.api.v1.update.incremental as incremental
    from cognee.modules.chunking.chunk_policy import ChunkPlan

    batches = []

    async def _extract(chunks, **_kwargs):
        batches.append(len(chunks))
        return []

    monkeypatch.setattr(incremental, "extract_graph_and_summarize", _extract)
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "publish_updated_data", AsyncMock())
    monkeypatch.setattr(incremental, "delete_chunks_incremental", AsyncMock(return_value=[]))
    monkeypatch.setattr(incremental, "prune_ledger_rows", AsyncMock())
    monkeypatch.setattr(
        incremental,
        "get_cognify_config",
        lambda: SimpleNamespace(
            chunks_per_batch=3, triplet_embedding=False, contradiction_detection=False
        ),
    )

    fresh = [SimpleNamespace(id=uuid4(), chunk_size=1) for _ in range(7)]
    bundle = {
        "staged": SimpleNamespace(),
        "document": SimpleNamespace(id=uuid4()),
        "stored_chunks": [],
        "plan": ChunkPlan(fresh=fresh, regions=1),
        "data_item": SimpleNamespace(id=uuid4()),
    }

    await incremental._write_and_publish(
        bundle, uuid4(), SimpleNamespace(id=uuid4()), None, None, None, None, uuid4()
    )

    assert batches == [3, 3, 1], f"expected bounded batches of 3, got {batches}"


@pytest.mark.asyncio
async def test_undecodable_stored_text_is_a_refusal_not_a_crash(tmp_path):
    """A pre-0.3.7 row pointing at binary content falls back instead of 500ing."""
    from cognee.api.v1.update.incremental import _read_processed_text

    binary = tmp_path / "legacy.bin"
    binary.write_bytes(b"caf\xe9 latin-1 bytes, not utf-8")

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await _read_processed_text(str(binary))

    assert raised.value.reason is RefusalReason.UNREADABLE_TEXT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
