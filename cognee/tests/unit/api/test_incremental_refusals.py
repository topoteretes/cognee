"""Refusals are declared, not guessed.

Every reason a chunk-level update falls back used to surface as one free-text
message and one log line, so an unsupported backend, an incompatible chunker
and a first ingestion were indistinguishable in the logs. Each raise site now
carries a typed reason.

The backend gate is a declared capability rather than a provider-name list, so
a graph adapter registered at runtime through ``use_graph_adapter()`` can take
the incremental path by satisfying the contract.
"""

from uuid import uuid4

import pytest

from cognee.api.v1.update.incremental import (
    IncrementalUpdateNotPossible,
    RefusalReason,
    incremental_update,
)


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
