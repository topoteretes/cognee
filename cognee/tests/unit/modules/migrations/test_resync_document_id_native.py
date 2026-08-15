"""Tests for the vector-preserving ``document_id`` re-key path used by the
fork-document rekey migration (``resync_document_id_native``).

Repointing the ``document_id`` payload scalar must NOT re-embed: the chunk text
and point id are unchanged, so the stored vector is already correct. PGVector
updates the JSON payload in place; backends without a native path (LanceDB,
etc.) return False so the caller re-embeds through the existing path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cognee.modules.migrations.versions import _vector_rekey
from cognee.modules.migrations.versions._vector_rekey import resync_document_id_native


def _named_engine(class_name: str):
    # The dispatch keys on ``vector_engine.__class__.__name__``.
    return type(class_name, (), {})()


@pytest.mark.asyncio
async def test_empty_targets_is_noop_and_native():
    # Nothing to do -> treated as handled, so the caller never re-embeds.
    assert await resync_document_id_native(_named_engine("PGVectorAdapter"), "C", {}) is True


@pytest.mark.asyncio
async def test_dispatch_pgvector_uses_native_path(monkeypatch):
    called = AsyncMock()
    monkeypatch.setattr(_vector_rekey, "_resync_pgvector_document_id", called)
    engine = _named_engine("PGVectorAdapter")
    assert await resync_document_id_native(engine, "DocumentChunk_text", {"a": "b"}) is True
    called.assert_awaited_once()


@pytest.mark.asyncio
async def test_lancedb_falls_back_to_reembed():
    # LanceDB's columnar payload schema can't take an in-place scalar update,
    # so it returns False and the caller re-embeds (current behavior).
    assert (
        await resync_document_id_native(_named_engine("LanceDBAdapter"), "C", {"a": "b"}) is False
    )


@pytest.mark.asyncio
async def test_unsupported_adapter_falls_back():
    assert (
        await resync_document_id_native(_named_engine("SomeOtherAdapter"), "C", {"a": "b"}) is False
    )
