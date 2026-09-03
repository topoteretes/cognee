"""Refusals are declared, not guessed.

Every reason a chunk-level update falls back used to surface as one free-text
message and one log line, so an unsupported backend, an incompatible chunker
and a first ingestion were indistinguishable in the logs. Each raise site now
carries a typed reason.

The backend gate is a declared capability rather than a provider-name list, so
a graph adapter registered at runtime through ``use_graph_adapter()`` can take
the incremental path by satisfying the contract.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from starlette.datastructures import UploadFile

from cognee.api.v1.update.incremental import (
    IncrementalUpdateNotPossible,
    RefusalReason,
    _changed_staged_metadata,
    _require_chunk_scoped_ownership,
    _require_stored_chunks_tile,
    _stage_new_content,
    incremental_update,
)
from cognee.infrastructure.databases.provenance import (
    make_chunk_source_ref_key,
    make_source_ref_key,
)
from cognee.modules.chunking.chunk_policy import DEFAULT_CHUNK_POLICY
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.tasks.ingestion.data_item import DataItem


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


def test_direct_text_ignores_its_content_derived_internal_name():
    old = SimpleNamespace(
        name="text_old.txt",
        extension="txt",
        mime_type="text/plain",
        original_extension="txt",
        original_mime_type="text/plain",
        loader_engine="text_loader",
    )
    staged = SimpleNamespace(**vars(old))
    staged.name = "text_new.txt"

    assert _changed_staged_metadata("new text", old, staged) == []


def test_user_named_upload_requires_full_update_when_renamed():
    old = SimpleNamespace(
        name="old.txt",
        extension="txt",
        mime_type="text/plain",
        original_extension="txt",
        original_mime_type="text/plain",
        loader_engine="text_loader",
    )
    staged = SimpleNamespace(**vars(old))
    staged.name = "new.txt"

    assert _changed_staged_metadata(SimpleNamespace(filename="new.txt"), old, staged) == ["name"]


def test_wrapped_user_named_upload_requires_full_update_when_renamed():
    old = SimpleNamespace(
        name="old.txt",
        extension="txt",
        mime_type="text/plain",
        original_extension="txt",
        original_mime_type="text/plain",
        loader_engine="text_loader",
    )
    staged = SimpleNamespace(**vars(old))
    staged.name = "new.txt"

    wrapped_upload = DataItem(data=SimpleNamespace(filename="new.txt"))
    assert _changed_staged_metadata(wrapped_upload, old, staged) == ["name"]


@pytest.mark.asyncio
async def test_v1_chunk_ownership_refuses_incremental_baseline(monkeypatch):
    import cognee.api.v1.update.incremental as incremental

    dataset_id, data_id, chunk_id = uuid4(), uuid4(), uuid4()
    snapshot = SimpleNamespace(source_ref_keys=[make_source_ref_key(dataset_id, data_id)])
    engine = SimpleNamespace(get_node_delete_data=AsyncMock(return_value={str(chunk_id): snapshot}))
    monkeypatch.setattr(incremental, "get_graph_engine", AsyncMock(return_value=engine))

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await _require_chunk_scoped_ownership(
            [{"id": str(chunk_id)}], dataset_id=dataset_id, data_id=data_id
        )

    assert raised.value.reason is RefusalReason.NO_BASELINE


@pytest.mark.asyncio
async def test_v2_chunk_ownership_is_an_incremental_baseline(monkeypatch):
    import cognee.api.v1.update.incremental as incremental

    dataset_id, data_id, chunk_id = uuid4(), uuid4(), uuid4()
    snapshot = SimpleNamespace(
        source_ref_keys=[make_chunk_source_ref_key(dataset_id, data_id, chunk_id)]
    )
    engine = SimpleNamespace(get_node_delete_data=AsyncMock(return_value={str(chunk_id): snapshot}))
    monkeypatch.setattr(incremental, "get_graph_engine", AsyncMock(return_value=engine))

    await _require_chunk_scoped_ownership(
        [{"id": str(chunk_id)}], dataset_id=dataset_id, data_id=data_id
    )


def test_stored_chunks_must_tile_the_stored_text():
    _require_stored_chunks_tile([{"text": "first"}, {"text": " second"}], "first second")

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        _require_stored_chunks_tile([{"text": "first"}, {"text": " third"}], "first second")

    assert raised.value.reason is RefusalReason.CHUNKS_NOT_TILING


@pytest.mark.asyncio
async def test_same_name_upload_uses_isolated_staging_path(monkeypatch, tmp_path):
    import cognee.api.v1.update.incremental as incremental

    current_original = tmp_path / "report.txt"
    current_original.write_bytes(b"old content")
    saved_names = []

    async def _save_staged(file, filename):
        saved_names.append(filename)
        destination = tmp_path / filename
        file.seek(0)
        destination.write_bytes(file.read())
        file.seek(0)
        return destination.as_uri()

    async def _load_staged(path, _preferred_loaders):
        return path, SimpleNamespace(loader_name="text_loader")

    upload_file = tempfile.SpooledTemporaryFile()
    upload_file.write(b"new content")
    upload_file.seek(0)
    upload = UploadFile(file=upload_file, filename="report.txt")

    monkeypatch.setattr(incremental, "save_data_to_file", _save_staged)
    monkeypatch.setattr(incremental, "data_item_to_text_file", _load_staged)

    staged = await _stage_new_content(upload, None)

    assert saved_names[0].startswith("staged_original_")
    assert saved_names[0].endswith(".txt")
    assert saved_names[0] != current_original.name
    assert current_original.read_bytes() == b"old content"
    assert Path(staged.original_data_location.removeprefix("file://")) != current_original
    assert staged.name == "report"


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
    """A non-in-tree adapter clears the cheap adapter gate by declaring support.

    Impossible before: the gate was a hardcoded {kuzu, ladybug, neo4j,
    postgres} name set, so a community adapter satisfying the contract could
    never participate. Provenance and vector capabilities are checked later,
    inside the selected dataset context.
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


@pytest.mark.asyncio
async def test_unmarked_graph_refuses_before_incremental_work(monkeypatch):
    """Legacy relational-ledger graphs stay on the full update path."""
    import cognee.api.v1.update.incremental as incremental

    data_id, dataset_id = uuid4(), uuid4()

    class _Adapter:
        supports_incremental_chunk_updates = True

    class _Vector:
        supports_payload_update = True

    class _Context:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def _get_graph_engine():
        return _Adapter()

    async def _authorize(_user, _dataset_id, _permission):
        return SimpleNamespace(id=dataset_id, owner_id=uuid4())

    monkeypatch.setattr(incremental, "get_graph_engine", _get_graph_engine)
    monkeypatch.setattr(incremental, "get_vector_engine_async", AsyncMock(return_value=_Vector()))
    monkeypatch.setattr(incremental, "get_authorized_dataset", _authorize)
    monkeypatch.setattr(
        incremental, "get_dataset_data", AsyncMock(return_value=[SimpleNamespace(id=data_id)])
    )
    monkeypatch.setattr(
        incremental,
        "get_data",
        AsyncMock(return_value=SimpleNamespace(id=data_id, raw_data_location="old.txt")),
    )
    monkeypatch.setattr(incremental, "dataset_lock", lambda _dataset_id: _Context())
    monkeypatch.setattr(
        incremental,
        "set_database_global_context_variables",
        lambda _dataset_id, _user_id: _Context(),
    )
    monkeypatch.setattr(incremental, "stores_provenance_in_graph", AsyncMock(return_value=False))
    run_incremental = AsyncMock()
    monkeypatch.setattr(incremental, "_run_incremental_update", run_incremental)

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await incremental_update(
            data_id,
            "replacement",
            dataset_id,
            user=SimpleNamespace(id=uuid4()),
        )

    assert raised.value.reason is RefusalReason.UNSUPPORTED_BACKEND
    run_incremental.assert_not_awaited()


def test_the_capability_is_off_by_default():
    """An adapter that has not been verified cannot opt in by accident."""
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    assert GraphDBInterface.supports_incremental_chunk_updates is False


def test_ladybug_declares_the_capability():
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    assert LadybugAdapter.supports_incremental_chunk_updates is True


def test_postgres_demo_declares_the_capability_with_its_narrow_move():
    """The demo adapter takes the incremental path: it stores provenance in-graph,
    its connections carry true endpoints, and it overrides update_chunk_index."""
    pytest.importorskip("asyncpg")
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
    from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter

    assert PostgresDemoAdapter.supports_incremental_chunk_updates is True
    assert PostgresDemoAdapter.update_chunk_index is not GraphDBInterface.update_chunk_index


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
        return "para\n" if location == "/tmp/old.txt" else "new\n"

    async def _stored_chunks(*_args, **_kwargs):
        return stored

    async def _stage(*_args, **_kwargs):
        return SimpleNamespace(raw_data_location="/tmp/new.txt", content_hash="NEW")

    monkeypatch.setattr(incremental, "get_data", _get_data)
    monkeypatch.setattr(incremental, "_read_processed_text", _read)
    monkeypatch.setattr(incremental, "_get_stored_chunks", _stored_chunks)
    monkeypatch.setattr(incremental, "_require_chunk_scoped_ownership", AsyncMock())
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
async def test_changed_content_with_changed_metadata_refuses_before_planning(monkeypatch):
    import cognee.api.v1.update.incremental as incremental

    data_id, dataset_id = uuid4(), uuid4()
    old_data = SimpleNamespace(
        id=data_id,
        raw_data_location="/tmp/old.txt",
        content_hash="OLD",
        external_metadata={},
        extension="txt",
        mime_type="text/plain",
        original_extension="txt",
        original_mime_type="text/plain",
        loader_engine="text_loader",
    )
    staged = SimpleNamespace(
        raw_data_location="/tmp/new.md",
        content_hash="NEW",
        extension="md",
        mime_type="text/markdown",
        original_extension="md",
        original_mime_type="text/markdown",
        loader_engine="text_loader",
    )

    monkeypatch.setattr(incremental, "get_data", AsyncMock(return_value=old_data))
    monkeypatch.setattr(
        incremental,
        "_read_processed_text",
        AsyncMock(side_effect=["old content", "new content"]),
    )
    monkeypatch.setattr(
        incremental,
        "_get_stored_chunks",
        AsyncMock(return_value=[{"id": str(uuid4()), "text": "old content"}]),
    )
    monkeypatch.setattr(incremental, "_require_chunk_scoped_ownership", AsyncMock())
    monkeypatch.setattr(incremental, "_stage_new_content", AsyncMock(return_value=staged))

    with pytest.raises(IncrementalUpdateNotPossible) as raised:
        await incremental._stage_and_plan(
            data_id,
            "new content",
            SimpleNamespace(id=dataset_id),
            SimpleNamespace(id=uuid4()),
            None,
            None,
            chunker=TextChunker,
            policy=DEFAULT_CHUNK_POLICY,
        )

    assert raised.value.reason is RefusalReason.UNSUPPORTED_METADATA


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
async def test_a_dataset_collaborator_may_update_a_row_they_do_not_own(monkeypatch):
    """Dataset permissions authorize this path, not Data.owner_id.

    datasets.delete_data lets a collaborator holding the dataset ACL delete a
    row the dataset owner ingested, and the full fallback reaches the row
    through it. Rejecting the same caller here made the permission update()
    demanded depend on which branch it took — and the incremental branch, the
    default one, was the stricter.
    """
    import cognee.api.v1.update.incremental as incremental

    data_id, dataset_id, owner_id = uuid4(), uuid4(), uuid4()
    collaborator = SimpleNamespace(id=uuid4())

    class _Context:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class _Adapter:
        supports_incremental_chunk_updates = True

    class _Vector:
        supports_payload_update = True

    seen_owner = {}

    def _context(_dataset_id, user_id):
        seen_owner["user_id"] = user_id
        return _Context()

    monkeypatch.setattr(incremental, "get_graph_engine", AsyncMock(return_value=_Adapter()))
    monkeypatch.setattr(incremental, "get_vector_engine_async", AsyncMock(return_value=_Vector()))
    monkeypatch.setattr(
        incremental,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=dataset_id, owner_id=owner_id)),
    )
    monkeypatch.setattr(
        incremental, "get_dataset_data", AsyncMock(return_value=[SimpleNamespace(id=data_id)])
    )
    # The row belongs to the dataset owner, not the caller.
    get_data = AsyncMock(
        return_value=SimpleNamespace(id=data_id, owner_id=owner_id, raw_data_location="old.txt")
    )
    monkeypatch.setattr(incremental, "get_data", get_data)
    monkeypatch.setattr(incremental, "dataset_lock", lambda _dataset_id: _Context())
    monkeypatch.setattr(incremental, "set_database_global_context_variables", _context)
    monkeypatch.setattr(incremental, "stores_provenance_in_graph", AsyncMock(return_value=True))
    run_incremental = AsyncMock(return_value={"status": "updated"})
    monkeypatch.setattr(incremental, "_run_incremental_update", run_incremental)

    result = await incremental_update(data_id, "replacement", dataset_id, user=collaborator)

    assert result == {"status": "updated"}
    run_incremental.assert_awaited()
    assert get_data.await_args.kwargs["verify_owner"] is False, (
        "row ownership must not gate a caller the dataset ACL already authorized"
    )
    assert seen_owner["user_id"] == owner_id, (
        "the dataset's databases resolve by dataset owner, as in run_tasks and delete_data"
    )


@pytest.mark.asyncio
async def test_the_recorded_budget_lookup_reads_the_dataset_owners_store(monkeypatch):
    """The refusal path reads the same store the incremental path writes.

    recorded_chunk_budget hands the full rebuild the token budget the stored
    chunks were cut against. Opening the dataset as the CALLER sent a
    collaborator's lookup to a different per-user store, where it found no
    chunks and reported "no recorded budget" — so the rebuild silently cut the
    document at the current default, which is the granularity drift this
    helper exists to prevent.
    """
    import cognee.api.v1.update.incremental as incremental

    data_id, dataset_id, owner_id = uuid4(), uuid4(), uuid4()
    collaborator = SimpleNamespace(id=uuid4())

    class _Context:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    seen_owner = {}

    def _context(_dataset_id, user_id):
        seen_owner["user_id"] = user_id
        return _Context()

    monkeypatch.setattr(
        incremental,
        "get_authorized_dataset",
        AsyncMock(return_value=SimpleNamespace(id=dataset_id, owner_id=owner_id)),
    )
    monkeypatch.setattr(incremental, "set_database_global_context_variables", _context)
    monkeypatch.setattr(
        incremental,
        "_get_stored_chunks",
        AsyncMock(return_value=[{"id": str(uuid4()), "max_chunk_tokens": 512}]),
    )
    monkeypatch.setattr(incremental, "get_max_chunk_tokens", AsyncMock(return_value=8192))

    budget = await incremental.recorded_chunk_budget(data_id, dataset_id, collaborator)

    assert budget == 512, "the document's recorded budget must survive the fallback"
    assert seen_owner["user_id"] == owner_id, (
        "the budget lookup resolves the dataset's store by dataset owner, like the write path"
    )


@pytest.mark.asyncio
async def test_the_recorded_budget_lookup_never_fails_the_update(monkeypatch):
    """It is an optimization on the fallback path, so it degrades, never raises."""
    import cognee.api.v1.update.incremental as incremental

    from cognee.modules.users.exceptions import PermissionDeniedError

    monkeypatch.setattr(
        incremental,
        "get_authorized_dataset",
        AsyncMock(side_effect=PermissionDeniedError("nope")),
    )

    assert (
        await incremental.recorded_chunk_budget(uuid4(), uuid4(), SimpleNamespace(id=uuid4()))
        is None
    )


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
