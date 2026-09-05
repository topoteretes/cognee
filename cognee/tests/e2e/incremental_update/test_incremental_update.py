"""Integration tests for chunk-level incremental updates (update endpoint).

Runs on the default local stack (kuzu + lancedb + sqlite) in a scratch root
with ENABLE_BACKEND_ACCESS_CONTROL=true, mocked LLM structured output and
mocked embeddings — CI-safe, no API keys.

Everything runs inside one test coroutine: cognee's cached engines bind to the
running event loop, so a single loop hosts the whole scenario suite.
"""

import asyncio
import contextvars
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

CHUNK_TOKENS = 60
MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def incremental_env():
    """Scratch roots, env overrides, config-cache resets, and the LLM mock."""
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_incr_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TRIPLET_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        # Multi-tenant isolated DBs need a backend that supports them
        # (ladybug/lancedb/sqlite/postgres). Neo4j Community cannot CREATE
        # DATABASE, so its run uses INCR_TEST_ACL=false (single-user mode).
        ENABLE_BACKEND_ACCESS_CONTROL=os.environ.get("INCR_TEST_ACL", "true"),
    )

    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=f"marker {n}") for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original_acreate = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    import cognee.api.v1.update.incremental as incremental_module

    async def _fixed_chunk_tokens():
        return CHUNK_TOKENS

    original_budget = incremental_module.get_max_chunk_tokens
    incremental_module.get_max_chunk_tokens = _fixed_chunk_tokens

    yield root

    LLMGateway.acreate_structured_output = original_acreate
    incremental_module.get_max_chunk_tokens = original_budget
    shutil.rmtree(root, ignore_errors=True)


def _paragraph(i: int) -> str:
    words = " ".join(f"w{i}{j:02d}" for j in range(38))
    return f"Paragraph {i} ENTP{i} ENTSHARED. {words}.\n"


async def _doc_chunk_nodes(document_id, text):
    """Full chunk nodes of a document, ordered by their position in text."""
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.graph.methods.delete_chunks_incremental import edge_endpoints

    graph = await get_graph_engine()
    chunk_ids = []
    for source, edge, target in await graph.get_connections(str(document_id)):
        if "is_part_of" not in str(edge.get("relationship_name", "")):
            continue
        source_id, target_id = edge_endpoints(source, edge, target)
        if target_id == str(document_id) and source_id != str(document_id):
            chunk_ids.append(source_id)
    nodes = await graph.get_nodes(chunk_ids) if chunk_ids else []
    return sorted(
        (n for n in nodes if n.get("text") is not None), key=lambda n: text.find(n["text"])
    )


async def _entity_names():
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    return {
        str(props.get("name", "")).lower() for _id, props in nodes if props.get("type") == "Entity"
    }


async def _stored_text(user, data_id) -> str:
    from cognee.api.v1.update.incremental import _read_processed_text
    from cognee.modules.data.methods import get_data

    return await _read_processed_text((await get_data(user.id, data_id)).raw_data_location)


@pytest.mark.asyncio
async def test_incremental_update_full_flow(incremental_env):
    await reset_backend_state()
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import create_user, get_default_user

    pristine = contextvars.copy_context()

    async def update_like_an_api_request(*args, **kwargs):
        """Fresh-context task: how a real API request calls update()."""
        loop = asyncio.get_running_loop()
        return await loop.create_task(cognee.update(*args, **kwargs), context=pristine.copy())

    text_v1 = "".join(_paragraph(i) for i in range(10))
    await cognee.add(text_v1, dataset_name="incr_it")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "incr_it")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)

    from cognee.context_global_variables import set_database_global_context_variables

    async with set_database_global_context_variables(dataset.id, user.id):
        pass  # graph/vector configs persist for this task's inspections

    data_id = (await get_dataset_data(dataset.id))[0].id
    old_nodes = await _doc_chunk_nodes(data_id, text_v1)
    assert len(old_nodes) >= 8, "expected a multi-chunk document"
    assert "entp4" in await _entity_names()

    # --- Edit straddling the ENTP4 marker chunk and its neighbour ----------- #
    marker_idx = next(i for i, n in enumerate(old_nodes) if "ENTP4" in n["text"])
    offsets, cursor = [], 0
    for node in old_nodes:
        offsets.append((cursor, cursor + len(node["text"])))
        cursor += len(node["text"])
    edit_start = text_v1.index("ENTP4")
    mid_next = text_v1.index(" ", (offsets[marker_idx + 1][0] + offsets[marker_idx + 1][1]) // 2)
    insertion = " ".join(f"ENTNEW{j:03d} fresh{j:03d}" for j in range(75))
    text_v2 = text_v1[:edit_start] + insertion + " CHANGED " + text_v1[mid_next:]

    result = await update_like_an_api_request(data_id, text_v2, dataset.id, user=user)
    assert result["status"] == "incremental", f"chunk-level path did not run: {result}"
    assert result["deleted_chunks"] == 2
    assert result["kept_chunks"] == len(old_nodes) - 2
    assert await _stored_text(user, data_id) == text_v2

    new_nodes = await _doc_chunk_nodes(data_id, text_v2)
    old_ids = {str(n["id"]) for n in old_nodes}
    new_ids = {str(n["id"]) for n in new_nodes}
    deleted = {str(old_nodes[marker_idx]["id"]), str(old_nodes[marker_idx + 1]["id"])}
    assert deleted.isdisjoint(new_ids), "replaced chunk nodes must be gone"
    assert (old_ids - deleted) <= new_ids, "surviving chunks must keep their node ids"
    assert "".join(n["text"] for n in new_nodes) == text_v2, "no-loss reassembly"
    assert [int(n["chunk_index"]) for n in new_nodes] == list(range(len(new_nodes)))
    assert all(
        n.get("content_hash") == hashlib.sha256(n["text"].encode()).hexdigest() for n in new_nodes
    )

    entities = await _entity_names()
    assert "entp4" not in entities, "chunk-exclusive entity must be orphan-deleted"
    assert "entshared" in entities, "shared entity must survive"
    assert "entnew001" in entities, "replacement-region entities must be ingested"

    # --- Multi-region: three disjoint edits handled as three small regions --- #
    total_before = len(new_nodes)
    text_multi = (
        "MULTI HEAD LINE\n"
        + text_v2[: len(text_v2) // 2]
        + "MID INSERT LINE\n"
        + text_v2[len(text_v2) // 2 :]
        + "MULTI TAIL LINE\n"
    )
    result_multi = await update_like_an_api_request(data_id, text_multi, dataset.id, user=user)
    assert result_multi["status"] == "incremental"
    assert result_multi["regions"] == 3, f"expected three regions: {result_multi}"
    assert result_multi["kept_chunks"] >= total_before - 6, (
        f"disjoint edits must keep the untouched middle: {result_multi}"
    )
    assert await _stored_text(user, data_id) == text_multi
    multi_nodes = await _doc_chunk_nodes(data_id, text_multi)
    assert "".join(n["text"] for n in multi_nodes) == text_multi
    assert sorted(int(n["chunk_index"]) for n in multi_nodes) == list(range(len(multi_nodes)))

    # Renumbering must reach the VECTOR payloads too (citations read
    # chunk_index from there) — via update_payload, without re-embedding.
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    vector_engine = await get_vector_engine_async()
    assert vector_engine.supports_payload_update, "in-tree adapters must support update_payload"
    vector_rows = await vector_engine.retrieve(
        "DocumentChunk_text", [str(n["id"]) for n in multi_nodes]
    )
    vector_index_by_id = {str(row.id): row.payload.get("chunk_index") for row in vector_rows}
    assert len(vector_rows) == len(multi_nodes)
    for node in multi_nodes:
        assert vector_index_by_id[str(node["id"])] == int(node["chunk_index"]), (
            "vector payload chunk_index must match the graph after renumbering"
        )
    text_v2 = text_multi  # concurrency section below edits on top of this state

    # --- Concurrent updates on the same document serialize on the lock ------- #
    text_v3 = text_v2.replace("CHANGED", "CHANGED-A ENTV3A", 1)
    text_v4 = text_v2.replace("CHANGED", "CHANGED-B ENTV3B", 1)
    results = await asyncio.gather(
        update_like_an_api_request(data_id, text_v3, dataset.id, user=user),
        update_like_an_api_request(data_id, text_v4, dataset.id, user=user),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), f"concurrent updates: {results}"
    final_text = await _stored_text(user, data_id)
    assert final_text in (text_v3, text_v4), "one of the two edits must have won"
    final_nodes = await _doc_chunk_nodes(data_id, final_text)
    assert "".join(n["text"] for n in final_nodes) == final_text
    assert [int(n["chunk_index"]) for n in final_nodes] == list(range(len(final_nodes)))

    # --- Crash between ingest and delete heals through the fallback ---------- #
    import cognee.api.v1.update.incremental as incremental_module

    original_delete = incremental_module.delete_chunks_incremental

    async def exploding_delete(chunk_ids, dataset_id, data_id):
        raise RuntimeError("simulated crash before deletion")

    incremental_module.delete_chunks_incremental = exploding_delete
    text_v5 = final_text.replace("Paragraph 7", "Paragraph 7 ENTV5", 1)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            await update_like_an_api_request(data_id, text_v5, dataset.id, user=user)
    finally:
        incremental_module.delete_chunks_incremental = original_delete

    # Retry with the same content: stored chunks no longer tile the stored
    # text (old + new region chunks coexist), so the full update takes over.
    retry = await update_like_an_api_request(data_id, text_v5, dataset.id, user=user)
    assert not (isinstance(retry, dict) and retry.get("status") == "incremental"), (
        "retry after crash must fall back to the full update"
    )
    # The full update is pinned to the existing row too, so callers keep the
    # same handle even when incremental preconditions fail.
    healed_data = (await get_dataset_data(dataset.id))[0]
    assert healed_data.id == data_id
    healed_text = await _stored_text(user, data_id)
    assert healed_text == text_v5
    healed_nodes = await _doc_chunk_nodes(data_id, healed_text)
    assert "".join(n["text"] for n in healed_nodes) == healed_text, "fallback healed the graph"

    # --- The router shape: a single UploadFile in a list ---------------------- #
    # FastAPI backs UploadFile with a SpooledTemporaryFile — classify() keys on
    # that exact type, so the fixture must too.
    import tempfile

    from starlette.datastructures import UploadFile

    def _upload(content: bytes, filename: str) -> UploadFile:
        spooled = tempfile.SpooledTemporaryFile()
        spooled.write(content)
        spooled.seek(0)
        return UploadFile(file=spooled, filename=filename)

    text_v6 = healed_text.replace("Paragraph 3", "Paragraph 3 ENTV6", 1)
    # Keep the existing user-visible metadata. A rename intentionally takes
    # the full path; this case exercises safe same-name staging instead.
    upload = _upload(
        text_v6.encode("utf-8"),
        f"{healed_data.name}.{healed_data.original_extension}",
    )
    result6 = await update_like_an_api_request(data_id, [upload], dataset.id, user=user)
    assert isinstance(result6, dict) and result6.get("status") == "incremental", (
        f"single-UploadFile update must run chunk-level: {result6}"
    )
    healed_text = await _stored_text(user, data_id)
    assert healed_text == text_v6, "UploadFile content must land as the stored text"

    # --- Permissions: non-permitted user is rejected, nothing changes -------- #
    from uuid import uuid4

    intruder = await create_user(f"intruder_{uuid4().hex[:8]}@example.com", "pw")
    with pytest.raises(Exception) as denied:
        await update_like_an_api_request(
            data_id, healed_text + " HACKED", dataset.id, user=intruder
        )
    assert (
        "Permission" in type(denied.value).__name__ or "Unauthorized" in type(denied.value).__name__
    )
    assert await _stored_text(user, data_id) == healed_text

    # --- Multi-item payload: refused outright (one document per update) ------- #
    # Reconciled contract: update() replaces exactly ONE document. The old
    # fall-back-to-full-flow behavior multiplied documents and churned ids.
    from cognee.modules.ingestion.exceptions import IngestionError

    uploads = [
        _upload(b"first file", "a.txt"),
        _upload(b"second file", "b.txt"),
    ]
    with pytest.raises(IngestionError):
        await update_like_an_api_request(data_id, uploads, dataset.id, user=user)
    assert await _stored_text(user, data_id) == healed_text, (
        "a refused multi-item update must not touch the stored document"
    )
