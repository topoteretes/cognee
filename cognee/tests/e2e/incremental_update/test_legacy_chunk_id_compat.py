"""Upgrade compatibility: graphs whose chunks carry OLD-scheme ids keep working.

Long-running deployments have graphs whose DocumentChunk ids were minted by the
released chunker (position-derived ``uuid5(NAMESPACE_OID, f"{doc_id}-{index}")``,
or bare content ids from ``chunk_by_paragraph``). The new content-derived
identity must not strand that data: every production path discovers chunks by
attributes and edges (``is_part_of`` traversal, ledger rows, provenance keys) —
never by re-deriving ids. This test pins that contract:

  1. ingest with the LEGACY id scheme (simulating a pre-upgrade deployment),
  2. run a chunk-level incremental update with the NEW code — kept chunks must
     retain their legacy ids, replaced legacy chunks (and their summaries and
     vector rows) must be fully removed, repositioned legacy chunks must get
     payload-only index updates,
  3. purge the rollback-ledger rows to force the legacy traversal delete path,
     then delete the document — the mixed-id graph must clean up completely.

Runs on the default local stack (kuzu + lancedb + sqlite), mocked LLM and
embeddings — CI-safe, no API keys.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

CHUNK_TOKENS = 60
MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def legacy_env():
    """Scratch roots, env overrides, config-cache resets, and the LLM mock."""
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_legacy_id_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
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
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    import cognee.api.v1.update.incremental as incremental_module

    async def _fixed_budget():
        return CHUNK_TOKENS

    original_budget = incremental_module.get_max_chunk_tokens
    incremental_module.get_max_chunk_tokens = _fixed_budget

    yield root

    LLMGateway.acreate_structured_output = original
    incremental_module.get_max_chunk_tokens = original_budget
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str, marker: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} {marker} {words}.\n"


async def _get_chunks(document_id):
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
        (n for n in nodes if n.get("text") is not None), key=lambda n: int(n["chunk_index"])
    )


def test_legacy_chunk_ids_survive_update_and_delete(legacy_env):
    asyncio.run(_scenario())


async def _scenario():
    await reset_backend_state()
    import cognee
    import cognee.modules.chunking.TextChunker as chunker_module
    from cognee.api.v1.update.incremental import _read_processed_text
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
    from cognee.modules.data.methods import get_data, get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    # --- phase 1: ingest under the LEGACY id scheme ------------------------ #
    real_minter = chunker_module.content_chunk_id
    minted_per_document: dict = {}

    def legacy_minter(document_id, content_hash, occurrence):
        index = minted_per_document.get(document_id, 0)
        minted_per_document[document_id] = index + 1
        return uuid5(NAMESPACE_OID, f"{document_id}-{index}")

    chunker_module.content_chunk_id = legacy_minter
    try:
        paragraphs = [_para(tag, f"ENT{tag.upper()}") for tag in ["a", "b", "c", "d", "e", "f"]]
        text_v1 = "".join(paragraphs)
        await cognee.add(text_v1, dataset_name="legacy_ids")
        user = await get_default_user()
        dataset = next(d for d in await get_datasets(user.id) if d.name == "legacy_ids")
        await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    finally:
        chunker_module.content_chunk_id = real_minter

    data_id = (await get_dataset_data(dataset.id))[0].id
    chunks = await _get_chunks(data_id)
    assert len(chunks) >= 4, "scenario needs several chunks"
    legacy_ids = [str(n["id"]) for n in chunks]
    for index, node in enumerate(chunks):
        expected = str(uuid5(NAMESPACE_OID, f"{data_id}-{index}"))
        assert str(node["id"]) == expected, "phase 1 must reproduce the released id scheme"

    # --- phase 2: incremental update on the legacy graph ------------------- #
    # Insert a new paragraph mid-document: downstream legacy chunks must shift
    # index (payload-only vector updates on legacy-id rows), and the edited
    # region must delete its legacy chunk completely.
    insert_at = sum(len(p) for p in paragraphs[:3])
    text_v2 = text_v1[:insert_at] + _para("x", "ENTX") + text_v1[insert_at:]
    result = await cognee.update(data_id, text_v2, dataset.id, user=user)
    assert isinstance(result, dict) and result.get("status") == "incremental", result

    data_id = (await get_dataset_data(dataset.id))[0].id
    stored = await _read_processed_text((await get_data(user.id, data_id)).raw_data_location)
    assert stored == text_v2

    chunks2 = await _get_chunks(data_id)
    texts2 = [n["text"] for n in chunks2]
    assert "".join(texts2) == text_v2, "graph must tile the updated text byte-exactly"
    indexes = [int(n["chunk_index"]) for n in chunks2]
    assert indexes == list(range(len(chunks2))), "indexes must stay contiguous"

    ids2 = {str(n["id"]) for n in chunks2}
    kept_legacy = [n for n in chunks2 if str(n["id"]) in set(legacy_ids)]
    assert len(kept_legacy) >= 3, "untouched chunks must RETAIN their legacy ids"
    fresh = [n for n in chunks2 if str(n["id"]) not in set(legacy_ids)]
    for node in fresh:
        expected = content_chunk_id(str(data_id), chunk_content_hash(node["text"]), 0)
        assert str(node["id"]) == str(expected), "fresh chunks must carry content-derived ids"

    deleted_legacy = [old_id for old_id in legacy_ids if old_id not in ids2]
    assert deleted_legacy, "the edited region must have replaced at least one legacy chunk"
    graph = await get_graph_engine()
    vector = await get_vector_engine_async()
    for old_id in deleted_legacy:
        assert await graph.get_node(old_id) is None, "replaced legacy chunk must leave the graph"
        summary_id = str(uuid5(UUID(old_id), "TextSummary"))
        assert await graph.get_node(summary_id) is None, "its summary must go with it"
        assert not await vector.retrieve("DocumentChunk_text", [old_id]), (
            "its vector row must go with it"
        )

    rows = await vector.retrieve("DocumentChunk_text", [str(n["id"]) for n in chunks2])
    by_id = {str(r.id): r.payload.get("chunk_index") for r in rows}
    assert len(rows) == len(chunks2) and all(
        by_id[str(n["id"])] == int(n["chunk_index"]) for n in chunks2
    ), "vector payload indexes (including legacy-id rows) must match the graph"

    # --- phase 3: attribute-based delete of the mixed-id graph ------------- #
    # Purge the rollback-ledger rows so delete cannot use recorded ids and must
    # take the legacy traversal path (get_document_subgraph) — the path real
    # pre-ledger deployments hit.
    from sqlalchemy import delete as sql_delete

    from cognee.modules.graph.models import Node as LedgerNode

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(sql_delete(LedgerNode).where(LedgerNode.data_id == data_id))
        await session.commit()

    await cognee.datasets.delete_data(dataset_id=dataset.id, data_id=data_id, user=user)

    assert await graph.get_node(str(data_id)) is None, "document node must be deleted"
    for node in chunks2:
        assert await graph.get_node(str(node["id"])) is None, (
            "every chunk — legacy and content-id alike — must be deleted by traversal"
        )
    all_ids = [str(n["id"]) for n in chunks2] + deleted_legacy
    assert not await vector.retrieve("DocumentChunk_text", all_ids), (
        "no vector rows may survive the delete"
    )
    nodes, _ = await graph.get_graph_data()
    leftover = [
        props
        for _, props in nodes
        if props.get("type") in ("DocumentChunk", "TextSummary", "TextDocument")
    ]
    assert not leftover, f"stranded nodes after delete: {leftover}"
