"""Chunk-scoped ownership refs (source_ref:v2), end to end.

The contract (SDK-6 proposal Phase 2-3, adapted to stable-id Data rows):

  1. after cognify, every chunk-produced artifact — the chunk node, its
     entities, its summary — carries the v2 ref of its producing chunk;
     the document node keeps the document-scoped v1 ref;
  2. output produced by SEVERAL chunks carries several v2 refs (one per
     owner) — the merge preserves ownership instead of erasing it;
  3. deleting ONE chunk's ref through the provenance planner removes only
     what that chunk solely owns; shared output survives to its last owner;
  4. document deletion removes v1 AND v2 refs — nothing is stranded;
  5. an incremental update stamps fresh chunks with their own v2 refs.

Runs on the default local stack (kuzu + lancedb + sqlite), mocked LLM and
embeddings — CI-safe, no API keys.
"""

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def ownership_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_chunk_ownership_test_"))

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
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str, shared_marker: str = "") -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {shared_marker} {words}.\n"


async def _refs_by_node(graph, dataset_id):
    return await graph.find_node_source_refs_by_dataset(str(dataset_id))


def test_chunk_ownership_refs_end_to_end(ownership_env):
    asyncio.run(_scenario())


async def _scenario():
    await reset_backend_state()
    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance import parse_source_ref_key
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    # ENTSHARED appears in BOTH paragraphs -> the shared entity has two owners.
    text = _para("a", "ENTSHARED") + _para("b", "ENTSHARED")
    await cognee.add(text, dataset_name="ownership")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "ownership")
    await cognee.cognify(datasets=[dataset.id], chunk_size=60)
    data_id = (await get_dataset_data(dataset.id))[0].id

    graph = await get_graph_engine()
    refs_by_node = await _refs_by_node(graph, dataset.id)
    nodes_by_id = {str(node_id): props for node_id, props in (await graph.get_graph_data())[0]}

    # ── 1+2. every artifact carries the right-scoped refs ─────────────────── #
    chunk_ids, shared_entity_owner_counts = [], []
    for node_id, refs in refs_by_node.items():
        parsed = [parse_source_ref_key(ref) for ref in refs]
        node_type = nodes_by_id.get(node_id, {}).get("type")
        if node_type == "TextDocument":
            assert all(p.version == 1 for p in parsed), "document node keeps the v1 ref"
        elif node_type == "DocumentChunk":
            assert any(p.version == 2 and str(p.chunk_id) == node_id for p in parsed), (
                "a chunk owns itself via its v2 ref"
            )
            chunk_ids.append(node_id)
        elif node_type in ("Marker", "TextSummary", "EntityType"):
            assert any(p.version == 2 for p in parsed), (
                f"{node_type} {node_id} must carry a chunk-scoped v2 ref"
            )
        if str(nodes_by_id.get(node_id, {}).get("name", "")).lower() == "entshared":
            shared_entity_owner_counts.append(
                len({str(p.chunk_id) for p in parsed if p.version == 2})
            )
    assert len(chunk_ids) >= 2, "the document must produce at least two chunks"
    assert shared_entity_owner_counts and shared_entity_owner_counts[0] >= 2, (
        "an entity produced by two chunks must carry BOTH chunk refs"
    )

    # ── 3. deleting one chunk's ref keeps shared output ──────────────────── #
    from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
    from cognee.infrastructure.databases.unified import get_unified_engine
    from uuid import UUID

    unified = await get_unified_engine()
    victim_chunk = chunk_ids[0]
    await unified.delete_by_source_ref(
        make_chunk_source_ref_key(dataset.id, data_id, UUID(victim_chunk))
    )
    assert await graph.get_node(victim_chunk) is None, "the chunk itself is gone"
    shared_alive = [
        node_id
        for node_id, props in (await graph.get_graph_data())[0]
        if str(props.get("name", "")).lower() == "entshared"
    ]
    assert shared_alive, "output shared with the surviving chunk must survive"

    # ── 5. incremental update stamps fresh chunks with v2 refs ───────────── #
    # (the deleted chunk breaks tiling; the update self-heals via full
    # rebuild, which itself must produce chunk-scoped refs again)
    text_v2 = text + _para("c")
    await cognee.update(data_id, text_v2, dataset.id, user=user)
    refs_by_node = await _refs_by_node(graph, dataset.id)
    fresh_v2 = [
        ref
        for refs in refs_by_node.values()
        for ref in refs
        if parse_source_ref_key(ref).version == 2
    ]
    assert fresh_v2, "post-update artifacts carry v2 refs"

    # ── 4. document deletion strands NOTHING ─────────────────────────────── #
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    await datasets_api.delete_data(dataset.id, data_id, user=user)
    leftover = [
        props
        for _, props in (await graph.get_graph_data())[0]
        if props.get("type") in ("DocumentChunk", "TextDocument", "TextSummary", "Marker")
    ]
    assert not leftover, f"document deletion must remove v1 AND v2 owned output: {leftover}"
