"""Chunk-budget persistence: updates keep a document's ingestion granularity.

A document is cognified at one token budget; the global configuration later
changes. An incremental update must re-chunk its edited region with the budget
RECORDED on the chunks it replaces — not the new global value — so the document
never becomes a patchwork of chunk sizes.

Runs on the default local stack (kuzu + lancedb + sqlite), mocked LLM and
embeddings — CI-safe, no API keys.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

INGEST_BUDGET = 60
CHANGED_GLOBAL_BUDGET = 400
MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def budget_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_budget_test_"))

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

    yield root

    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {words}.\n"


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


def test_update_respects_recorded_chunk_budget(budget_env):
    asyncio.run(_scenario())


async def _scenario():
    await reset_backend_state()
    import cognee
    import cognee.api.v1.update.incremental as incremental_module
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    # --- ingest at the ORIGINAL budget ------------------------------------- #
    paragraphs = [_para(tag) for tag in ["a", "b", "c", "d", "e", "f"]]
    text_v1 = "".join(paragraphs)
    await cognee.add(text_v1, dataset_name="budget")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "budget")
    await cognee.cognify(datasets=[dataset.id], chunk_size=INGEST_BUDGET)
    data_id = (await get_dataset_data(dataset.id))[0].id

    chunks = await _get_chunks(data_id)
    assert len(chunks) >= 4
    assert all(int(n.get("max_chunk_tokens") or 0) == INGEST_BUDGET for n in chunks), (
        "ingestion must stamp every chunk with the budget it was cut against"
    )

    # --- the global configuration changes ---------------------------------- #
    async def _changed_global():
        return CHANGED_GLOBAL_BUDGET

    original_budget = incremental_module.get_max_chunk_tokens
    incremental_module.get_max_chunk_tokens = _changed_global
    try:
        # Replace two adjacent paragraphs with three new ones (~90 tokens):
        # under the recorded budget (60) that region must become MULTIPLE
        # chunks; under the new global (400) it would collapse into one.
        start = sum(len(p) for p in paragraphs[:2])
        end = start + len(paragraphs[2]) + len(paragraphs[3])
        replacement = _para("x") + _para("y") + _para("z")
        text_v2 = text_v1[:start] + replacement + text_v1[end:]

        result = await cognee.update(data_id, text_v2, dataset.id, user=user)
        assert isinstance(result, dict) and result.get("status") == "incremental", result
    finally:
        incremental_module.get_max_chunk_tokens = original_budget

    data_id = (await get_dataset_data(dataset.id))[0].id
    chunks2 = await _get_chunks(data_id)
    assert "".join(n["text"] for n in chunks2) == text_v2

    old_ids = {str(n["id"]) for n in chunks}
    fresh = [n for n in chunks2 if str(n["id"]) not in old_ids]
    assert len(fresh) >= 2, (
        "the region must be cut at the recorded budget (60), not the new global (400) "
        f"— got {len(fresh)} fresh chunk(s)"
    )
    assert all(int(n.get("max_chunk_tokens") or 0) == INGEST_BUDGET for n in fresh), (
        "fresh region chunks must be stamped with the budget they were cut against"
    )
    assert all(int(n.get("chunk_size") or 0) <= INGEST_BUDGET for n in fresh)

    # Kept chunks retain their stamp through renumbering/rehydration.
    kept = [n for n in chunks2 if str(n["id"]) in old_ids]
    assert kept and all(int(n.get("max_chunk_tokens") or 0) == INGEST_BUDGET for n in kept), (
        "kept chunks must not lose their recorded budget"
    )
