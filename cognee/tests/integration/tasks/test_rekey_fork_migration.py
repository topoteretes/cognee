"""The rekey_fork_document_ids chain migration, against a real cognified graph.

Simulates exactly what the relational backfill leaves behind for a fork
dataset: the Data row carries a fresh canonical id (``legacy_id`` recording
the pre-fork original, ledger ``data_id`` repointed) while the GRAPH still
holds the document node under the pre-fork id. Then runs the real chain
migration and asserts:

  1. the graph document node moves to the canonical id, properties intact;
  2. the chunks' ``is_part_of`` edges follow, and their ``document_id``
     display property is updated;
  3. the migration is idempotent (second run changes nothing);
  4. end to end: deletion by the PRE-FORK id through the real API cleans the
     re-keyed graph completely — with the legacy heal never needed.

Runs on the default local stack (kuzu + lancedb + sqlite), mocked LLM and
embeddings — CI-safe, no API keys.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

CHUNK_TOKENS = 60
MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def rekey_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_rekey_fork_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
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


def _para(tag: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {words}.\n"


def test_rekey_fork_migration_end_to_end(rekey_env):
    asyncio.run(_scenario())


async def _scenario():
    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.models import Data
    from cognee.modules.graph.models import Node as LedgerNode
    from cognee.modules.migrations.migration import MigrationContext
    from cognee.modules.migrations.versions.rekey_fork_document_ids import migrate
    from cognee.modules.users.methods import get_default_user
    from sqlalchemy import insert as sql_insert, select, update as sql_update

    # --- Cognify a real document ------------------------------------------- #
    text = "".join(_para(tag) for tag in ["a", "b", "c", "d"])
    await cognee.add(text, dataset_name="fork_mig")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "fork_mig")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    old_row = (await get_dataset_data(dataset.id))[0]
    old_id = old_row.id

    graph = await get_graph_engine()
    assert await graph.get_node(str(old_id)) is not None, "doc node exists under the old id"

    # --- Simulate the backfill fork ---------------------------------------- #
    new_id = uuid4()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = (await session.execute(select(Data).where(Data.id == old_id))).scalar_one()
        values = {column.name: getattr(row, column.name) for column in Data.__table__.columns}
        values["id"] = new_id
        values["legacy_id"] = old_id
        await session.execute(sql_insert(Data.__table__).values(**values))
        await session.execute(
            sql_update(LedgerNode).where(LedgerNode.data_id == old_id).values(data_id=new_id)
        )
        await session.execute(Data.__table__.delete().where(Data.id == old_id))
        await session.commit()

    # --- Run the chain migration ------------------------------------------- #
    context = MigrationContext(
        graph_engine=graph,
        vector_engine=await get_vector_engine_async(),
        dataset_id=None,  # global mode (access control off)
    )
    await migrate(context)

    assert await graph.get_node(str(old_id)) is None, "old doc node re-keyed away"
    new_node = await graph.get_node(str(new_id))
    assert new_node is not None, "doc node exists under the canonical id"
    assert new_node.get("name") == old_row.name, "document properties preserved"

    # Chunks follow: is_part_of edges land on the new node; document_id updated.
    chunk_ids = []
    for source, edge, target in await graph.get_connections(str(new_id)):
        if "is_part_of" in str(edge.get("relationship_name", "")):
            source_id = str(edge.get("source_node_id") or source.get("id"))
            if source_id != str(new_id):
                chunk_ids.append(source_id)
    assert chunk_ids, "chunks are attached to the re-keyed document node"
    for chunk in await graph.get_nodes(chunk_ids):
        assert str(chunk.get("document_id")) == str(new_id), (
            "chunk document_id display property follows the canonical id"
        )

    # --- Idempotency -------------------------------------------------------- #
    await migrate(context)
    assert await graph.get_node(str(new_id)) is not None
    assert await graph.get_node(str(old_id)) is None

    # --- End to end: delete by the PRE-FORK id, heal never needed ----------- #
    import importlib

    legacy_delete_module = importlib.import_module("cognee.modules.graph.methods.legacy_delete")
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    original_subgraph_delete = legacy_delete_module.delete_document_subgraph

    async def _no_heal(document_id, mode="soft"):
        assert str(document_id) == str(new_id), (
            "post-migration deletion must enter the graph by the canonical id only"
        )
        return await original_subgraph_delete(document_id, mode)

    legacy_delete_module.delete_document_subgraph = _no_heal
    try:
        await datasets_api.delete_data(dataset.id, old_id, user=user)
    finally:
        legacy_delete_module.delete_document_subgraph = original_subgraph_delete

    assert await get_dataset_data(dataset.id) == []
    nodes, _ = await graph.get_graph_data()
    leftover = [
        props
        for _, props in nodes
        if props.get("type") in ("DocumentChunk", "TextDocument", "TextSummary")
    ]
    assert not leftover, f"stranded graph nodes after delete: {leftover}"


def test_update_ledger_references_scoped_and_global(rekey_env):
    """The ledger-backend branch: slug / edge-endpoint updates honor dataset scope.

    Graph-provenance stacks never write the ledger (scenario above); backends
    that do keep it need the migration's plain-column updates to move exactly
    the scoped rows — and all rows in global mode — without touching counts.
    """
    asyncio.run(_ledger_scenario())


async def _ledger_scenario():
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.models import Edge as LedgerEdge, Node as LedgerNode
    from cognee.modules.migrations.versions.rekey_fork_document_ids import (
        _update_ledger_references,
    )

    old_id, new_id = uuid4(), uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        for dataset_id in (dataset_a, dataset_b):
            session.add(
                LedgerNode(
                    id=uuid4(),
                    slug=old_id,
                    user_id=uuid4(),
                    data_id=uuid4(),
                    dataset_id=dataset_id,
                    type="TextDocument",
                    indexed_fields=[],
                )
            )
            session.add(
                LedgerEdge(
                    id=uuid4(),
                    slug=uuid4(),
                    user_id=uuid4(),
                    data_id=uuid4(),
                    dataset_id=dataset_id,
                    source_node_id=uuid4(),
                    destination_node_id=old_id,
                    relationship_name="is_part_of",
                )
            )
        await session.commit()

    async def snapshot():
        async with engine.get_async_session() as session:
            nodes = (await session.execute(select(LedgerNode))).scalars().all()
            edges = (await session.execute(select(LedgerEdge))).scalars().all()
        assert len(nodes) == 2 and len(edges) == 2, "updates never add or drop rows"
        return (
            {node.dataset_id: node.slug for node in nodes},
            {edge.dataset_id: edge.destination_node_id for edge in edges},
        )

    # Scoped run: only dataset_a's references move.
    await _update_ledger_references({str(old_id): str(new_id)}, dataset_a)
    slugs, targets = await snapshot()
    assert slugs[dataset_a] == new_id and slugs[dataset_b] == old_id
    assert targets[dataset_a] == new_id and targets[dataset_b] == old_id

    # Global run (access control off): the remaining dataset follows.
    await _update_ledger_references({str(old_id): str(new_id)}, None)
    slugs, targets = await snapshot()
    assert slugs[dataset_b] == new_id and targets[dataset_b] == new_id
