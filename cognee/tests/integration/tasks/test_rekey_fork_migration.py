"""The rekey_fork_document_ids chain migration, against a real cognified graph.

Simulates exactly what the relational backfill leaves behind for a fork
dataset: the Data row carries a fresh canonical id (``legacy_id`` recording
the pre-fork original, ledger ``data_id`` repointed) while the GRAPH still
holds the document node under the pre-fork id. Then runs the real chain
migration and asserts:

  1. the graph document node moves to the canonical id, properties intact;
  2. the chunks' ``is_part_of`` edges follow, and their ``document_id`` is
     updated in the graph AND in the vector index payload (search Evidence);
  3. the migration is idempotent (second run changes nothing);
  4. end to end: deletion by the PRE-FORK id through the real API cleans the
     re-keyed graph completely — entering it by the canonical id only;
  5. the ledger-backend branch honors dataset scoping (and global mode);
  6. downgrade fully reverses the store state (graph node, chunk properties,
     vector payloads, provenance refs), is idempotent, and the round trip
     re-migrate -> delete-by-pre-fork-id still converges;
  7. shared-store keeper: when the pre-fork id is still a live row (access
     control off — one graph for all datasets), the fork pair is
     provenance-only, the keeper's node/chunks/payloads untouched, and
     refcounted deletion then works dataset by dataset.

All scenarios share one event loop (cached engines bind asyncio locks to the
first loop). Runs on the default local stack (kuzu + lancedb + sqlite),
mocked LLM and embeddings — CI-safe, no API keys.
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


async def _cognify_document(dataset_name: str):
    """Add + cognify one marker document; return (user, dataset, data_row)."""
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    text = "".join(_para(tag) for tag in ["a", "b", "c", "d"])
    await cognee.add(text, dataset_name=dataset_name)
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == dataset_name)
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    return user, dataset, (await get_dataset_data(dataset.id))[0]


async def _flatten_to_document_scoped_refs(graph, dataset_id, data_id):
    """Rewrite a freshly built graph into the PRE-REFACTOR ref shape.

    Old cognee stamped the whole document subgraph with the document-scoped
    v1 key; current cognify stamps chunk-produced artifacts with chunk-scoped
    v2 refs instead. Scenarios that simulate data written by old versions
    (the shared-store keeper) must flatten those v2 refs back to the v1 key,
    or the simulation tests a graph shape no old version ever produced.
    """
    from cognee.infrastructure.databases.provenance import (
        make_source_ref_key,
        parse_source_ref_key,
    )

    doc_key = make_source_ref_key(dataset_id, data_id)

    def _v2_refs(refs):
        selected = []
        for ref in refs:
            parsed = parse_source_ref_key(ref)
            if parsed.version == 2 and str(parsed.data_id) == str(data_id):
                selected.append(ref)
        return selected

    for node_id, refs in (await graph.find_node_source_refs_by_dataset(str(dataset_id))).items():
        v2_refs = _v2_refs(refs)
        if v2_refs:
            await graph.attach_node_source_refs([node_id], [doc_key], None)
            await graph.remove_node_source_refs([node_id], v2_refs)
    for edge, refs in (await graph.find_edge_source_refs_by_dataset(str(dataset_id))).items():
        v2_refs = _v2_refs(refs)
        if v2_refs:
            await graph.attach_edge_source_refs([edge], [doc_key], None)
            await graph.remove_edge_source_refs([edge], v2_refs)


async def _simulate_backfill_fork(old_id, fork_dataset_id=None, keep_keeper=False):
    """Copy the Data row under a fresh id with legacy_id, as the backfill would.

    Default shape: the migrated dataset's row was the split one — ledger
    repointed, old row gone. ``keep_keeper=True`` models the shared-store
    shape instead: the keeper dataset retains the original row and the fork
    row lands in ``fork_dataset_id``.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.modules.graph.models import Node as LedgerNode
    from sqlalchemy import insert as sql_insert, select, update as sql_update

    new_id = uuid4()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = (await session.execute(select(Data).where(Data.id == old_id))).scalar_one()
        values = {column.name: getattr(row, column.name) for column in Data.__table__.columns}
        values["id"] = new_id
        values["legacy_id"] = old_id
        if fork_dataset_id is not None:
            values["dataset_id"] = fork_dataset_id
        await session.execute(sql_insert(Data.__table__).values(**values))
        if not keep_keeper:
            await session.execute(
                sql_update(LedgerNode).where(LedgerNode.data_id == old_id).values(data_id=new_id)
            )
            await session.execute(Data.__table__.delete().where(Data.id == old_id))
        await session.commit()
    return new_id


async def _doc_chunk_ids(graph, document_id) -> list:
    """Ids of the chunks attached to a document node via is_part_of."""
    chunk_ids = []
    for source, edge, _target in await graph.get_connections(str(document_id)):
        if "is_part_of" in str(edge.get("relationship_name", "")):
            source_id = str(edge.get("source_node_id") or source.get("id"))
            if source_id != str(document_id):
                chunk_ids.append(source_id)
    return chunk_ids


async def _vector_payload_document_ids(vector_engine, chunk_ids: list) -> set:
    """The distinct document_id payload scalars on the chunks' vector rows.

    Backend-aware like the migrations' own vector helpers: LanceDB rows are
    read through the table query API, PGVector through SQLAlchemy.
    """
    if vector_engine.__class__.__name__ == "PGVectorAdapter":
        from sqlalchemy import select as sql_select

        table = await vector_engine.get_table("DocumentChunk_text")
        async with vector_engine.get_async_session() as session:
            payloads = [
                row.payload or {}
                for row in (
                    await session.execute(
                        sql_select(table.c.payload).where(table.c.id.in_(chunk_ids))
                    )
                ).all()
            ]
    else:
        table = await vector_engine.get_collection("DocumentChunk_text")
        escaped = [chunk_id.replace("'", "''") for chunk_id in chunk_ids]
        where = "id IN ({})".format(", ".join(f"'{chunk_id}'" for chunk_id in escaped))
        payloads = [row.get("payload") or {} for row in await table.query().where(where).to_list()]
    assert len(payloads) == len(chunk_ids), "every chunk has exactly one vector row"
    return {str(payload.get("document_id")) for payload in payloads}


def test_rekey_fork_migration_end_to_end(rekey_env):
    # One event loop for every scenario: cached engines hold asyncio locks
    # created on first use, and a second asyncio.run would strand them on a
    # closed loop (the same reason the scoped suite runs one loop per module).
    async def _all_scenarios():
        await _scenario()
        await _ledger_scenario()
        await _downgrade_scenario()
        await _keeper_scenario()

    asyncio.run(_all_scenarios())


async def _scenario():
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.migrations.migration import MigrationContext
    from cognee.modules.migrations.versions.rekey_fork_document_ids import migrate

    # --- Cognify a real document, then simulate the backfill fork ----------- #
    user, dataset, old_row = await _cognify_document("fork_mig")
    old_id = old_row.id

    graph = await get_graph_engine()
    assert await graph.get_node(str(old_id)) is not None, "doc node exists under the old id"

    new_id = await _simulate_backfill_fork(old_id)

    # --- Run the chain migration ------------------------------------------- #
    vector_engine = await get_vector_engine_async()
    context = MigrationContext(
        graph_engine=graph,
        vector_engine=vector_engine,
        dataset_id=None,  # global mode (access control off)
    )
    await migrate(context)

    assert await graph.get_node(str(old_id)) is None, "old doc node re-keyed away"
    new_node = await graph.get_node(str(new_id))
    assert new_node is not None, "doc node exists under the canonical id"
    assert new_node.get("name") == old_row.name, "document properties preserved"

    # Chunks follow: is_part_of edges land on the new node; document_id updated
    # in the graph AND in the vector index payload (search Evidence).
    chunk_ids = await _doc_chunk_ids(graph, new_id)
    assert chunk_ids, "chunks are attached to the re-keyed document node"
    for chunk in await graph.get_nodes(chunk_ids):
        assert str(chunk.get("document_id")) == str(new_id), (
            "chunk document_id display property follows the canonical id"
        )
    assert await _vector_payload_document_ids(vector_engine, chunk_ids) == {str(new_id)}, (
        "chunk vector payloads cite the canonical document id"
    )

    # --- Idempotency -------------------------------------------------------- #
    await migrate(context)
    assert await graph.get_node(str(new_id)) is not None
    assert await graph.get_node(str(old_id)) is None
    assert await _vector_payload_document_ids(vector_engine, chunk_ids) == {str(new_id)}

    # --- End to end: delete by the PRE-FORK id ------------------------------- #
    import importlib

    legacy_delete_module = importlib.import_module("cognee.modules.graph.methods.legacy_delete")
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    original_subgraph_delete = legacy_delete_module.delete_document_subgraph

    async def _canonical_only(document_id, mode="soft"):
        assert str(document_id) == str(new_id), (
            "post-migration deletion must enter the graph by the canonical id only"
        )
        return await original_subgraph_delete(document_id, mode)

    legacy_delete_module.delete_document_subgraph = _canonical_only
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


async def _ledger_scenario():
    """The ledger-backend branch: slug / edge-endpoint updates honor dataset scope.

    Graph-provenance stacks never write the ledger (scenario above); backends
    that do keep it need the migration's plain-column updates to move exactly
    the scoped rows — and all rows in global mode — without touching counts.
    """
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


async def _downgrade_scenario():
    """Downgrade fully reverses the migration, and the round trip converges.

    migrate -> downgrade must restore the exact pre-migration store state:
    document node back under the pre-fork id, chunk ``document_id`` reverted
    in graph and vector payloads, provenance source refs back on the legacy
    key. Downgrade is idempotent, a re-migrate lands canonical again, and the
    re-migrated state still deletes cleanly by the pre-fork id.
    """
    from cognee.api.v1.datasets.datasets import datasets as datasets_api
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance import make_source_ref_key
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.migrations.migration import MigrationContext
    from cognee.modules.migrations.versions.rekey_fork_document_ids import downgrade, migrate

    user, dataset, old_row = await _cognify_document("fork_down")
    old_id = old_row.id
    graph = await get_graph_engine()
    vector_engine = await get_vector_engine_async()

    chunk_ids = await _doc_chunk_ids(graph, old_id)
    assert chunk_ids, "cognify attached chunks to the document"
    assert await _vector_payload_document_ids(vector_engine, chunk_ids) == {str(old_id)}

    new_id = await _simulate_backfill_fork(old_id)
    old_key = make_source_ref_key(dataset.id, old_id)
    new_key = make_source_ref_key(dataset.id, new_id)

    context = MigrationContext(graph_engine=graph, vector_engine=vector_engine, dataset_id=None)

    async def assert_state(document_id, other_id, document_key, other_key, label):
        assert await graph.get_node(str(document_id)) is not None, f"{label}: doc node present"
        assert await graph.get_node(str(other_id)) is None, f"{label}: other id absent"
        assert set(await _doc_chunk_ids(graph, document_id)) == set(chunk_ids), (
            f"{label}: chunk edges attached"
        )
        for chunk in await graph.get_nodes(chunk_ids):
            assert str(chunk.get("document_id")) == str(document_id), f"{label}: graph document_id"
        assert await _vector_payload_document_ids(vector_engine, chunk_ids) == {str(document_id)}, (
            f"{label}: vector payload document_id"
        )
        assert await graph.find_nodes_by_source_ref(document_key), f"{label}: provenance refs"
        assert not await graph.find_nodes_by_source_ref(other_key), (
            f"{label}: no stale provenance refs"
        )

    # --- migrate: canonical everywhere -------------------------------------- #
    await migrate(context)
    await assert_state(new_id, old_id, new_key, old_key, "after migrate")

    # --- downgrade: exact pre-migration store state -------------------------- #
    await downgrade(context)
    await assert_state(old_id, new_id, old_key, new_key, "after downgrade")

    # --- downgrade idempotency ----------------------------------------------- #
    await downgrade(context)
    await assert_state(old_id, new_id, old_key, new_key, "after second downgrade")

    # --- round trip: re-migrate lands canonical again ------------------------ #
    await migrate(context)
    await assert_state(new_id, old_id, new_key, old_key, "after re-migrate")

    # --- and the re-migrated state still deletes cleanly by the pre-fork id -- #
    await datasets_api.delete_data(dataset.id, old_id, user=user)
    assert await get_dataset_data(dataset.id) == []
    nodes, _ = await graph.get_graph_data()
    leftover = [
        props
        for _, props in nodes
        if props.get("type") in ("DocumentChunk", "TextDocument", "TextSummary")
    ]
    assert not leftover, f"stranded graph nodes after roundtrip delete: {leftover}"


async def _keeper_scenario():
    """Shared-store keeper: the fork pair is provenance-only, the keeper untouched.

    With access control off every dataset shares one graph. Pre-refactor, a
    document shared by datasets A and B was ONE graph node carrying both
    datasets' provenance refs; the backfill keeps the original row for A (the
    keeper) and gives B a fork row. The migration must not steal the node,
    chunk properties, or vector payloads from the keeper — it may only move
    B's own provenance keys. Then refcounted deletion works dataset by
    dataset: deleting A leaves the shared subgraph alive for B; deleting B by
    its PRE-FORK id removes it completely.
    """
    from cognee.api.v1.datasets.datasets import datasets as datasets_api
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance import make_source_ref_key
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import load_or_create_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.migrations.migration import MigrationContext
    from cognee.modules.migrations.versions.rekey_fork_document_ids import migrate

    user, dataset_a, row_a = await _cognify_document("keep_a")
    old_id = row_a.id
    graph = await get_graph_engine()
    vector_engine = await get_vector_engine_async()
    chunk_ids = await _doc_chunk_ids(graph, old_id)
    assert chunk_ids
    # Pre-refactor data carried ONLY the document-scoped key on the whole
    # subgraph — flatten the chunk-scoped refs current cognify writes.
    await _flatten_to_document_scoped_refs(graph, dataset_a.id, old_id)

    # Dataset B: pre-refactor shared membership = B's provenance refs on the
    # SAME subgraph, then the backfill fork row (keeper row kept).
    dataset_b = (await load_or_create_datasets(["keep_b"], [], user))[0]
    key_a = make_source_ref_key(dataset_a.id, old_id)
    key_b_old = make_source_ref_key(dataset_b.id, old_id)

    shared_node_ids = await graph.find_nodes_by_source_ref(key_a)
    shared_edges = await graph.find_edges_by_source_ref(key_a)
    assert shared_node_ids and shared_edges
    await graph.attach_node_source_refs(shared_node_ids, [key_b_old])
    await graph.attach_edge_source_refs(shared_edges, [key_b_old])

    new_id = await _simulate_backfill_fork(old_id, fork_dataset_id=dataset_b.id, keep_keeper=True)
    key_b_new = make_source_ref_key(dataset_b.id, new_id)

    context = MigrationContext(graph_engine=graph, vector_engine=vector_engine, dataset_id=None)
    await migrate(context)

    async def assert_keeper_intact(label):
        assert await graph.get_node(str(old_id)) is not None, f"{label}: keeper node stays"
        assert await graph.get_node(str(new_id)) is None, f"{label}: no node under the fork id"
        for chunk in await graph.get_nodes(chunk_ids):
            assert str(chunk.get("document_id")) == str(old_id), (
                f"{label}: chunk document_id stays with the keeper"
            )
        assert await _vector_payload_document_ids(vector_engine, chunk_ids) == {str(old_id)}, (
            f"{label}: vector payloads stay with the keeper"
        )
        assert await graph.find_nodes_by_source_ref(key_a), f"{label}: keeper provenance intact"
        assert not await graph.find_nodes_by_source_ref(key_b_old), (
            f"{label}: fork legacy provenance moved"
        )
        assert await graph.find_nodes_by_source_ref(key_b_new), (
            f"{label}: fork canonical provenance present"
        )

    await assert_keeper_intact("after migrate")
    await migrate(context)
    await assert_keeper_intact("after re-run")

    # Refcounted deletion: A first (shared subgraph survives for B) ...
    await datasets_api.delete_data(dataset_a.id, old_id, user=user)
    assert await get_dataset_data(dataset_a.id) == []
    assert await graph.get_node(str(old_id)) is not None, (
        "shared subgraph survives while the fork dataset still references it"
    )

    # ... then B by its PRE-FORK id: everything goes.
    await datasets_api.delete_data(dataset_b.id, old_id, user=user)
    assert await get_dataset_data(dataset_b.id) == []
    nodes, _ = await graph.get_graph_data()
    leftover = [
        props
        for _, props in nodes
        if props.get("type") in ("DocumentChunk", "TextDocument", "TextSummary")
    ]
    assert not leftover, f"stranded graph nodes after keeper+fork delete: {leftover}"
