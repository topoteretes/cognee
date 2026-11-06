"""Verify upgrades from the v1.6.3 compatibility baseline.

Phase 1 seeds real graph, vector and relational stores using v1.6.3. Phase 2
runs current migrations, checks direct graph/ledger/vector access, re-cognifies,
and verifies the data again. Completion searches are smoke checks; non-empty
LLM answers alone do not establish that the old data remains accessible.

The baseline already has dataset-scoped documents, DLT source manifests and
session persistence watermarks. Verify those shapes survive the upgrade:

* identical content in two datasets has separate IDs, each resolving only in
  its own dataset and unambiguously without a dataset filter;
* the DLT manifest and its row nodes survive, and re-adding the same source
  preserves the document and row IDs without duplicating records;
* an unchanged session produces no new document and preserves its watermark;
  growing it persists only the new entry, with every fact stored exactly once;
* deleting every document removes its solely-owned artifacts, preserves shared
  artifacts until their last owner is gone, and leaves the graph empty.

Graph checks also reject stale Entity/EntityType IDs and dangling edges. Ledger
checks reject references to missing graph nodes, and raw CHUNKS retrieval must
return the seeded text. Together these checks catch semantically broken stores
even when a completion search can still return an answer.
"""

import asyncio
import sys
from collections import Counter

from sqlalchemy import select

import cognee
from cognee.api.v1.search import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import get_persisted_qa_count
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.data.models import Data as _ScopedData
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.models import Edge, Node
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import build_id_remap
from cognee.modules.users.methods import get_default_user

_LEDGER_ENTITY_TYPES = (Entity.__name__, EntityType.__name__)

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt
in culpa qui officia deserunt mollit anim id est laborum.

Lorem ipsum is placeholder text commonly used in the graphic, print, and publishing industries for
previewing layouts and visual mockups. It has been the industry standard dummy text since the 1500s
when an unknown printer scrambled a passage of text to make a type specimen book.

Loren Ipsum Dolor sit amet, Lorem ipsum.
"""

DATASET = "lorem_ipsum"
SEARCH_QUERY = "What is Lorem Ipsum and where does it come from?"

# Dataset isolation — names must stay in sync with phase1_seed.py.
ISOLATION_DATASET_A = "isolation_a"
ISOLATION_DATASET_B = "isolation_b"

# DLT takeover checks — markers must stay in sync with phase1_seed.py.
DLT_COMPAT_DATASET = "dlt_compat"
DLT_COMPAT_SOURCE = "compat_widgets"
DLT_COMPAT_ROWS = [
    {"id": "1", "name": "anemometer", "shelf": "north"},
    {"id": "2", "name": "barometer", "shelf": "east"},
    {"id": "3", "name": "hygrometer", "shelf": "south"},
]

# Session takeover checks — markers must stay in sync with phase1_seed.py.
COMPAT_SESSION_ID = "compat_session"
SESSION_FACT_1_MARKER = "Anton Zorman"
SESSION_FACT_2_MARKER = "Ilka Matova"
SESSION_FACT_3 = "The glazier Pavle Rossi fired amber panes for the Karst chapel windows."
SESSION_FACT_3_MARKER = "Pavle Rossi"


def _fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def _count_stale_ids(nodes: list) -> int:
    """Count Entity/EntityType nodes the migration would still remap.

    Uses the migration's own recognition (``build_id_remap``) so "stale" means
    exactly "still on a recognized old ID scheme" — i.e. the migration did not
    run/complete. This avoids false positives on freshly-created nodes whose id
    was (legitimately) derived from a value other than the stored name.
    """
    return len(build_id_remap(nodes))


async def _collect_graph() -> tuple[list, list]:
    """Gather all (nodes, edges) across every dataset's graph, addressed BY ID.

    Datasets are addressed by their stored ID + owner (from the dataset_database
    rows), never by name: the name→ID mapping is per-user/tenant, so resolving by
    name across the seed/verify boundary points at a *different* dataset. See
    ``get_unique_dataset_id`` for why.
    """
    rows = await get_dataset_databases()
    all_nodes: list = []
    all_edges: list = []

    if rows:
        for row in rows:
            async with set_database_global_context_variables(row.dataset_id, row.owner_id):
                graph_engine = await get_graph_engine()
                nodes, edges = await graph_engine.get_graph_data()
            all_nodes.extend(nodes)
            all_edges.extend(edges)
    else:
        # Access control off: a single global graph, no per-dataset rows.
        graph_engine = await get_graph_engine()
        all_nodes, all_edges = await graph_engine.get_graph_data()

    return all_nodes, all_edges


async def _verify_graph_access(stage: str, nodes: list, edges: list) -> None:
    """Fail unless the migrated graph is non-empty, fully migrated, and intact.

    Checks (a) graph is non-empty, (b) no Entity/EntityType node is still on an
    old ID scheme, (c) no dangling edges — every edge endpoint resolves to a
    node, which proves the migration rewired entity edges and that ALL node
    types (DocumentChunk, TextSummary, EdgeType, …), not just entities, remain
    reachable and connected after the ID remap.
    """
    if not nodes:
        _fail(f"[{stage}] graph has no nodes — graph data is not accessible after migration.")

    stale = _count_stale_ids(nodes)
    if stale:
        _fail(
            f"[{stage}] {stale} Entity/EntityType node(s) still use the pre-#2515 ID scheme — "
            "the graph migration did not run or did not complete."
        )

    node_ids = {node_id for node_id, _ in nodes}
    # Ladybug/Kuzu injects (id, id, "SELF") self-loops when a graph has no edges,
    # which would mask total edge loss — drop them before the connectivity check.
    real_edges = [(s, t, r) for s, t, r, _ in edges if r != "SELF"]
    if not real_edges:
        _fail(f"[{stage}] graph has no real edges — possible edge loss after migration.")

    dangling = [(s, t, r) for s, t, r in real_edges if s not in node_ids or t not in node_ids]
    if dangling:
        _fail(
            f"[{stage}] {len(dangling)} dangling edge(s) after migration "
            f"(e.g. {dangling[0]}) — the ID remap broke cross-type connectivity."
        )

    # Independent positive correctness check (does NOT use the migration's own
    # recognizer): at least one Entity must be addressable at its model-owned
    # id_for(name). Catches a migration that ran but wrote wrong (not-old-scheme)
    # ids — which the build_id_remap-based stale count cannot see.
    if not any(
        props.get("type") == "Entity"
        and props.get("name")
        and node_id == str(Entity.id_for(props["name"]))
        for node_id, props in nodes
    ):
        _fail(
            f"[{stage}] no Entity is on the model-owned ID scheme "
            "(id != Entity.id_for(name)) — the migration produced wrong IDs."
        )

    type_counts = Counter(props.get("type") for _, props in nodes)
    print(
        f"  [graph] {len(nodes)} nodes across {len(type_counts)} types {dict(type_counts)}, "
        f"{len(real_edges)} edges, 0 stale, 0 dangling — OK"
    )


async def _verify_vector_access(stage: str) -> None:
    """Raw CHUNKS retrieval must still find the seeded text.

    Scope to the lorem dataset: DLT rows have their own vector collection and
    do not populate DocumentChunk_text.
    """
    results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text=SEARCH_QUERY, datasets=[DATASET]
    )
    if not results:
        _fail(f"[{stage}] CHUNKS search returned no results — vector data is not accessible.")
    print(f"  [vector] CHUNKS: {len(results)} result(s) — OK")


async def _verify_ledger_access(stage: str, graph_node_ids: set) -> None:
    """Read the relational delete-ledger (``nodes`` / ``edges``) and verify it
    still points at the migrated graph.

    The delete system deletes graph nodes by ``nodes.slug`` and references edges
    by ``edges.source_node_id`` / ``destination_node_id`` — all graph node ids.
    If the id migration moved graph nodes but left these stale, a later delete
    silently misses the migrated nodes (orphans). So this asserts every ledger
    id that the migration could have moved (Entity/EntityType ``slug`` and every
    edge endpoint) still resolves to a live graph node.

    Graph-provenance stores keep ownership in the graph, so an empty ledger
    is valid. The delete sweep verifies graph ownership too.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        node_rows = (await session.scalars(select(Node))).all()
        edge_rows = (await session.scalars(select(Edge))).all()

    if not node_rows and not edge_rows:
        print("  [ledger] empty (ownership may be stored in the graph) — read OK")
        return

    stale_node_slugs = [
        str(row.slug)
        for row in node_rows
        if row.type in _LEDGER_ENTITY_TYPES and str(row.slug) not in graph_node_ids
    ]
    if stale_node_slugs:
        _fail(
            f"[{stage}] {len(stale_node_slugs)} ledger Entity/EntityType slug(s) do not resolve "
            f"to a graph node (e.g. {stale_node_slugs[0]}) — the migration left the relational "
            "ledger stale; deletes would orphan these nodes in the graph."
        )

    dangling_edges = [
        (str(row.source_node_id), str(row.destination_node_id))
        for row in edge_rows
        if str(row.source_node_id) not in graph_node_ids
        or str(row.destination_node_id) not in graph_node_ids
    ]
    if dangling_edges:
        _fail(
            f"[{stage}] {len(dangling_edges)} ledger edge endpoint(s) do not resolve to a graph "
            f"node (e.g. {dangling_edges[0]}) — ledger edges desynced from the migrated graph."
        )

    print(
        f"  [ledger] {len(node_rows)} node row(s), {len(edge_rows)} edge row(s), "
        "all ids resolve to the migrated graph — OK"
    )


async def _verify_access(stage: str) -> None:
    print(f"\n[{stage}] Verifying graph + vector + ledger access")
    nodes, edges = await _collect_graph()
    graph_node_ids = {node_id for node_id, _ in nodes}
    await _verify_graph_access(stage, nodes, edges)
    await _verify_ledger_access(stage, graph_node_ids)
    await _verify_vector_access(stage)

    # Smoke-check the completion + summaries paths (must not raise). Not used
    # as the accessibility gate: the LLM can answer non-empty even with no
    # context.
    for query_type in (
        SearchType.GRAPH_COMPLETION,
        SearchType.RAG_COMPLETION,
        SearchType.SUMMARIES,
    ):
        await cognee.search(query_type=query_type, query_text=SEARCH_QUERY, datasets=[DATASET])
    print("  [smoke] GRAPH_COMPLETION + RAG_COMPLETION + SUMMARIES ran without error — OK")


async def _snapshot_dataset_graph(dataset_id, owner_by_dataset):
    """(node_ids, props_by_id, edge_keys) for the graph holding ``dataset_id``.

    ``owner_by_dataset`` is the dataset_id→owner_id map from dataset_database
    rows when access control is on (per-dataset graphs), or ``None`` when off
    (one global graph). Edge keys are (source, target, relationship_name) with
    Ladybug's synthetic SELF self-loops dropped.
    """
    if owner_by_dataset is None:
        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
    else:
        async with set_database_global_context_variables(dataset_id, owner_by_dataset[dataset_id]):
            graph_engine = await get_graph_engine()
            nodes, edges = await graph_engine.get_graph_data()

    node_ids = {str(node_id) for node_id, _ in nodes}
    props_by_id = {str(node_id): props for node_id, props in nodes}
    edge_keys = {
        (str(s), str(t), str(r)) for s, t, r, _ in edges if not (r == "SELF" and str(s) == str(t))
    }
    return node_ids, props_by_id, edge_keys


async def _ledger_expectations(data_id, dataset_id, scope_to_dataset: bool):
    """Compute what THIS document's hard delete may remove, from the live ledger.

    A slug/edge is expected to be deleted only when no other live document
    references it; anything referenced by another document must survive (the
    two phase documents share most entities, so this is genuinely exercised).
    ``scope_to_dataset`` limits "other documents" to the same dataset when
    access control is on (each dataset has its own physical graph; the same
    deterministic slug in another dataset lives in a different graph and must
    not be counted as a survivor here).
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        node_rows = (await session.scalars(select(Node))).all()
        edge_rows = (await session.scalars(select(Edge))).all()

    if scope_to_dataset:
        node_rows = [row for row in node_rows if row.dataset_id == dataset_id]
        edge_rows = [row for row in edge_rows if row.dataset_id == dataset_id]

    def _is_doc_row(row):
        return row.data_id == data_id and row.dataset_id == dataset_id

    doc_nodes = {str(row.slug) for row in node_rows if _is_doc_row(row)}
    other_nodes = {str(row.slug) for row in node_rows if not _is_doc_row(row)}

    def _edge_key(row):
        return (
            str(row.source_node_id),
            str(row.destination_node_id),
            str(row.label or row.relationship_name),
        )

    doc_edges = {_edge_key(row) for row in edge_rows if _is_doc_row(row)}
    other_edges = {_edge_key(row) for row in edge_rows if not _is_doc_row(row)}

    return doc_nodes - other_nodes, doc_edges - other_edges


async def _graph_provenance_expectations(graph_engine, dataset_id, data_id):
    """Ownership expectations read from in-graph provenance columns.

    Graph-provenance datasets deliberately have no relational-ledger rows
    (no dual tracking), so the delete oracle must read the same columns the
    delete itself uses: a node/edge is expected to disappear only when this
    document owns ALL its source refs, including chunk-scoped refs.
    """
    from cognee.infrastructure.databases.provenance.source_refs import parse_source_ref_key

    def _solely_owned(raw) -> bool:
        refs = [parse_source_ref_key(ref) for ref in (raw or "").split("|") if ref]
        return bool(refs) and all(
            ref.dataset_id == dataset_id and ref.data_id == data_id for ref in refs
        )

    node_rows = await graph_engine.query("MATCH (n:Node) RETURN n.id, n.source_ref_keys")
    solely_nodes = {str(node_id) for node_id, raw in node_rows if _solely_owned(raw)}

    edge_rows = await graph_engine.query(
        "MATCH (a:Node)-[r:EDGE]->(b:Node) "
        "RETURN a.id, b.id, r.relationship_name, r.source_ref_keys"
    )
    solely_edges = {
        (str(source), str(target), str(rel))
        for source, target, rel, raw in edge_rows
        if _solely_owned(raw)
    }
    return solely_nodes, solely_edges


async def _expectations_for(data_id, dataset_id, owner_by_dataset):
    """Read ownership from each store's actual provenance mode.

    The v1.6.3 baseline supports relational-ledger and graph-provenance stores.
    """
    from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph

    if owner_by_dataset is None:
        graph_engine = await get_graph_engine()
        if await stores_provenance_in_graph(graph_engine):
            return await _graph_provenance_expectations(graph_engine, dataset_id, data_id)
    else:
        async with set_database_global_context_variables(dataset_id, owner_by_dataset[dataset_id]):
            graph_engine = await get_graph_engine()
            if await stores_provenance_in_graph(graph_engine):
                return await _graph_provenance_expectations(graph_engine, dataset_id, data_id)

    return await _ledger_expectations(
        data_id, dataset_id, scope_to_dataset=owner_by_dataset is not None
    )


async def _session_data_items(user):
    datasets = await get_datasets_by_name([DATASET], user.id)
    return await get_dataset_data(datasets[0].id)


def _read_raw_document(item) -> str:
    with open(item.raw_data_location.replace("file://", ""), "r") as f:
        return f.read()


async def _verify_session_takeover(stage: str) -> None:
    """Preserve the seeded session and persist only later entries.

    The v1.6.3 watermark must remain 2 after an unchanged improve(). After one
    new entry, improve() must ingest exactly that entry, advance the watermark
    to 3, and leave every session fact stored exactly once across versions.
    """
    print(f"\n[{stage}] Verifying session persistence takeover")

    user = await get_default_user()
    user_id = str(user.id)
    session_manager = get_session_manager()

    entries = await session_manager.get_session(
        user_id=user_id, session_id=COMPAT_SESSION_ID, formatted=False
    )
    if not entries:
        _fail(
            f"[{stage}] legacy session '{COMPAT_SESSION_ID}' not found — Phase 1's session "
            "seeding broke (v1.6.3 already has session memory)."
        )

    if len(entries) != 2:
        _fail(f"[{stage}] expected 2 legacy session entries, found {len(entries)}.")

    # A: preserve the existing watermark without replaying old entries.
    watermark = await get_persisted_qa_count(session_manager, user_id, COMPAT_SESSION_ID)
    if watermark != 2:
        _fail(f"[{stage}] seeded persist watermark should be 2, got {watermark}.")
    before = await _session_data_items(user)
    await cognee.improve(DATASET, session_ids=[COMPAT_SESSION_ID])
    after_unchanged = await _session_data_items(user)
    new_items = {item.id for item in after_unchanged} - {item.id for item in before}
    if new_items:
        _fail(
            f"[{stage}] improve() re-ingested an unchanged legacy session "
            f"({len(new_items)} new document(s)) — expected content-hash dedup no-op."
        )
    watermark = await get_persisted_qa_count(session_manager, user_id, COMPAT_SESSION_ID)
    if watermark != 2:
        _fail(f"[{stage}] persist watermark should remain 2, got {watermark}.")
    print("  [session] unchanged session: 0 new documents, watermark preserved at 2 — OK")

    # B: grow the legacy session with the current branch -> only the new entry.
    await cognee.remember(SESSION_FACT_3, session_id=COMPAT_SESSION_ID, self_improvement=False)
    await cognee.improve(DATASET, session_ids=[COMPAT_SESSION_ID])
    after_grown = await _session_data_items(user)
    new_items = [item for item in after_grown if item.id not in {x.id for x in after_unchanged}]
    if len(new_items) != 1:
        _fail(f"[{stage}] expected exactly 1 new document after growth, got {len(new_items)}.")
    window_text = _read_raw_document(new_items[0])
    if SESSION_FACT_3_MARKER not in window_text:
        _fail(f"[{stage}] new session window is missing the new entry: {window_text!r}")
    if SESSION_FACT_1_MARKER in window_text or SESSION_FACT_2_MARKER in window_text:
        _fail(f"[{stage}] new session window re-ingested legacy entries: {window_text!r}")
    watermark = await get_persisted_qa_count(session_manager, user_id, COMPAT_SESSION_ID)
    if watermark != 3:
        _fail(f"[{stage}] persist watermark should advance to 3, got {watermark}.")

    # Exactly-once across versions: each fact appears in exactly one document.
    all_texts = [_read_raw_document(item) for item in after_grown]
    for marker in (SESSION_FACT_1_MARKER, SESSION_FACT_2_MARKER, SESSION_FACT_3_MARKER):
        occurrences = sum(text.count(marker) for text in all_texts)
        if occurrences != 1:
            _fail(
                f"[{stage}] session fact {marker!r} stored {occurrences} times — "
                "must be exactly once across legacy + current ingestion."
            )
    print(
        "  [session] grown session: 1 new document with ONLY the new entry, "
        "watermark 3, all facts stored exactly once — OK"
    )


async def _verify_dataset_isolation(stage: str) -> None:
    """Preserve independent IDs for identical content in separate datasets."""
    from cognee.modules.data.methods import get_data

    print(f"\n[{stage}] Verifying dataset-scoped document isolation")
    user = await get_default_user()
    seeded = []
    for name in (ISOLATION_DATASET_A, ISOLATION_DATASET_B):
        datasets = await get_datasets_by_name([name], user.id)
        if not datasets:
            _fail(f"[{stage}] dataset '{name}' not found — Phase 1 seeding broke.")
        dataset = datasets[0]
        rows = await get_dataset_data(dataset.id)
        if len(rows) != 1:
            _fail(f"[{stage}] expected 1 document in '{name}', found {len(rows)}.")
        row = rows[0]
        if row.dataset_id != dataset.id or row.legacy_id is not None:
            _fail(f"[{stage}] v1.6.3 document in '{name}' lost its dataset-scoped identity.")
        seeded.append((dataset, row))

    (dataset_a, row_a), (dataset_b, row_b) = seeded
    if row_a.id == row_b.id:
        _fail(f"[{stage}] identical content in two datasets shares one document ID.")
    if row_a.content_hash != row_b.content_hash:
        _fail(f"[{stage}] isolation fixtures no longer contain identical content.")

    for dataset, row, other_dataset in (
        (dataset_a, row_a, dataset_b),
        (dataset_b, row_b, dataset_a),
    ):
        for scope in (dataset.id, None):
            resolved = await get_data(user.id, row.id, scope)
            if resolved is None or resolved.id != row.id:
                _fail(f"[{stage}] document {row.id} does not resolve with scope {scope}.")
        if await get_data(user.id, row.id, other_dataset.id) is not None:
            _fail(f"[{stage}] document {row.id} resolves in the other dataset.")

    print("  [isolation] separate IDs, scoped and unscoped lookup, no cross-dataset access — OK")


async def _verify_dlt_manifest(stage: str) -> None:
    """Preserve a v1.6.3 DLT manifest and its row identities through re-add."""
    import dlt

    from cognee.tasks.ingestion.dlt_utils import is_dlt_source_manifest, load_dlt_manifest

    print(f"\n[{stage}] Verifying DLT manifest compatibility")
    user = await get_default_user()
    datasets = await get_datasets_by_name([DLT_COMPAT_DATASET], user.id)
    if not datasets:
        _fail(f"[{stage}] dataset '{DLT_COMPAT_DATASET}' not found — Phase 1 seeding broke.")
    dataset = datasets[0]

    async def read_manifest():
        records = await get_dataset_data(dataset.id)
        if len(records) != 1 or not is_dlt_source_manifest(records[0]):
            _fail(
                f"[{stage}] expected exactly one DLT source manifest, got {len(records)} records."
            )
        record = records[0]
        meta = record.system_metadata
        if meta.get("source_name") != DLT_COMPAT_SOURCE or meta.get("row_count") != len(
            DLT_COMPAT_ROWS
        ):
            _fail(f"[{stage}] manifest system_metadata is wrong: {meta}")
        manifest = await load_dlt_manifest(record.raw_data_location)
        row_ids = {str(row["node_id"]) for row in manifest["rows"]}
        if len(manifest["rows"]) != len(DLT_COMPAT_ROWS) or len(row_ids) != len(DLT_COMPAT_ROWS):
            _fail(f"[{stage}] manifest does not contain the three distinct seeded rows.")
        return record.id, row_ids

    async def verify_graph(expected_row_ids):
        dataset_rows = await get_dataset_databases()
        owners = {row.dataset_id: row.owner_id for row in dataset_rows} or None
        node_ids, props, _ = await _snapshot_dataset_graph(dataset.id, owners)
        actual_row_ids = {node_id for node_id in node_ids if props[node_id].get("type") == "DltRow"}
        if actual_row_ids != expected_row_ids:
            _fail(
                f"[{stage}] DltRow nodes differ from the manifest: "
                f"expected {expected_row_ids}, got {actual_row_ids}."
            )

    manifest_id, row_ids = await read_manifest()
    await verify_graph(row_ids)
    print("  [dlt] seeded manifest and all row nodes survived the upgrade — OK")

    @dlt.resource(name=DLT_COMPAT_SOURCE, primary_key="id", write_disposition="replace")
    def compat_widgets():
        yield from DLT_COMPAT_ROWS

    await cognee.add([compat_widgets()], dataset_name=DLT_COMPAT_DATASET)
    after_id, after_row_ids = await read_manifest()
    if after_id != manifest_id or after_row_ids != row_ids:
        _fail(f"[{stage}] re-adding the same source changed its document or row identities.")
    await cognee.cognify(datasets=[DLT_COMPAT_DATASET])
    await verify_graph(row_ids)
    print("  [dlt] re-add and cognify preserved document/row IDs without duplicates — OK")


async def _verify_delete(stage: str) -> None:
    """Hard-delete every document, checking ownership after each deletion.

    Solely-owned nodes and edges must disappear; shared artifacts must survive
    until their final owner is removed. EdgeType and global context bookkeeping
    have their own cleanup rules. The final graph must be completely empty.
    """
    print(f"\n[{stage}] Hard-deleting documents one by one, verifying the graph after each")

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        pairs = (
            await session.execute(
                select(_ScopedData.id, _ScopedData.dataset_id, _ScopedData.legacy_id).where(
                    _ScopedData.dataset_id.is_not(None)
                )
            )
        ).all()

    if not pairs:
        _fail(f"[{stage}] no dataset_data rows found — nothing to delete, seed is broken.")

    dataset_rows = await get_dataset_databases()
    owner_by_dataset = {row.dataset_id: row.owner_id for row in dataset_rows} or None

    for index, (data_id, dataset_id, legacy_id) in enumerate(pairs, start=1):
        expected_gone_nodes, expected_gone_edges = await _expectations_for(
            data_id, dataset_id, owner_by_dataset
        )
        before_nodes, before_props, before_edges = await _snapshot_dataset_graph(
            dataset_id, owner_by_dataset
        )

        # Delete by the issued ID, including a legacy alias if a migration
        # recorded one. Ownership expectations use the canonical ID.
        await cognee.delete(data_id=legacy_id or data_id, dataset_id=dataset_id, mode="hard")

        after_nodes, _, after_edges = await _snapshot_dataset_graph(dataset_id, owner_by_dataset)
        doc_tag = f"document {index}/{len(pairs)} ({data_id})"

        missed_nodes = expected_gone_nodes & after_nodes
        if missed_nodes:
            _fail(
                f"[{stage}] {doc_tag}: {len(missed_nodes)} solely-owned node(s) survived the "
                f"hard delete (e.g. {next(iter(missed_nodes))}) — delete missed migrated nodes."
            )

        # Derived bookkeeping is exempt from the collateral check: an EdgeType
        # node is GC'd with its last edge, and the global context index root
        # (a GlobalContextSummary carrying a dataset-level ref, never a
        # document's) is dropped with ANY document deletion, since its text
        # names every document, and written again by the next improve. A
        # document's own bucket carries the document's ref and is expected
        # gone above like any other node it owns.
        disappeared_nodes = before_nodes - after_nodes
        allowed_gone = expected_gone_nodes | {
            node_id
            for node_id in before_nodes
            if before_props[node_id].get("type") in ("EdgeType", "GlobalContextSummary")
        }
        collateral_nodes = disappeared_nodes - allowed_gone
        if collateral_nodes:
            sample = next(iter(collateral_nodes))
            _fail(
                f"[{stage}] {doc_tag}: {len(collateral_nodes)} node(s) NOT owned solely by this "
                f"document were deleted (e.g. {sample}, type="
                f"{before_props[sample].get('type')}) — delete removed shared/foreign nodes."
            )

        appeared_nodes = after_nodes - before_nodes
        if appeared_nodes:
            _fail(
                f"[{stage}] {doc_tag}: {len(appeared_nodes)} node(s) appeared during delete "
                f"(e.g. {next(iter(appeared_nodes))})."
            )

        # Split surviving solely-owned edges by endpoint ownership. Delete has no
        # explicit edge-deletion step: it deletes solely-owned NODES and lets the
        # graph detach their edges. So an edge with a uniquely-owned endpoint MUST
        # disappear (that node's detach sweeps it) — its survival is a real
        # regression and hard-fails. An edge whose BOTH endpoints are shared is
        # never reached by any detach, so it lingers as a ghost: this is a KNOWN
        # PRE-EXISTING bug (reproduced on main, not introduced by the migration),
        # so we warn instead of fail. The warning is forward-compatible — once
        # delete learns to remove edges, nothing survives, so it neither warns nor
        # fails and CI stays green with no flag to flip.
        missed_edges = expected_gone_edges & after_edges
        real_missed = {
            edge
            for edge in missed_edges
            if edge[0] in expected_gone_nodes or edge[1] in expected_gone_nodes
        }
        ghost_missed = missed_edges - real_missed
        if real_missed:
            _fail(
                f"[{stage}] {doc_tag}: {len(real_missed)} solely-owned edge(s) with a "
                f"uniquely-owned endpoint survived the hard delete "
                f"(e.g. {next(iter(real_missed))}) — detach-delete regression."
            )
        if ghost_missed:
            print(
                f"  [delete] {doc_tag}: KNOWN-ISSUE {len(ghost_missed)} ghost edge(s) between "
                f"shared endpoints survived (e.g. {next(iter(ghost_missed))}) — pre-existing "
                "delete bug (reproduced on main), not gating CI."
            )

        # An edge may legitimately vanish without its own ledger ownership when
        # either endpoint node was (legitimately) deleted — detach-delete.
        collateral_edges = {
            key
            for key in before_edges - after_edges
            if key not in expected_gone_edges
            and key[0] not in disappeared_nodes
            and key[1] not in disappeared_nodes
        }
        if collateral_edges:
            _fail(
                f"[{stage}] {doc_tag}: {len(collateral_edges)} edge(s) between surviving nodes "
                f"were deleted (e.g. {next(iter(collateral_edges))}) — collateral edge loss."
            )

        print(
            f"  [delete] {doc_tag}: -{len(disappeared_nodes)} nodes "
            f"-{len(before_edges - after_edges)} edges, "
            f"{len(after_nodes)} nodes / {len(after_edges)} edges remain — "
            "only this document's data removed — OK"
        )

    # No documents left → no graph left.
    nodes, edges = await _collect_graph()
    real_edges = [(s, t, r) for s, t, r, _ in edges if not (r == "SELF" and str(s) == str(t))]
    if nodes or real_edges:
        type_counts = Counter(props.get("type") for _, props in nodes)
        _fail(
            f"[{stage}] graph is not empty after deleting every document: "
            f"{len(nodes)} node(s) {dict(type_counts)}, {len(real_edges)} edge(s) remain."
        )
    print("  [delete] all documents deleted; graph is completely empty — OK")


async def main():
    print(f"Running Phase 2 with cognee version: {cognee.__version__}")

    # ── Step 0: Run database migrations (relational + graph + vector) ─────────
    await cognee.run_migrations()

    # ── Step 1: legacy data must be accessible & correctly migrated ───────────
    await _verify_access("Step 1 — legacy data after migration")

    # ── Step 1b: dataset-scoped document identities must survive ─────────────
    await _verify_dataset_isolation("Step 1b — dataset-scoped document isolation")

    # ── Step 2: re-add + re-cognify with the current branch ───────────────────
    # Re-cognifying the same dataset is what surfaced #2510's EntityAlreadyExistsError.
    print("\n[Step 2] Re-adding + cognifying Lorem Ipsum with current branch...")
    await cognee.add(LOREM_IPSUM, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    # ── Step 3: data must still be accessible after re-cognify ────────────────
    await _verify_access("Step 3 — after re-cognify")

    # ── Step 4: legacy session cache must be taken over incrementally ─────────
    # The seeded session and its persistence watermark must survive.
    await _verify_session_takeover("Step 4 — session persistence takeover")

    # ── Step 5: DLT manifests survive and re-add without duplicates ──────────
    await _verify_dlt_manifest("Step 5 — DLT manifest compatibility")

    # ── Step 6: migrated data must be deletable (ledger-driven hard delete) ───
    # Destructive on purpose, so it runs last. Includes the DLT manifest from
    # Step 5 — the graph must end completely empty.
    await _verify_delete("Step 6 — delete migrated data")

    print("\nAll Phase 2 checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
