"""
Backwards Compatibility Test

Phase 1 -  run with a legacy cognee version

Seeds the database with Lorem Ipsum data: add → cognify → search.

Phase 2 - run with current Cognee branch

Runs startup migrations (relational + graph + vector), then verifies that the
legacy-seeded graph, vector AND relational-ledger data is actually *accessible
and correct* after migration, then re-cognifies and verifies again.

Why these specific checks?
--------------------------
A migration can leave the database query-able yet semantically broken — e.g.
the node-ID change in PR #2515 (issue #2510) did not raise, it silently left
old graphs with the pre-namespacing IDs, breaking projection/search and causing
re-cognify to raise ``EntityAlreadyExistsError``.

Gating on completion-style search (GRAPH_COMPLETION/RAG_COMPLETION) is NOT
reliable: the LLM can return a non-empty "I don't know" answer even when graph
retrieval is empty. So this test gates on signals that genuinely fail when data
is inaccessible:

* graph — read the dataset's graph directly and assert it has nodes AND that no
  Entity/EntityType node is still on the pre-#2515 ID scheme (which would mean
  the migration didn't run/complete);
* ledger — read the relational ``nodes``/``edges`` tables and assert every id
  the migration could have moved still resolves to a live graph node (a stale
  ledger id silently orphans nodes on delete);
* vector — a CHUNKS search (raw vector retrieval) must return results.
* delete — hard-delete documents one by one; after each, only the nodes/edges
  owned solely by that document disappear, shared ones survive until their last
  owner goes, and the graph ends empty. Delete resolves nodes via the ledger
  (``nodes.slug``), so this is the end-to-end proof of ledger/graph lockstep: a
  stale ledger id makes delete silently orphan a migrated node, which the static
  checks above can't catch.

Completion searches are still exercised afterwards as a smoke check (must not
raise), just not used as the accessibility gate.

* sessions — the current branch must take over the legacy session cache that
  Phase 1 seeded, without re-ingesting: improve() on the untouched session
  ingests nothing new (the full window is byte-identical to the legacy
  snapshot, content-hash dedup absorbs it) and writes the persist watermark;
  after growing the session, the next improve() ingests exactly one document
  containing ONLY the new entry; every session fact ends up stored exactly
  once across both versions. A missing legacy session is a hard failure — the
  CI pin never goes below v1.2.0 (the earliest release with the SQL session
  cache), so absence means Phase 1's seeding broke.
"""

import asyncio
import sys
from collections import Counter

import cognee
from sqlalchemy import select

from cognee.api.v1.search import SearchType
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_persist_watermark import get_persisted_qa_count
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.data.models import DatasetData
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
    """CHUNKS is raw vector retrieval (no LLM fallback); empty == inaccessible."""
    results = await cognee.search(query_type=SearchType.CHUNKS, query_text=SEARCH_QUERY)
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

    A legacy seed predating the ledger leaves the tables empty; that is not a
    failure here (re-cognify on the current branch repopulates them and Step 3
    re-checks), but reading them still proves the relational store is reachable.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        node_rows = (await session.scalars(select(Node))).all()
        edge_rows = (await session.scalars(select(Edge))).all()

    if not node_rows and not edge_rows:
        print("  [ledger] empty (legacy seed predates the node/edge ledger) — read OK, skipped")
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
        await cognee.search(query_type=query_type, query_text=SEARCH_QUERY)
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


async def _session_data_items(user):
    datasets = await get_datasets_by_name([DATASET], user.id)
    return await get_dataset_data(datasets[0].id)


def _read_raw_document(item) -> str:
    with open(item.raw_data_location.replace("file://", ""), "r") as f:
        return f.read()


async def _verify_session_takeover(stage: str) -> None:
    """Verify the current branch takes over the legacy session cache incrementally.

    A. improve() on the untouched legacy session must ingest NOTHING new — the
       full window is byte-identical to the legacy whole-session snapshot, so
       content-hash dedup absorbs it — and must write the persist watermark.
    B. After the session grows by one entry, improve() must ingest exactly ONE
       new document containing ONLY the new entry (legacy entries are never
       re-ingested), and every session fact must be stored exactly once across
       both versions.
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
            "seeding broke (the CI pin never goes below v1.2.0, which has session memory)."
        )

    if len(entries) != 2:
        _fail(f"[{stage}] expected 2 legacy session entries, found {len(entries)}.")

    # A: unchanged legacy session -> dedup no-op + watermark heals.
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
        _fail(f"[{stage}] persist watermark should heal to 2, got {watermark}.")
    print("  [session] unchanged legacy session: 0 new documents, watermark healed to 2 — OK")

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


async def _verify_delete(stage: str) -> None:
    """Hard-delete documents one by one, checking the graph after each.

    Delete resolves graph nodes by the ledger's ``nodes.slug``, so deleting
    migrated data exercises ledger/graph lockstep — a stale slug makes delete
    silently miss nodes rather than raise, so we gate on the resulting graph, not
    the call. After each delete, diff before/after: nodes/edges owned only by
    that document are gone (no orphans), shared ones survive until their last
    owner goes (no collateral damage; EdgeType nodes are exempt — derived
    bookkeeping GC'd with their last edge), and nothing new appears. After the
    last document the graph must be empty.
    """
    print(f"\n[{stage}] Hard-deleting documents one by one, verifying the graph after each")

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        pairs = (await session.execute(select(DatasetData.data_id, DatasetData.dataset_id))).all()

    if not pairs:
        _fail(f"[{stage}] no dataset_data rows found — nothing to delete, seed is broken.")

    dataset_rows = await get_dataset_databases()
    owner_by_dataset = {row.dataset_id: row.owner_id for row in dataset_rows} or None

    for index, (data_id, dataset_id) in enumerate(pairs, start=1):
        expected_gone_nodes, expected_gone_edges = await _ledger_expectations(
            data_id, dataset_id, scope_to_dataset=owner_by_dataset is not None
        )
        before_nodes, before_props, before_edges = await _snapshot_dataset_graph(
            dataset_id, owner_by_dataset
        )

        await cognee.delete(data_id=data_id, dataset_id=dataset_id, mode="hard")

        after_nodes, _, after_edges = await _snapshot_dataset_graph(dataset_id, owner_by_dataset)
        doc_tag = f"document {index}/{len(pairs)} ({data_id})"

        missed_nodes = expected_gone_nodes & after_nodes
        if missed_nodes:
            _fail(
                f"[{stage}] {doc_tag}: {len(missed_nodes)} solely-owned node(s) survived the "
                f"hard delete (e.g. {next(iter(missed_nodes))}) — delete missed migrated nodes."
            )

        disappeared_nodes = before_nodes - after_nodes
        allowed_gone = expected_gone_nodes | {
            node_id for node_id in before_nodes if before_props[node_id].get("type") == "EdgeType"
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

    # ── Step 2: re-add + re-cognify with the current branch ───────────────────
    # Re-cognifying the same dataset is what surfaced #2510's EntityAlreadyExistsError.
    print("\n[Step 2] Re-adding + cognifying Lorem Ipsum with current branch...")
    await cognee.add(LOREM_IPSUM, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    # ── Step 3: data must still be accessible after re-cognify ────────────────
    await _verify_access("Step 3 — after re-cognify")

    # ── Step 4: legacy session cache must be taken over incrementally ─────────
    # A missing legacy session hard-fails: the pin never goes below v1.2.0.
    await _verify_session_takeover("Step 4 — session persistence takeover")

    # ── Step 5: migrated data must be deletable (ledger-driven hard delete) ───
    # Destructive on purpose, so it runs last.
    await _verify_delete("Step 5 — delete migrated data")

    print("\nAll Phase 2 checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
