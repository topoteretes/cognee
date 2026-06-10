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

Completion searches are still exercised afterwards as a smoke check (must not
raise), just not used as the accessibility gate.
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
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.models import Edge, Node
from cognee.modules.migrations.graph.namespace_entity_type_node_ids import build_id_remap

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


async def main():
    print(f"Running Phase 2 with cognee version: {cognee.__version__}")

    # ── Step 0: Run database migrations (relational + graph + vector) ─────────
    await cognee.run_startup_migrations()

    # ── Step 1: legacy data must be accessible & correctly migrated ───────────
    await _verify_access("Step 1 — legacy data after migration")

    # ── Step 2: re-add + re-cognify with the current branch ───────────────────
    # Re-cognifying the same dataset is what surfaced #2510's EntityAlreadyExistsError.
    print("\n[Step 2] Re-adding + cognifying Lorem Ipsum with current branch...")
    await cognee.add(LOREM_IPSUM, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    # ── Step 3: data must still be accessible after re-cognify ────────────────
    await _verify_access("Step 3 — after re-cognify")

    print("\nAll Phase 2 checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
