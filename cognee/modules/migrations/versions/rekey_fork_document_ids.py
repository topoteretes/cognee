"""Chain migration: re-key fork documents' graph identity to canonical ids.

The dataset-scoping relational backfill (alembic ``d6e8f0a2b4c6``) split
pre-refactor shared ``Data`` rows: non-keeper datasets received fresh
relational ids, with ``Data.legacy_id`` recording the pre-fork original. The
relational ledger's ``data_id`` was repointed there — but the per-dataset
GRAPH stores were deliberately untouched, so a fork dataset's graph document
node (and everything derived from it) still carries the pre-fork id.

This migration closes that gap, per database pair:

1. Build the map ``legacy_id -> id`` from the relational rows of this
   database's dataset (or all datasets in global mode).
2. Skip pairs whose graph holds no old node (nothing to do) or already holds
   the new node (idempotent re-run). On shared stores, pairs whose pre-fork
   id is still owned by a live KEEPER row are provenance-only: the shared
   node, ledger references, chunk properties, and vector payloads stay with
   the keeper; only the fork's own per-dataset provenance keys move. When
   several forks share one legacy id, exactly one deterministic claimant
   re-keys the node; the rest are provenance-only too.
3. Derived stores first (README rule 3): the chunks' vector index rows are
   re-upserted so their ``document_id`` payload scalar (search Evidence)
   cites the canonical id — chunk POINT ids never change; relational ledger
   ``slug`` / edge-endpoint references are updated in place (plain column
   updates — the ledger is only written on non-graph-provenance backends);
   the ``document_id`` display property on the document's chunk nodes is
   updated in the graph.
4. Graph: the document node is re-keyed with full property preservation,
   incident edges recreated on the new endpoint, provenance snapshots
   restored, survivors re-asserted — all via the shared ``_migrate_graph``
   mechanics from the namespace migration.
5. Provenance LAST: on graph-provenance backends (where deletion resolves
   through source ref keys embedding the data id, not the ledger), every
   node/edge ref ``src:v:{dataset}:{legacy_id}`` is moved to the canonical
   key — attach-then-remove, driven by the fork rows so an interrupted run
   converges on re-run.

After this runs, delete/read paths reach fork documents by their canonical id
with no legacy awareness. There is deliberately no runtime fallback for
un-migrated stores: a fork delete there fails fast (DocumentSubgraphNotFoundError)
instead of mutating stores the migration has not touched.

Deliberate boundary: chunk POINT ids are untouched — nothing recomputes them
from the document id on this code line. The payload sync goes through
``index_data_points`` (cognify's own write path), batched the same way
(``index_data_points_batched``), so the stored row shape matches production
writes on every vector backend; text is unchanged, so the re-embedded vector
is equivalent.

Fork rows are rare (same user, identical content, several datasets, before
the upgrade), so this is cheap: one indexed relational query in the common
case, per-document work only where forks exist.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
    make_chunk_source_ref_key,
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models import Data
from cognee.modules.migrations.migration import MigrationContext
from cognee.shared.logging_utils import get_logger

from ._vector_rekey import index_data_points_batched
from .namespace_entity_type_node_ids import _make_node, _migrate_graph

logger = get_logger(__name__)

_IS_PART_OF = "is_part_of"


async def _fork_rows(dataset_id: Optional[UUID]) -> list:
    """(legacy_id, canonical_id, dataset_id) triples for this scope's fork rows."""
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = select(Data.id, Data.legacy_id, Data.dataset_id).filter(Data.legacy_id.is_not(None))
        if dataset_id is not None:
            query = query.filter(Data.dataset_id == dataset_id)
        rows = (await session.execute(query)).all()
    return [(str(row.legacy_id), str(row.id), row.dataset_id) for row in rows]


def _document_chunk_ids(edges: list, document_ids: set) -> dict:
    """chunk_node_id -> old document id, from is_part_of edges into the docs."""
    chunk_owners: dict = {}
    for source_id, target_id, relationship_name, _properties in edges:
        if relationship_name == _IS_PART_OF and str(target_id) in document_ids:
            chunk_owners[str(source_id)] = str(target_id)
    return chunk_owners


async def _update_ledger_references(id_map: dict, dataset_id: Optional[UUID]) -> None:
    """Point ledger slug / edge-endpoint references at the canonical doc id.

    Plain column updates — deletion reads rows by (data_id, dataset_id) and
    deletes graph nodes by the slug values, so only the referenced ids must
    move. Ledger primary keys are left untouched (they are only an upsert
    dedup key for future writes, which use the canonical id anyway).
    """
    from sqlalchemy import update as sql_update

    from cognee.modules.graph.models import Edge as LedgerEdge, Node as LedgerNode

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        for old_id, new_id in id_map.items():
            node_query = sql_update(LedgerNode).where(LedgerNode.slug == UUID(old_id))
            if dataset_id is not None:
                node_query = node_query.where(LedgerNode.dataset_id == dataset_id)
            await session.execute(node_query.values(slug=UUID(new_id)))

            for column in (LedgerEdge.source_node_id, LedgerEdge.destination_node_id):
                edge_query = sql_update(LedgerEdge).where(column == UUID(old_id))
                if dataset_id is not None:
                    edge_query = edge_query.where(LedgerEdge.dataset_id == dataset_id)
                await session.execute(edge_query.values(**{column.key: UUID(new_id)}))
        await session.commit()


def _run_id_for_key(snapshot, source_ref_key: str) -> Optional[str]:
    """The pipeline run id recorded for this key, so rollback linkage survives."""
    for run_ref in getattr(snapshot, "source_run_refs", None) or []:
        if get_source_ref_key_from_source_run_ref(run_ref) == source_ref_key:
            return str(get_pipeline_run_id_from_source_run_ref(run_ref))
    return None


_LADYBUG_PROVIDERS = {"ladybug", "kuzu"}


def _is_ladybug_graph() -> bool:
    """True when the graph store is Ladybug/Kuzu (embedded, string-encoded provenance)."""
    from cognee.infrastructure.databases.graph.config import get_graph_config

    return get_graph_config().graph_database_provider.lower() in _LADYBUG_PROVIDERS


async def _fast_move_provenance_ladybug(graph_engine, old_key: str, new_key: str) -> bool:
    """Rewrite provenance keys in place with two set-based statements.

    The generic path lists every artifact carrying ``old_key``, reads its
    provenance, then attaches the new key and removes the old one — 3 reads
    and 2 writes over the whole subgraph, each a per-row primary-key probe.
    On a 100k-node fork that is ~400k point updates (~11 min measured).

    Ladybug stores provenance as delimiter-wrapped strings, and a re-key only
    changes the data-id half of the key, so the entire move is a substring
    rewrite the engine can do in one scan per artifact kind — no id lists
    cross the process boundary and no row is probed individually.
    ``source_run_refs`` embeds the key verbatim, so the same replace fixes it;
    ``source_dataset_ids`` and ``source_run_ids`` do not contain the data id
    and stay correct untouched.

    Artifacts holding BOTH keys (an earlier run interrupted mid-move) are
    deliberately left alone here — a blind replace would duplicate the new
    key — and fall through to the generic path, which dedupes. Returns False
    when the engine rejects the statements, so the caller can fall back
    wholesale.
    """
    rewrite = """
    MATCH (n:Node)
    WHERE n.source_ref_keys CONTAINS $old_key AND NOT n.source_ref_keys CONTAINS $new_key
    SET n.source_ref_keys = replace(n.source_ref_keys, $old_key, $new_key),
        n.source_run_refs = replace(n.source_run_refs, $old_key, $new_key)
    """
    rewrite_edges = """
    MATCH ()-[r:EDGE]->()
    WHERE r.source_ref_keys CONTAINS $old_key AND NOT r.source_ref_keys CONTAINS $new_key
    SET r.source_ref_keys = replace(r.source_ref_keys, $old_key, $new_key),
        r.source_run_refs = replace(r.source_run_refs, $old_key, $new_key)
    """
    params = {"old_key": old_key, "new_key": new_key}
    try:
        await graph_engine.query(rewrite, params)
        await graph_engine.query(rewrite_edges, params)
    except Exception as error:  # unsupported syntax / non-string provenance
        logger.info(
            "rekey_fork_document_ids: fast provenance move unavailable (%s), "
            "using the generic path",
            error,
        )
        return False
    return True


async def _rekey_graph_provenance(graph_engine, fork_rows: list) -> None:
    """Move graph-embedded provenance from pre-fork keys to canonical keys.

    Graph-provenance backends (e.g. Ladybug) never write the relational
    ledger; deletion resolves through source ref keys embedding the
    (dataset_id, data_id) pair, stamped on every node and edge of the
    document's subgraph. Those keys must follow the canonical id or a
    post-migration delete matches nothing. Attach-then-remove keeps every
    artifact deletable through at least one key throughout, and the sweep is
    driven by the fork rows rather than the node re-key, so a run interrupted
    between the two converges on the next attempt (a moved key finds nothing).
    """
    if not await stores_provenance_in_graph(graph_engine):
        return

    fast_path = _is_ladybug_graph()

    for old_id, new_id, dataset_id in fork_rows:
        old_key = make_source_ref_key(dataset_id, UUID(old_id))
        new_key = make_source_ref_key(dataset_id, UUID(new_id))

        # Set-based rewrite first; whatever it cannot express (artifacts that
        # already carry both keys) is left for the generic path below, which
        # then finds only that residue instead of the whole subgraph.
        if fast_path:
            fast_path = await _fast_move_provenance_ladybug(graph_engine, old_key, new_key)

        try:
            node_ids = await graph_engine.find_nodes_by_source_ref(old_key)
            edge_identities = await graph_engine.find_edges_by_source_ref(old_key)
        except UnsupportedProvenanceCapability:
            return

        if fast_path and not node_ids and not edge_identities:
            logger.info(
                "rekey_fork_document_ids: moved provenance refs from %s to %s "
                "via set-based rewrite",
                old_key,
                new_key,
            )
            continue

        # Group artifacts by pipeline run id and attach per group: a dataset's
        # subgraph overwhelmingly carries one run id, so this collapses the
        # 100k-node / 295k-edge per-item loops — each a write+checkpoint
        # subprocess round-trip whose page churn alone could exhaust kuzu's
        # max_db_size — into a handful of batched calls.
        if node_ids:
            snapshots = await graph_engine.get_node_delete_data(node_ids)
            nodes_by_run: dict = {}
            for node_id in node_ids:
                run_id = _run_id_for_key(snapshots.get(node_id), old_key)
                nodes_by_run.setdefault(run_id, []).append(node_id)
            for run_id, grouped_ids in nodes_by_run.items():
                await graph_engine.attach_node_source_refs(grouped_ids, [new_key], run_id)
            await graph_engine.remove_node_source_refs(node_ids, [old_key])

        if edge_identities:
            edge_snapshots = await graph_engine.get_edge_delete_data(edge_identities)
            edges_by_run: dict = {}
            for edge in edge_identities:
                run_id = _run_id_for_key(edge_snapshots.get(edge), old_key)
                edges_by_run.setdefault(run_id, []).append(edge)
            for run_id, grouped_edges in edges_by_run.items():
                await graph_engine.attach_edge_source_refs(grouped_edges, [new_key], run_id)
            await graph_engine.remove_edge_source_refs(edge_identities, [old_key])

        logger.info(
            "rekey_fork_document_ids: moved provenance refs on %d node(s), %d edge(s) "
            "from %s to %s",
            len(node_ids),
            len(edge_identities),
            old_key,
            new_key,
        )

    await _rekey_chunk_scoped_provenance(graph_engine, fork_rows)


async def _rekey_chunk_scoped_provenance(graph_engine, fork_rows: list) -> None:
    """Move chunk-scoped (source_ref:v2) refs from pre-fork ids to canonical ids.

    Artifacts produced by a chunk carry ``source_ref:v2:{dataset}:{data_id}:
    {chunk_id}`` instead of the document-scoped v1 key, so the v1 sweep above
    never finds them — a post-migration delete by the canonical id would
    strand every chunk-owned node. The chunk ids are not derivable from the
    fork rows, so the keys are discovered from the dataset's ref maps and each
    one is re-issued under the canonical data id (same chunk id),
    attach-then-remove like the v1 sweep so an interrupted run converges.
    """
    ids_by_dataset: dict = {}
    for old_id, new_id, dataset_id in fork_rows:
        ids_by_dataset.setdefault(dataset_id, {})[str(old_id)] = str(new_id)

    for dataset_id, id_map in ids_by_dataset.items():
        try:
            refs_by_node = await graph_engine.find_node_source_refs_by_dataset(str(dataset_id))
            refs_by_edge = await graph_engine.find_edge_source_refs_by_dataset(str(dataset_id))
        except UnsupportedProvenanceCapability:
            return

        def _forked_v2_refs(ref_map: dict) -> dict:
            """old v2 ref key -> (new v2 ref key, holder ids)."""
            moves: dict = {}
            for holder, refs in ref_map.items():
                for ref in refs:
                    try:
                        parsed = parse_source_ref_key(ref)
                    except ValueError:
                        continue
                    if parsed.version != 2 or str(parsed.data_id) not in id_map:
                        continue
                    new_key = make_chunk_source_ref_key(
                        dataset_id, UUID(id_map[str(parsed.data_id)]), parsed.chunk_id
                    )
                    moves.setdefault(ref, (new_key, []))[1].append(holder)
            return moves

        node_moves = _forked_v2_refs(refs_by_node)
        edge_moves = _forked_v2_refs(refs_by_edge)

        for old_ref, (new_key, node_ids) in node_moves.items():
            snapshots = await graph_engine.get_node_delete_data(node_ids)
            for node_id in node_ids:
                run_id = _run_id_for_key(snapshots.get(node_id), old_ref)
                await graph_engine.attach_node_source_refs([node_id], [new_key], run_id)
            await graph_engine.remove_node_source_refs(node_ids, [old_ref])

        for old_ref, (new_key, edge_identities) in edge_moves.items():
            edge_snapshots = await graph_engine.get_edge_delete_data(edge_identities)
            for edge in edge_identities:
                run_id = _run_id_for_key(edge_snapshots.get(edge), old_ref)
                await graph_engine.attach_edge_source_refs([edge], [new_key], run_id)
            await graph_engine.remove_edge_source_refs(edge_identities, [old_ref])

        if node_moves or edge_moves:
            logger.info(
                "rekey_fork_document_ids: moved %d chunk-scoped node ref key(s), "
                "%d edge ref key(s) in dataset %s",
                len(node_moves),
                len(edge_moves),
                dataset_id,
            )


class ForkChunkIndexPoint(DataPoint):
    """Carrier for re-upserting a fork chunk's vector index row.

    Passed to ``vector_engine.index_data_points`` — the same write path cognify
    uses — so the adapter builds its own ``IndexSchema`` row from these fields
    (reference scalars pulled via ``getattr``) and the stored payload shape
    matches every existing row, on every vector backend. The point id is the
    chunk id, which never changes; only the ``document_id`` payload scalar
    moves.
    """

    text: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    source_chunk_id: Optional[str] = None
    metadata: dict = {"index_fields": ["text"]}


async def _sync_chunk_vector_payloads(
    vector_engine, properties_by_node: dict, normalized_edges: list, claimant_pairs: list
) -> None:
    """Re-upsert fork chunks' vector rows so payload ``document_id`` is canonical.

    The chunk index payload carries ``document_id`` as a search-Evidence
    scalar; without this it would cite the pre-fork id until the document's
    next natural rewrite. Driven by the claimant pairs (edges into either the
    legacy or the canonical document node are matched), so a run interrupted
    around the graph re-key converges here on re-run; the upsert is
    idempotent. Text is unchanged, so re-embedding yields an equivalent
    vector. Skipped when the chunk collection does not exist (graph-only
    pipelines never vector-index chunks).
    """
    if vector_engine is None:
        return

    doc_targets: dict = {}
    for old_id, new_id, _dataset_id in claimant_pairs:
        doc_targets[old_id] = new_id
        doc_targets[new_id] = new_id

    carriers = []
    for source_id, target_id, relationship_name, _properties in normalized_edges:
        target = doc_targets.get(target_id)
        if relationship_name != _IS_PART_OF or target is None:
            continue
        chunk_properties = properties_by_node.get(source_id)
        if not chunk_properties or "text" not in chunk_properties:
            continue
        carriers.append(
            ForkChunkIndexPoint(
                id=UUID(source_id),
                text=chunk_properties["text"],
                document_id=target,
                document_name=chunk_properties.get("document_name"),
                chunk_index=chunk_properties.get("chunk_index"),
                source_chunk_id=chunk_properties.get("source_chunk_id"),
                importance_weight=chunk_properties.get("importance_weight"),
                belongs_to_set=chunk_properties.get("belongs_to_set"),
            )
        )
    if not carriers:
        return

    if not await vector_engine.has_collection("DocumentChunk_text"):
        return
    await index_data_points_batched(vector_engine, "DocumentChunk", "text", carriers)
    logger.info(
        "rekey_fork_document_ids: synced document_id payload on %d chunk vector row(s)",
        len(carriers),
    )


async def _keeper_blocked_pairs(fork_rows: list, dataset_id: Optional[UUID]) -> set:
    """Old-position ids whose node identity is owned by an UNRELATED live row.

    On shared stores (access control off — one graph for every dataset) the
    keeper dataset kept the pre-fork id through the backfill, and the single
    graph document node under that id IS the keeper's document. A fork pair
    whose original survives as the keeper must never re-key the node, ledger
    references, chunk properties, or vector payloads — only its own
    per-dataset provenance keys move. With isolated stores the keeper lives
    in a different database pair, so the scoped query returns nothing and
    the full re-key applies.

    Pair-aware on purpose: in the downgrade direction the old position holds
    the canonical id, whose live row is the pair ITSELF (its ``legacy_id`` is
    the pair's target) — that is the row being moved, not a competing owner,
    so it must not block.
    """
    if not fork_rows:
        return set()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        query = select(Data.id, Data.legacy_id).filter(
            Data.id.in_({UUID(old_id) for old_id, _new_id, _dataset_id in fork_rows})
        )
        if dataset_id is not None:
            query = query.filter(Data.dataset_id == dataset_id)
        owners = {str(row.id): row.legacy_id for row in (await session.execute(query)).all()}

    blocked = set()
    for old_id, new_id, _dataset_id in fork_rows:
        if old_id not in owners:
            continue  # no live row under the old id: nothing to protect
        if str(owners[old_id]) != new_id:
            blocked.add(old_id)  # an unrelated row (the keeper) owns this identity
    return blocked


def _claimant_pairs(fork_rows: list, blocked: set, properties_by_node: dict) -> list:
    """One (old_id, new_id, dataset_id) claimant per re-keyable pre-fork id.

    A graph node can carry only one id, so when several forks share a legacy
    id (shared store, keeper deleted after the backfill) exactly one fork
    claims the node re-key; the rest stay provenance-only. A fork whose
    canonical node already exists in the graph wins (a partial prior run is
    reality, not preference); otherwise the lowest canonical id, so re-runs
    pick the same claimant.
    """
    by_old: dict = {}
    for old_id, new_id, dataset_id in fork_rows:
        if old_id in blocked:
            continue
        by_old.setdefault(old_id, []).append((new_id, dataset_id))

    claimants = []
    for old_id, candidates in by_old.items():
        candidates.sort(key=lambda pair: (pair[0] not in properties_by_node, pair[0]))
        new_id, dataset_id = candidates[0]
        claimants.append((old_id, new_id, dataset_id))
    return claimants


async def _apply(context: MigrationContext, fork_rows: list) -> None:
    graph_engine = context.graph_engine
    nodes, edges = await graph_engine.get_graph_data()
    properties_by_node = {str(node_id): dict(props or {}) for node_id, props in nodes}
    normalized_edges = [
        (str(edge[0]), str(edge[1]), edge[2], (edge[3] if len(edge) > 3 else {}) or {})
        for edge in edges
    ]

    blocked = await _keeper_blocked_pairs(fork_rows, context.dataset_id)
    claimant_pairs = _claimant_pairs(fork_rows, blocked, properties_by_node)

    id_map: dict = {}
    properties_by_id: dict = {}
    for old_id, new_id, _dataset_id in claimant_pairs:
        if old_id not in properties_by_node:
            continue  # not in this graph, or already migrated and old node gone
        if new_id in properties_by_node:
            continue  # idempotent re-run: canonical node already present
        properties = dict(properties_by_node[old_id])
        properties["id"] = old_id
        properties_by_id[old_id] = properties
        id_map[old_id] = new_id

    # Derived store, claimant-driven: chunk vector payloads first. Keeper and
    # non-claimant pairs are excluded — their chunks stay with the node's id.
    await _sync_chunk_vector_payloads(
        context.vector_engine, properties_by_node, normalized_edges, claimant_pairs
    )

    if id_map:
        await _rekey_graph_nodes(
            graph_engine, id_map, properties_by_id, properties_by_node, normalized_edges, context
        )

    # Provenance refs move last: the node re-key restores the pre-fork keys
    # onto the canonical document node verbatim, so the sweep must follow it.
    await _rekey_graph_provenance(graph_engine, fork_rows)


async def _rekey_graph_nodes(
    graph_engine,
    id_map: dict,
    properties_by_id: dict,
    properties_by_node: dict,
    normalized_edges: list,
    context: MigrationContext,
) -> None:
    # Derived stores first: relational ledger references move before the graph.
    await _update_ledger_references(id_map, context.dataset_id)

    # Chunk display metadata: the chunks' document_id property follows the
    # canonical id (property update via full-property upsert — MERGE replaces
    # the whole blob, so the complete stored property set is carried).
    chunk_owners = _document_chunk_ids(normalized_edges, set(id_map.keys()))
    chunk_carriers = []
    for chunk_id, old_document_id in chunk_owners.items():
        chunk_properties = properties_by_node.get(chunk_id)
        if not chunk_properties or "document_id" not in chunk_properties:
            continue
        chunk_carriers.append(
            _make_node({**chunk_properties, "id": chunk_id, "document_id": id_map[old_document_id]})
        )
    if chunk_carriers:
        await graph_engine.add_nodes(chunk_carriers)

    # Graph last: re-key the document nodes themselves (also updates each
    # node's own document_id property when it mirrors the node id).
    for old_id in id_map:
        if str(properties_by_id[old_id].get("document_id")) == old_id:
            properties_by_id[old_id]["document_id"] = id_map[old_id]
    remapped_edges = await _migrate_graph(graph_engine, id_map, properties_by_id, normalized_edges)
    logger.info(
        "rekey_fork_document_ids: re-keyed %d fork document node(s), %d edge(s)",
        len(id_map),
        remapped_edges,
    )


async def migrate(context: MigrationContext) -> None:
    fork_rows = await _fork_rows(context.dataset_id)
    if not fork_rows:
        return
    await _apply(context, fork_rows)


async def downgrade(context: MigrationContext) -> None:
    """Reverse: move fork document nodes back to their pre-fork (legacy) ids.

    ROLLBACK ORDER: run this BEFORE the alembic downgrade (``d6e8f0a2b4c6``).
    The reverse map here is built from ``Data.legacy_id``; the alembic
    downgrade drops that column, after which this no-ops and fork graphs stay
    stranded on canonical ids the old code cannot resolve.
    """
    forward = await _fork_rows(context.dataset_id)
    if not forward:
        return
    await _apply(context, [(new_id, old_id, dataset_id) for old_id, new_id, dataset_id in forward])
