"""Graph migration: move Entity / EntityType node IDs to the model-owned scheme.

Background
----------
Historically node IDs were generated from the bare name::

    type node   -> generate_node_id(node_type)
    entity node -> generate_node_id(node_id)

so ``Entity("institution")`` and ``EntityType("institution")`` hashed to the
*same* UUID, which caused ``EntityAlreadyExistsError`` on a second ``cognify``
(topoteretes/cognee#2515 / issue #2510).

IDs are now derived by the model itself via ``DataPoint.id_for`` —
``uuid5(NAMESPACE_OID, f"{ClassName}:{normalized_value}")`` — so the namespace
comes from the node's class (``Entity:…`` vs ``EntityType:…``) and can neither
collide nor be forgotten at a call site. This migration brings graphs created
with the older schemes up to that model-owned scheme.

Three stores, one id
--------------------
An Entity/EntityType id is a primary key in THREE places, and all three must
move together or the system splits:

  * graph DB    — the node id
  * vector DB   — the point id (``str(data_point.id)``); used to seed graph
    traversal during retrieval, so a stale vector id silently breaks search.
  * relational ledger — ``nodes.slug`` and ``edges.source_node_id`` /
    ``edges.destination_node_id``; the delete system deletes graph nodes *by
    these values*, so a stale ledger id silently orphans migrated nodes.

Plus one derived id: when triplet embeddings are enabled, ``Triplet_text``
point ids are hashed from the *edge endpoint ids* (see ``add_data_points``), so
remapping an entity id also moves every triplet point keyed off its edges.

Vector payload shape
--------------------
Every vector point cognee writes goes through ``index_data_points``, which
stores the adapter's own ``IndexSchema`` shape — ``{id, text, belongs_to_set}``
— NOT a dump of the source model. Re-keying therefore re-inserts through the
same ``vector_engine.index_data_points`` write path (a small carrier provides
the new id and the unchanged text), never by reconstructing ``Entity(**payload)``,
so the stored shape stays identical to cognify-written rows on every backend.

Vector re-key strategy
----------------------
Re-keying a point means moving its row to a new primary key. Two paths:

  * native (LanceDB, PGVector) — the stored row, INCLUDING its vector, is moved
    to the new id directly inside the store (SQL UPDATE / row copy). No
    re-embedding, no embedding backend needed, no per-point cost beyond the
    store operation. Dispatched by adapter type in ``_rekey_native``.
  * generic (every other backend) — the point is re-inserted through
    ``index_data_points`` under the new id, which re-embeds the (unchanged)
    text; the old point is then deleted. Needs the embedding backend available,
    exactly as ``cognify`` does.

How the new ID is recovered
---------------------------
Both ``generate_node_id`` and the model's identity normalization lower-case,
strip apostrophes and turn spaces into underscores, and ``generate_node_name``
lower-cases and strips apostrophes. So the stored ``name`` property is enough to
recompute the node's ID under any scheme — no need to reverse the UUID hash::

    Entity.id_for(name)                 -> the new, model-owned ID
    generate_node_id(name)              -> the released "bare" ID
    generate_node_id(f"entity:{name}")  -> the interim ID (this branch, never released)

A node is only remapped when its current ID matches one of the recognized old
schemes for its kind; anything else is left untouched (already migrated, or an
ID not derived from its name).

Ordering and retry-safety
--------------------------
The ``{old_id: new_id}`` map can only be computed while the graph still holds
the old ids, so we re-key the *derived* stores first and rename the graph last:

    1. vector re-key   (entity points, then triplet points)
    2. ledger UPDATEs  (nodes.slug, edges.source/destination, legacy ledger)
    3. graph rename    (add new nodes, rewire edges, delete old nodes)

Every step is idempotent (re-keying an already-migrated store is a no-op), so a
crash at any point leaves a partially-migrated database that the next startup
finishes — the map is re-derived from whatever old-scheme nodes remain. An
Entity whose vector point cannot be re-keyed (no text in the payload and no
name on the graph node) is dropped from the map entirely, so it stays fully on
the old scheme in all three stores (consistent) rather than half-migrated, and
is logged.

Only the portable :class:`GraphDBInterface` / :class:`VectorDBInterface`
operations are used. Re-added graph nodes are wrapped by per-class carriers
(``_make_node``) that satisfy every adapter's ``add_nodes`` contract
(``model_dump`` for Ladybug/Postgres; iteration + correct class name for Neo4j's
label).

Caveats:
- ``get_graph_data`` does not return ``created_at``/``updated_at``, so a
  remapped graph node's timestamps are reset to the migration time. Semantic
  properties are preserved; original timestamps are not recoverable here.
- On backends without a native re-key path (see "Vector re-key strategy"), the
  vector re-key re-embeds the (unchanged) text via ``index_data_points`` and
  needs the embedding backend available, exactly as ``cognify`` does. LanceDB
  and PGVector move the stored vector and never re-embed.
- The legacy ``graph_relationship_ledger`` has no ``dataset_id`` column, so its
  rows are updated globally; this is benign because old ids are deterministic
  and every dataset is migrated in the same run.
"""

import logging
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.migrations.migration import MigrationContext

logger = logging.getLogger(__name__)

# Stored DataPoint ``type`` property -> (model class that owns the new id via
# ``id_for``, interim per-kind prefix this branch briefly used before id_for).
_NODE_KINDS = {
    "Entity": (Entity, "entity"),
    "EntityType": (EntityType, "type"),
}


# Per-class carrier types so a re-added graph node satisfies EVERY adapter's
# add_nodes contract, not just one. Adapters read incoming nodes differently:
#   * Ladybug / Postgres call ``node.model_dump()`` (or ``vars()``).
#   * Neo4j calls ``dict(node)`` (requires the object to be iterable) and uses
#     the serialized ``type`` (falling back to ``type(node).__name__``) as label.
# A plain object only satisfies the first; the Neo4j path must also be iterable
# AND carry the real class name (``Entity``/``EntityType``). We build one tiny
# class per node-type name, cached, so ``type(carrier).__name__`` is correct.
_carrier_classes: dict = {}


def _make_node(properties: dict):
    """Return a graph-node carrier (id + all properties) accepted by any adapter."""
    node_type = properties.get("type") or "Node"
    carrier_cls = _carrier_classes.get(node_type)
    if carrier_cls is None:
        carrier_cls = type(
            node_type,  # __name__ -> correct Neo4j label (Entity/EntityType)
            (),
            {
                "__init__": lambda self, props: self.__dict__.update(props),
                "__iter__": lambda self: iter(self.__dict__.items()),  # dict(node) for Neo4j
                "model_dump": lambda self: dict(self.__dict__),  # Ladybug/Postgres
            },
        )
        _carrier_classes[node_type] = carrier_cls
    return carrier_cls(properties)


def build_id_remap(nodes: list) -> dict:
    """Return ``{old_id: new_id}`` for Entity/EntityType nodes on a recognized old scheme.

    A node is included only when its current id matches a known historical
    derivation of its stored ``name`` (bare or interim-prefixed) and differs from
    the model-owned id. Fresh nodes already on the new scheme — and nodes whose id
    was derived from a value other than the stored name (the rare ``node.id !=
    node.name`` case, which can't be recomputed and the original scheme couldn't
    either) — are intentionally excluded. Reused by the backwards-compat test so
    "still needs migrating" means exactly "this migration would remap it".
    """
    id_map: dict = {}
    skipped = 0

    for node_id, properties in nodes:
        kind = _NODE_KINDS.get(properties.get("type"))
        if kind is None:
            # Only Entity / EntityType IDs changed.
            continue
        model_cls, interim_prefix = kind

        name = properties.get("name")
        if not name:
            continue

        new_id = str(model_cls.id_for(name))
        if node_id == new_id:
            # Already on the model-owned scheme.
            continue

        # Recognized historical schemes this node could still be on.
        recognized_old_ids = {
            str(generate_node_id(name)),  # released "bare" scheme
            str(generate_node_id(f"{interim_prefix}:{name}")),  # interim, never released
        }
        if node_id in recognized_old_ids:
            id_map[node_id] = new_id
        else:
            # ID was not derived from this node's name; cannot remap safely.
            skipped += 1
            logger.warning(
                "Skipping %s node %s: id is not a recognized old hash of name %r.",
                properties.get("type"),
                node_id,
                name,
            )

    if skipped:
        logger.warning("Entity/EntityType ID migration skipped %d unrecognized node(s).", skipped)

    return id_map


def _index_fields(model_cls) -> list[str]:
    """The vector collections for a kind are ``f'{Type}_{field}'`` per index field."""
    metadata_default = model_cls.model_fields["metadata"].default or {}
    return list(metadata_default.get("index_fields") or [])


class _RekeyedPoint(DataPoint):
    """Carrier for re-inserting an existing vector point under a new id.

    Passed to ``vector_engine.index_data_points`` — the same write path cognify
    uses — so the adapter builds its own ``IndexSchema`` row ``{id, text,
    belongs_to_set}`` from it and the stored shape matches every existing row,
    on every vector backend.
    """

    text: str
    metadata: dict = {"index_fields": ["text"]}


async def _rekey_lancedb(vector_engine, collection: str, id_map: dict) -> None:
    """Move LanceDB rows to their new ids, carrying the stored vector over.

    Reads the full rows (vector included), re-adds them under the new id with
    the payload's embedded ``id`` updated, then deletes the old rows. Uses
    plain ``add`` (append) — NOT ``merge_insert``, which lance 0.32 can panic
    with on tables carrying deletion vectors. Compacts afterwards when the
    table supports it (the subprocess-proxy table does not expose
    ``optimize``), so later cognify ``merge_insert`` writes see a clean table.
    """
    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        table = await vector_engine.get_collection(collection)
    except CollectionNotFoundError:
        return

    old_ids = [str(old_id) for old_id in id_map]
    if len(old_ids) == 1:
        where = f"id = '{old_ids[0]}'"
    else:
        id_list = ", ".join(f"'{old_id}'" for old_id in old_ids)
        where = f"id IN ({id_list})"

    rows = await table.query().where(where).to_list()
    if not rows:
        return

    new_rows = []
    for row in rows:
        new_id = id_map[str(row["id"])]
        new_row = dict(row)
        new_row["id"] = new_id
        payload = dict(new_row.get("payload") or {})
        payload["id"] = new_id
        new_row["payload"] = payload
        new_rows.append(new_row)

    await table.add(new_rows)
    await vector_engine.delete_data_points(collection, [str(row["id"]) for row in rows])

    optimize = getattr(table, "optimize", None)
    if optimize is not None:
        await optimize()


async def _rekey_pgvector(vector_engine, collection: str, id_map: dict) -> None:
    """Move PGVector rows to their new ids with SQL UPDATEs — the vector column
    never moves, so nothing is re-embedded."""
    from sqlalchemy import select, update

    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        table = await vector_engine.get_table(collection)
    except CollectionNotFoundError:
        return

    async with vector_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(table.c.id, table.c.payload).where(table.c.id.in_(list(id_map)))
            )
        ).all()
        for row in rows:
            new_id = id_map[str(row.id)]
            payload = dict(row.payload or {})
            payload["id"] = new_id
            await session.execute(
                update(table).where(table.c.id == row.id).values(id=new_id, payload=payload)
            )
        await session.commit()


async def _rekey_native(vector_engine, collection: str, id_map: dict) -> bool:
    """Vector-preserving re-key for backends that support it.

    Returns ``True`` when the collection was handled natively (vectors moved,
    nothing re-embedded); ``False`` means the caller must use the generic
    re-embed path. Dispatch is by adapter class name so the optional backend
    packages are never imported here.
    """
    if not id_map:
        return True

    # NOT type(vector_engine): the engine arrives wrapped (_VectorEngineHandle /
    # the cache's _LeasedValueProxy), and both spoof ``__class__`` to the real
    # adapter class precisely so checks like this resolve through the wrapper.
    adapter = vector_engine.__class__.__name__
    if adapter == "LanceDBAdapter":
        await _rekey_lancedb(vector_engine, collection, id_map)
        return True
    if adapter == "PGVectorAdapter":
        await _rekey_pgvector(vector_engine, collection, id_map)
        return True
    return False


async def _migrate_vector(vector_engine, id_map: dict, properties_by_id: dict) -> set:
    """Re-key Entity/EntityType vector points from old id to new id.

    Real payloads are ``IndexSchema``-shaped (``{id, text, belongs_to_set}`` —
    written by ``index_data_points``), so the embedded value is read from the
    payload's ``text`` (falling back to the graph node's own index property).

    Returns the set of old ids whose point could NOT be re-keyed (no text to
    re-embed); the caller drops these from the id map so the entity stays
    consistently on the old scheme everywhere rather than being half-migrated.
    Points absent from the vector store are simply skipped (no obstacle), so
    this is a no-op on a graph-only deployment and idempotent on re-runs (the
    old id is already gone).
    """
    failed: set = set()
    if vector_engine is None:
        return failed

    # Group old ids by kind so we hit the right collection(s).
    ids_by_kind: dict = {}
    for old_id in id_map:
        node_type = properties_by_id.get(old_id, {}).get("type")
        ids_by_kind.setdefault(node_type, []).append(old_id)

    for node_type, old_ids in ids_by_kind.items():
        kind = _NODE_KINDS.get(node_type)
        if kind is None:
            continue
        model_cls, _ = kind

        for index_field in _index_fields(model_cls):
            collection = f"{node_type}_{index_field}"

            # Vector-preserving fast path; nothing can fail per-point here
            # (no text needed), so no ids join the failed set.
            if await _rekey_native(
                vector_engine, collection, {old_id: id_map[old_id] for old_id in old_ids}
            ):
                continue

            rows = await vector_engine.retrieve(collection, [str(o) for o in old_ids])

            new_points = []
            migrated_old_ids = []
            for row in rows:
                old_id = str(row.id)
                new_id = id_map.get(old_id)
                if new_id is None:
                    continue
                payload = dict(row.payload or {})
                text = payload.get("text") or properties_by_id.get(old_id, {}).get(index_field)
                if not text:
                    failed.add(old_id)
                    logger.warning(
                        "Vector re-key skipped for %s point %s: no embeddable text in "
                        "the payload or on the graph node.",
                        node_type,
                        old_id,
                    )
                    continue
                new_points.append(
                    _RekeyedPoint(
                        id=UUID(new_id),
                        text=text,
                        belongs_to_set=payload.get("belongs_to_set") or [],
                    )
                )
                migrated_old_ids.append(old_id)

            if new_points:
                await vector_engine.index_data_points(node_type, index_field, new_points)
                await vector_engine.delete_data_points(collection, migrated_old_ids)

    return failed


def _build_triplet_remap(edges: list, id_map: dict) -> dict:
    """``{old_triplet_point_id: new_triplet_point_id}`` for edges touching remapped nodes.

    ``Triplet_text`` point ids are ``generate_node_id(source_id + relationship +
    target_id)`` (see ``add_data_points._create_triplets_from_graph``), so when
    an edge endpoint moves, its triplet point id moves too. Mirrors the delete
    path, which recomputes these ids from the (migrated) ledger edge endpoints.
    """
    triplet_map: dict = {}
    for source_id, target_id, relationship_name, _ in edges:
        # Skip Ladybug's synthetic placeholder edges (see _migrate_graph).
        if relationship_name == "SELF" and source_id == target_id:
            continue
        new_source = id_map.get(source_id, source_id)
        new_target = id_map.get(target_id, target_id)
        if new_source == source_id and new_target == target_id:
            continue
        old_triplet_id = str(generate_node_id(str(source_id) + relationship_name + str(target_id)))
        new_triplet_id = str(
            generate_node_id(str(new_source) + relationship_name + str(new_target))
        )
        triplet_map[old_triplet_id] = new_triplet_id
    return triplet_map


async def _migrate_triplet_vector(vector_engine, triplet_map: dict) -> None:
    """Re-key ``Triplet_text`` points whose edge endpoints were remapped.

    The collection exists only when triplet embedding was enabled; like the
    delete path (``delete_from_graph_and_vector``), a missing collection is
    treated as nothing-to-do.
    """
    if vector_engine is None or not triplet_map:
        return

    # Vector-preserving fast path (missing collection is a no-op inside).
    if await _rekey_native(vector_engine, "Triplet_text", triplet_map):
        return

    try:
        rows = await vector_engine.retrieve("Triplet_text", list(triplet_map.keys()))
    except Exception:  # noqa: BLE001 - collection absent when feature is off
        return

    new_points = []
    migrated_old_ids = []
    for row in rows:
        old_id = str(row.id)
        new_id = triplet_map.get(old_id)
        if new_id is None:
            continue
        payload = dict(row.payload or {})
        text = payload.get("text")
        if not text:
            logger.warning("Triplet re-key skipped for point %s: payload has no text.", old_id)
            continue
        new_points.append(
            _RekeyedPoint(
                id=UUID(new_id),
                text=text,
                belongs_to_set=payload.get("belongs_to_set") or [],
            )
        )
        migrated_old_ids.append(old_id)

    if new_points:
        await vector_engine.index_data_points("Triplet", "text", new_points)
        await vector_engine.delete_data_points("Triplet_text", migrated_old_ids)


async def _migrate_ledger(id_map: dict, dataset_id) -> None:
    """Repoint the relational delete-ledger rows from old node ids to new ones.

    Updates ``nodes.slug`` and ``edges.source_node_id`` / ``destination_node_id``
    — scoped to ``dataset_id`` when given (access control on: one database pair
    per dataset), unscoped when ``None`` (global mode: one graph backs every
    dataset's ledger rows, and old ids are deterministic, so the update is
    correct across all of them) — plus the legacy ``graph_relationship_ledger``
    (always unscoped — it has no dataset column). The PK of a ``nodes`` row is
    left as-is: it is an internal surrogate and the delete path keys on
    ``slug``, which we update.
    """
    if not id_map:
        return

    # Lazy imports: these relational models pull in the ORM Base and would risk
    # an import cycle if loaded when the migration module is first imported.
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger
    from cognee.modules.graph.models import Edge, Node

    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        for old_id, new_id in id_map.items():
            old_uuid, new_uuid = UUID(old_id), UUID(new_id)

            node_stmt = update(Node).where(Node.slug == old_uuid)
            source_stmt = update(Edge).where(Edge.source_node_id == old_uuid)
            target_stmt = update(Edge).where(Edge.destination_node_id == old_uuid)
            if dataset_id is not None:
                node_stmt = node_stmt.where(Node.dataset_id == dataset_id)
                source_stmt = source_stmt.where(Edge.dataset_id == dataset_id)
                target_stmt = target_stmt.where(Edge.dataset_id == dataset_id)

            await session.execute(node_stmt.values(slug=new_uuid))
            await session.execute(source_stmt.values(source_node_id=new_uuid))
            await session.execute(target_stmt.values(destination_node_id=new_uuid))

            await session.execute(
                update(GraphRelationshipLedger)
                .where(GraphRelationshipLedger.source_node_id == old_uuid)
                .values(source_node_id=new_uuid)
            )
            await session.execute(
                update(GraphRelationshipLedger)
                .where(GraphRelationshipLedger.destination_node_id == old_uuid)
                .values(destination_node_id=new_uuid)
            )

        await session.commit()


async def _migrate_graph(graph_engine, id_map: dict, properties_by_id: dict, edges: list) -> int:
    """Remap old-scheme node ids (and their edges) in the graph database."""
    # 1) Create the remapped nodes (new IDs, all original properties preserved).
    new_nodes = [
        _make_node({**properties_by_id[old_id], "id": new_id}) for old_id, new_id in id_map.items()
    ]
    await graph_engine.add_nodes(new_nodes)

    # 2) Re-create every edge touching a remapped node onto the new endpoints.
    #    Edges between two unchanged nodes are left as-is.
    remapped_edges = []
    for source_id, target_id, relationship_name, edge_properties in edges:
        # Ladybug's get_graph_data fabricates (id, id, "SELF") placeholder
        # edges for an edgeless graph; persisting them would write fake
        # relationships into the database.
        if relationship_name == "SELF" and source_id == target_id:
            continue
        new_source = id_map.get(source_id, source_id)
        new_target = id_map.get(target_id, target_id)
        if new_source != source_id or new_target != target_id:
            new_properties = dict(edge_properties or {})
            # cognify embeds the endpoint ids INSIDE edge properties
            # (expand_with_nodes_and_edges), and retrieval prefers those over
            # the actual topology — they must move with the endpoints.
            if "source_node_id" in new_properties:
                new_properties["source_node_id"] = new_source
            if "target_node_id" in new_properties:
                new_properties["target_node_id"] = new_target
            remapped_edges.append((new_source, new_target, relationship_name, new_properties))
    if remapped_edges:
        await graph_engine.add_edges(remapped_edges)

    # 3) Delete the old nodes. A detach-delete also drops their stale edges,
    #    which we already re-created against the new node IDs above.
    await graph_engine.delete_nodes(list(id_map.keys()))

    return len(remapped_edges)


async def migrate(context: MigrationContext) -> None:
    """Remap old-scheme Entity/EntityType ids across graph, vector and ledger."""
    graph_engine = context.graph_engine
    nodes, edges = await graph_engine.get_graph_data()
    if not nodes:
        return

    id_map = build_id_remap(nodes)
    if not id_map:
        logger.info("Entity/EntityType ID migration: no old-scheme nodes found.")
        return

    properties_by_id = {node_id: properties for node_id, properties in nodes}

    # Re-key the derived stores first (while the graph still holds the old ids),
    # rename the graph last. See module docstring for why this order is retry-safe.
    failed = await _migrate_vector(context.vector_engine, id_map, properties_by_id)
    if failed:
        for old_id in failed:
            id_map.pop(old_id, None)
        logger.warning(
            "Entity/EntityType ID migration: %d node(s) left on the old scheme "
            "(vector point could not be re-keyed); they remain consistent across stores.",
            len(failed),
        )
    if not id_map:
        return

    await _migrate_triplet_vector(context.vector_engine, _build_triplet_remap(edges, id_map))
    await _migrate_ledger(id_map, context.dataset_id)
    remapped_edge_count = await _migrate_graph(graph_engine, id_map, properties_by_id, edges)

    logger.info(
        "Entity/EntityType ID migration: remapped %d node(s) and %d edge(s).",
        len(id_map),
        remapped_edge_count,
    )
