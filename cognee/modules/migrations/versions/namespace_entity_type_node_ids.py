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
    store operation. Migration-local by design (``_rekey_native`` below) —
    the VectorDBInterface is deliberately not extended for this.
  * generic (every other backend) — the point is re-inserted through
    ``index_data_points`` under the new id, which re-embeds the (unchanged)
    text; the old point is then deleted. Needs the embedding backend available,
    exactly as ``cognify`` does.

How the new ID is recovered
---------------------------
Both the historical ``generate_node_id`` and the model's identity normalization
lower-case, strip apostrophes and turn spaces into underscores, and
``generate_node_name`` lower-cases and strips apostrophes. So the stored
``name`` property is enough to recompute the node's ID under any scheme — no
need to reverse the UUID hash::

    uuid5(OID, f"Entity:{norm(name)}")   -> the new, model-owned ID
    uuid5(OID, norm(name))               -> the released "bare" ID
    uuid5(OID, norm(f"entity:{name}"))   -> the interim ID (never released)

A node is only remapped when its current ID matches one of the recognized old
schemes for its kind; anything else is left untouched (already migrated, or an
ID not derived from its name).

FROZEN derivations
------------------
This migration vendors its own copies of every id derivation, normalization
rule and collection name it depends on (``_frozen_*`` below, ``_INDEX_FIELDS``)
instead of importing the live model code. A migration's revision must identify
one deterministic transformation forever; if ``Entity.id_for`` or
``identity_fields`` evolve later, that is a NEW migration appended to the
chain — it must never silently change what this one means. Do not "deduplicate"
these copies against the live functions. (The ``_RekeyedPoint`` carrier is the
deliberate exception: it rides the LIVE ``index_data_points`` write path so
re-inserted rows always match what current adapters store.)

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
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions._vector_rekey import (
    RekeyedPoint as _RekeyedPoint,
    lancedb_where as _lancedb_where,
    rekey_lancedb as _rekey_lancedb,
    rekey_native as _rekey_native,
    rekey_pgvector as _rekey_pgvector,
)

logger = logging.getLogger(__name__)

# Stored DataPoint ``type`` property -> interim per-kind prefix this branch
# briefly used before the model-owned scheme. The kind string itself is the
# class-name namespace of the new scheme.
_NODE_KINDS = {
    "Entity": "entity",
    "EntityType": "type",
}

# FROZEN: vector collections per kind at the time this migration shipped
# (collection name = f"{kind}_{field}"). Do not derive from live metadata.
_INDEX_FIELDS = {
    "Entity": ["name"],
    "EntityType": ["name"],
}
_TRIPLET_COLLECTION = "Triplet_text"


def _frozen_normalize(value: str) -> str:
    """FROZEN copy of the normalization shared by every scheme this migration
    touches (``generate_node_id`` and ``DataPoint._normalize_identity_value``
    as of cognee 1.2.0): lower-case, spaces to underscores, strip apostrophes."""
    return value.lower().replace(" ", "_").replace("'", "")


def _frozen_bare_id(text: str) -> str:
    """FROZEN: the released pre-#2515 scheme (``generate_node_id`` as of 1.1.x).

    Also the Triplet point-id derivation (``generate_node_id(source + rel +
    target)``), which ``add_data_points`` used verbatim.
    """
    return str(uuid5(NAMESPACE_OID, _frozen_normalize(text)))


def _frozen_model_id(kind: str, name: str) -> str:
    """FROZEN: the model-owned target scheme this migration maps TO
    (``DataPoint.id_for`` as of cognee 1.2.0): class-name-namespaced uuid5."""
    return str(uuid5(NAMESPACE_OID, f"{kind}:{_frozen_normalize(name)}"))


def _frozen_edge_pk(
    tenant_id, user_id, dataset_id, source_id, relationship_name: str, target_id
) -> UUID:
    """FROZEN copy of the ``edges`` PK derivation in ``upsert_edges`` (1.2.0):
    ``uuid5(OID, tenant+user+dataset+source+relationship_name+target)``, all
    str()-joined with no separators. We recompute it onto the new endpoints so
    the next cognify's ``on_conflict_do_nothing(id)`` upsert recognizes the row
    instead of inserting a duplicate edge. (f-strings render None/UUID like
    str(), matching the write path for both single- and multi-tenant.)
    """
    return uuid5(
        NAMESPACE_OID,
        f"{tenant_id}{user_id}{dataset_id}{source_id}{relationship_name}{target_id}",
    )


def _frozen_node_pk(tenant_id, user_id, dataset_id, data_id, slug) -> UUID:
    """FROZEN copy of the ``nodes`` PK derivation in ``upsert_nodes`` (1.2.0):
    ``uuid5(OID, tenant+user+dataset+data_id+node.id)``, where ``node.id`` is the
    row's ``slug``. Recomputed onto the new slug so the next cognify dedupes on
    the PK instead of inserting a duplicate provenance row for (slug, data_id).
    """
    return uuid5(
        NAMESPACE_OID,
        f"{tenant_id}{user_id}{dataset_id}{data_id}{slug}",
    )


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
        node_type = properties.get("type")
        interim_prefix = _NODE_KINDS.get(node_type)
        if interim_prefix is None:
            # Only Entity / EntityType IDs changed.
            continue

        name = properties.get("name")
        if not name:
            continue

        new_id = _frozen_model_id(node_type, name)
        if node_id == new_id:
            # Already on the model-owned scheme.
            continue

        # Recognized historical schemes this node could still be on.
        recognized_old_ids = {
            _frozen_bare_id(name),  # released "bare" scheme
            _frozen_bare_id(f"{interim_prefix}:{name}"),  # interim, never released
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
        index_fields = _INDEX_FIELDS.get(node_type)
        if index_fields is None:
            continue

        for index_field in index_fields:
            collection = f"{node_type}_{index_field}"

            # Vector-preserving fast path; nothing can fail per-point here
            # (no text needed), so no ids join the failed set.
            if await _rekey_native(
                vector_engine, collection, {old_id: id_map[old_id] for old_id in old_ids}
            ):
                continue

            from cognee.infrastructure.databases.vector.exceptions import (
                CollectionNotFoundError,
            )

            try:
                rows = await vector_engine.retrieve(collection, [str(o) for o in old_ids])
            except CollectionNotFoundError:
                # Graph-only deployment / collection never created: no points
                # to move for this kind. Same tolerance the triplet path has.
                continue

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
        old_triplet_id = _frozen_bare_id(str(source_id) + relationship_name + str(target_id))
        new_triplet_id = _frozen_bare_id(str(new_source) + relationship_name + str(new_target))
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
    if await _rekey_native(vector_engine, _TRIPLET_COLLECTION, triplet_map):
        return

    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        rows = await vector_engine.retrieve(_TRIPLET_COLLECTION, list(triplet_map.keys()))
    except CollectionNotFoundError:
        # Triplet embedding was never enabled for this database.
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
        await vector_engine.index_data_points("Triplet", "text", new_points)  # -> Triplet_text
        await vector_engine.delete_data_points("Triplet_text", migrated_old_ids)


async def _tenant_by_dataset(session, dataset_ids: set) -> dict:
    """``{dataset_id: tenant_id}`` for recomputing ledger PKs.

    The ledger rows don't store the ``tenant_id`` their PK is hashed from, so we
    recover it from the owning DATASET (``create_dataset`` stamps
    ``Dataset.tenant_id = user.tenant_id``, so it matches what ``upsert_*``
    hashed and stays bound to the data even if the user later changes tenant). A
    missing dataset is left out of the dict, and the caller leaves that PK
    untouched — self-validation never uses a tenant that can't reproduce the id.
    """
    if not dataset_ids:
        return {}
    from sqlalchemy import select

    from cognee.modules.data.models import Dataset

    rows = (
        await session.execute(
            select(Dataset.id, Dataset.tenant_id).where(Dataset.id.in_(dataset_ids))
        )
    ).all()
    return {dataset_id: tenant_id for dataset_id, tenant_id in rows}


async def _recompute_ledger_pks(
    session, dataset_id, *, model, affected_stmt, migrated_values, old_pk, new_pk, kind: str
) -> None:
    """Move remapped ledger rows onto a PK recomputed from their new identity.

    Shared by the nodes and edges ledgers: both embed a remapped id in their PK
    (``_frozen_node_pk`` / ``_frozen_edge_pk``), so leaving it stale lets the next
    cognify insert a duplicate row. The per-model parts are passed in; the
    algorithm is identical:

    - self-validating: rewrite the PK only when the stored id is exactly
      reproducible from the row. If the tenant is unrecoverable, still update the
      migrated columns (delete-ledger correctness) but leave the PK — never guess.
    - merge-safe: if the target id already exists (an SDK wrote new-scheme first,
      or two rows collapse into one), drop this stale row instead of colliding.

    ``migrated_values(row)`` returns the non-id columns to update (always
    applied); ``old_pk(row, tenant)`` / ``new_pk(row, tenant, migrated)`` derive
    the stored and target PKs.
    """
    from sqlalchemy import delete, select, update

    scope_stmt = select(model.id)
    if dataset_id is not None:
        affected_stmt = affected_stmt.where(model.dataset_id == dataset_id)
        scope_stmt = scope_stmt.where(model.dataset_id == dataset_id)

    affected = (await session.execute(affected_stmt)).all()
    if not affected:
        return

    # Stable ids = every in-scope id EXCEPT the rows we are about to remap away
    # (their old ids are being vacated, so they must not count as collision
    # targets). New ids are added to this set as we assign them.
    affected_old_ids = {row.id for row in affected}
    existing_ids = {row_id for (row_id,) in (await session.execute(scope_stmt)).all()}
    existing_ids -= affected_old_ids

    tenant_by_dataset = await _tenant_by_dataset(session, {row.dataset_id for row in affected})

    unrecoverable = 0
    merged = 0
    for row in affected:
        migrated = migrated_values(row)

        new_id = row.id  # default: leave the PK if we cannot prove the new one
        if row.dataset_id in tenant_by_dataset:
            tenant_id = tenant_by_dataset[row.dataset_id]
            if old_pk(row, tenant_id) == row.id:
                new_id = new_pk(row, tenant_id, migrated)
            else:
                unrecoverable += 1
        else:
            unrecoverable += 1

        if new_id != row.id and new_id in existing_ids:
            # The new-scheme row already exists — drop this stale duplicate so
            # the next cognify finds exactly one row for the logical identity.
            await session.execute(delete(model).where(model.id == row.id))
            merged += 1
            continue

        await session.execute(update(model).where(model.id == row.id).values(id=new_id, **migrated))
        existing_ids.add(new_id)

    if unrecoverable:
        logger.warning(
            "%s-ledger migration: kept the original id on %d row(s) whose tenant could not be "
            "recovered (columns still migrated). A later cognify may insert a duplicate row for "
            "these; they remain delete-correct.",
            kind,
            unrecoverable,
        )
    if merged:
        logger.info(
            "%s-ledger migration: merged %d already-migrated duplicate row(s).", kind, merged
        )


async def _migrate_ledger_edges(session, remap: dict, dataset_id) -> None:
    """Recompute ``edges`` PKs onto their remapped endpoints (see ``_frozen_edge_pk``)."""
    from sqlalchemy import or_, select

    from cognee.modules.graph.models import Edge

    old_uuids = {UUID(old) for old in remap}
    affected_stmt = select(
        Edge.id,
        Edge.user_id,
        Edge.dataset_id,
        Edge.source_node_id,
        Edge.destination_node_id,
        Edge.relationship_name,
    ).where(or_(Edge.source_node_id.in_(old_uuids), Edge.destination_node_id.in_(old_uuids)))

    def migrated(row):
        new_source = remap.get(str(row.source_node_id))
        new_target = remap.get(str(row.destination_node_id))
        return {
            "source_node_id": UUID(new_source) if new_source else row.source_node_id,
            "destination_node_id": UUID(new_target) if new_target else row.destination_node_id,
        }

    await _recompute_ledger_pks(
        session,
        dataset_id,
        model=Edge,
        affected_stmt=affected_stmt,
        migrated_values=migrated,
        old_pk=lambda row, tenant: _frozen_edge_pk(
            tenant,
            row.user_id,
            row.dataset_id,
            row.source_node_id,
            row.relationship_name,
            row.destination_node_id,
        ),
        new_pk=lambda row, tenant, m: _frozen_edge_pk(
            tenant,
            row.user_id,
            row.dataset_id,
            m["source_node_id"],
            row.relationship_name,
            m["destination_node_id"],
        ),
        kind="Edge",
    )


async def _migrate_ledger_nodes(session, remap: dict, dataset_id) -> None:
    """Recompute ``nodes`` PKs onto their remapped slug (see ``_frozen_node_pk``)."""
    from sqlalchemy import select

    from cognee.modules.graph.models import Node

    old_uuids = {UUID(old) for old in remap}
    affected_stmt = select(Node.id, Node.user_id, Node.dataset_id, Node.data_id, Node.slug).where(
        Node.slug.in_(old_uuids)
    )

    await _recompute_ledger_pks(
        session,
        dataset_id,
        model=Node,
        affected_stmt=affected_stmt,
        migrated_values=lambda row: {"slug": UUID(remap[str(row.slug)])},
        old_pk=lambda row, tenant: _frozen_node_pk(
            tenant, row.user_id, row.dataset_id, row.data_id, row.slug
        ),
        new_pk=lambda row, tenant, m: _frozen_node_pk(
            tenant, row.user_id, row.dataset_id, row.data_id, m["slug"]
        ),
        kind="Node",
    )


async def _migrate_ledger(id_map: dict, dataset_id) -> None:
    """Repoint the relational delete-ledger from old node ids to new ones.

    Rewrites ``nodes`` (slug + PK) and ``edges`` (endpoints + PK) — both PKs
    embed a node id this migration moves, so leaving them stale would let the
    next cognify insert duplicate rows. Scoped to ``dataset_id`` when given,
    unscoped in global mode (one graph backs every dataset; old ids are
    deterministic). The legacy ``graph_relationship_ledger`` is always unscoped
    (no dataset column) and only its endpoints move — its PK is a random
    timestamp, not endpoint-derived.
    """
    if not id_map:
        return

    # Lazy imports: these relational models pull in the ORM Base and would risk
    # an import cycle if loaded when the migration module is first imported.
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.graph.legacy.GraphRelationshipLedger import GraphRelationshipLedger

    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        await _migrate_ledger_nodes(session, id_map, dataset_id)

        # The legacy ledger's PK is a random timestamp uuid5 (not endpoint-
        # derived), so only its endpoints move; bulk update per remapped id.
        for old_id, new_id in id_map.items():
            old_uuid, new_uuid = UUID(old_id), UUID(new_id)
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

        await _migrate_ledger_edges(session, id_map, dataset_id)

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


def _is_hybrid_backend() -> bool:
    """True when graph and vector live in ONE store (e.g. Neptune Analytics).

    On hybrid backends the vector "points" are graph nodes sharing the entity's
    id; this migration's separate vector/graph steps would corrupt them (the
    vector delete is a DETACH DELETE of the live graph node). Refuse rather
    than corrupt — uses the same detection the unified engine factory uses.
    """
    from cognee.infrastructure.databases.graph.config import get_graph_context_config
    from cognee.infrastructure.databases.unified.get_unified_engine import _is_hybrid_provider
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config

    return _is_hybrid_provider(get_graph_context_config(), get_vectordb_context_config())


def build_id_remap_reverse(nodes: list) -> dict:
    """``{model_owned_id: released_bare_id}`` for nodes on the NEW scheme.

    Inverse of :func:`build_id_remap`, targeting the RELEASED bare scheme (the
    interim scheme was never released, so it is never a downgrade target).
    Nodes not on the model-owned scheme are left untouched. NOTE: the bare
    scheme is the one with the Entity/EntityType name collision — downgrading
    a graph holding both ``Entity("x")`` and ``EntityType("x")`` faithfully
    reproduces that collision (two map entries to one id); the merge-safe
    re-key and graph upsert collapse them, exactly as old cognee stored them.
    """
    id_map: dict = {}
    for node_id, properties in nodes:
        node_type = properties.get("type")
        if node_type not in _NODE_KINDS:
            continue
        name = properties.get("name")
        if not name:
            continue
        if node_id != _frozen_model_id(node_type, name):
            continue
        id_map[node_id] = _frozen_bare_id(name)
    return id_map


async def _apply_id_map(context: MigrationContext, nodes: list, edges: list, id_map: dict) -> None:
    """Apply an id remap across all three stores, derived stores first.

    Direction-agnostic core shared by :func:`migrate` (old -> model-owned) and
    :func:`downgrade` (model-owned -> bare). The map must be computed while the
    graph still holds the SOURCE ids; the graph is renamed last so a crash at
    any point leaves a state the next run finishes (see module docstring).
    """
    properties_by_id = {node_id: properties for node_id, properties in nodes}

    failed = await _migrate_vector(context.vector_engine, id_map, properties_by_id)
    if failed:
        for old_id in failed:
            id_map.pop(old_id, None)
        logger.warning(
            "Entity/EntityType ID remap: %d node(s) left on their current scheme "
            "(vector point could not be re-keyed); they remain consistent across stores.",
            len(failed),
        )
    if not id_map:
        return

    await _migrate_triplet_vector(context.vector_engine, _build_triplet_remap(edges, id_map))
    await _migrate_ledger(id_map, context.dataset_id)
    remapped_edge_count = await _migrate_graph(
        context.graph_engine, id_map, properties_by_id, edges
    )

    logger.info(
        "Entity/EntityType ID remap: remapped %d node(s) and %d edge(s).",
        len(id_map),
        remapped_edge_count,
    )


async def migrate(context: MigrationContext) -> None:
    """Remap old-scheme Entity/EntityType ids across graph, vector and ledger."""
    if _is_hybrid_backend():
        logger.warning(
            "Entity/EntityType ID migration does not support hybrid graph+vector "
            "backends; skipping. The database keeps its current id scheme."
        )
        return

    nodes, edges = await context.graph_engine.get_graph_data()
    if not nodes:
        return

    id_map = build_id_remap(nodes)
    if not id_map:
        logger.info("Entity/EntityType ID migration: no old-scheme nodes found.")
        return

    await _apply_id_map(context, nodes, edges, id_map)


async def downgrade(context: MigrationContext) -> None:
    """Reverse :func:`migrate`: model-owned ids back to the released bare scheme.

    Same three-store lockstep and retry-safety as the upgrade, with the map
    inverted. Only useful when rolling back to a pre-#2515 cognee release —
    the bare scheme reintroduces the Entity/EntityType name collision that
    release had (see :func:`build_id_remap_reverse`).
    """
    if _is_hybrid_backend():
        logger.warning(
            "Entity/EntityType ID migration does not support hybrid graph+vector "
            "backends; skipping downgrade. The database keeps its current id scheme."
        )
        return

    nodes, edges = await context.graph_engine.get_graph_data()
    if not nodes:
        return

    id_map = build_id_remap_reverse(nodes)
    if not id_map:
        logger.info("Entity/EntityType ID downgrade: no model-owned-scheme nodes found.")
        return

    await _apply_id_map(context, nodes, edges, id_map)
