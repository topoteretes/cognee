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

Only the portable :class:`GraphDBInterface` operations are used
(``get_graph_data``/``add_nodes``/``add_edges``/``delete_nodes``). Re-added
nodes are wrapped by per-class carriers (``_make_node``) that satisfy every
adapter's ``add_nodes`` contract (``model_dump`` for Ladybug/Postgres;
iteration + correct class name for Neo4j's label).

Caveats:
- ``get_graph_data`` does not return ``created_at``/``updated_at``, so a
  remapped node's timestamps are reset to the migration time. Semantic
  properties are preserved; original timestamps are not recoverable here.
- Verified end-to-end on Ladybug/Kuzu and (relationally) Postgres. Neo4j and
  Neptune use the same interface and the carrier targets their contracts, but
  are not yet exercised by an end-to-end test.
"""

import logging

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models import Entity, EntityType

logger = logging.getLogger(__name__)

# Stored DataPoint ``type`` property -> (model class that owns the new id via
# ``id_for``, interim per-kind prefix this branch briefly used before id_for).
_NODE_KINDS = {
    "Entity": (Entity, "entity"),
    "EntityType": (EntityType, "type"),
}


# Per-class carrier types so a re-added node satisfies EVERY adapter's add_nodes
# contract, not just one. Adapters read incoming nodes differently:
#   * Ladybug / Postgres call ``node.model_dump()`` (or ``vars()``).
#   * Neo4j calls ``dict(node)`` (requires the object to be iterable) and uses
#     ``type(node).__name__`` as the node's label.
# A plain object only satisfies the first; a Neo4j carrier must also be iterable
# AND carry the real class name (``Entity``/``EntityType``). We build one tiny
# class per node-type name, cached, so ``type(carrier).__name__`` is correct.
_carrier_classes: dict = {}


def _make_node(properties: dict):
    """Return a node carrier (id + all properties) accepted by any adapter."""
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


async def migrate(graph_engine) -> None:
    """Remap old-scheme Entity/EntityType node IDs (and their edges) in place."""
    nodes, edges = await graph_engine.get_graph_data()
    if not nodes:
        return

    id_map = build_id_remap(nodes)
    if not id_map:
        logger.info("Entity/EntityType ID migration: no old-scheme nodes found.")
        return

    properties_by_id = {node_id: properties for node_id, properties in nodes}

    # 1) Create the remapped nodes (new IDs, all original properties preserved).
    new_nodes = [
        _make_node({**properties_by_id[old_id], "id": new_id}) for old_id, new_id in id_map.items()
    ]
    await graph_engine.add_nodes(new_nodes)

    # 2) Re-create every edge touching a remapped node onto the new endpoints.
    #    Edges between two unchanged nodes are left as-is.
    remapped_edges = []
    for source_id, target_id, relationship_name, edge_properties in edges:
        new_source = id_map.get(source_id, source_id)
        new_target = id_map.get(target_id, target_id)
        if new_source != source_id or new_target != target_id:
            remapped_edges.append(
                (new_source, new_target, relationship_name, edge_properties or {})
            )
    if remapped_edges:
        await graph_engine.add_edges(remapped_edges)

    # 3) Delete the old nodes. A detach-delete also drops their stale edges,
    #    which we already re-created against the new node IDs above.
    await graph_engine.delete_nodes(list(id_map.keys()))

    logger.info(
        "Entity/EntityType ID migration: remapped %d node(s) and %d edge(s).",
        len(id_map),
        len(remapped_edges),
    )
