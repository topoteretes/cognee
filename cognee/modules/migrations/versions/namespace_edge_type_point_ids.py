"""Migration: namespace EdgeType vector-point ids by the model class.

``EdgeType`` points (the ``EdgeType_relationship_name`` collection, one point
per distinct edge retrieval text, used by triplet search to score edges and by
the delete flows to drop orphaned types) were keyed by the BARE derivation
``uuid5(NAMESPACE_OID, normalized_text)`` — hand-rolled in the late
``generate_edge_id`` instead of the DataPoint identity mechanism. EdgeType now
declares ``identity_fields`` like every identity-bearing model, so its ids are
namespaced (``uuid5(NAMESPACE_OID, "EdgeType:" + normalized_text)``) and every
lookup site computes them through ``EdgeType.id_for``. This migration moves
existing points to the new ids.

Vector-only: EdgeType never becomes a graph node and is absent from the
relational ledger, so there is no cross-store ordering to worry about. The
source of truth for which points exist is the GRAPH's edges (mirroring how
``index_graph_edges`` creates the points), so points whose edges were deleted
out-of-band are left behind — same graph-driven limitation as the entity
migration, and harmless (orphaned points were already unreachable by id).

Everything derived is FROZEN here (see the entity migration's preamble for
why); the live model must never be imported into this module.
"""

import logging
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.modules.migrations.migration import MigrationContext
from cognee.modules.migrations.versions._vector_rekey import (
    RekeyedPoint,
    rekey_native,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FROZEN derivations (this migration's permanent contract — never edit)
# ---------------------------------------------------------------------------

_COLLECTION_KIND = "EdgeType"
_INDEX_FIELD = "relationship_name"
_COLLECTION = f"{_COLLECTION_KIND}_{_INDEX_FIELD}"


def _frozen_normalize(value: str) -> str:
    """The released normalization (lower, spaces->_, strip apostrophes)."""
    return value.lower().replace(" ", "_").replace("'", "")


def _frozen_bare_id(text: str) -> UUID:
    """OLD scheme: ``generate_edge_id`` as released — bare hash of the text."""
    return uuid5(NAMESPACE_OID, _frozen_normalize(text))


def _frozen_model_id(text: str) -> UUID:
    """NEW scheme: DataPoint identity derivation for EdgeType, as shipped."""
    return uuid5(NAMESPACE_OID, f"EdgeType:{_frozen_normalize(text)}")


def _frozen_edge_text(relationship_name, properties: dict) -> str:
    """The released ``get_edge_retrieval_text`` rule: nonblank ``edge_text``
    from the edge's properties, else nonblank relationship name, else ""."""

    def nonblank(value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return nonblank((properties or {}).get("edge_text")) or nonblank(relationship_name) or ""


def _edge_texts(edges: list) -> list[str]:
    """Distinct edge retrieval texts present in the graph (point source set)."""
    texts = []
    seen = set()
    for source_id, target_id, relationship_name, properties in edges:
        if relationship_name == "SELF" and str(source_id) == str(target_id):
            continue  # Ladybug's synthetic placeholder edges — never indexed.
        text = _frozen_edge_text(relationship_name, properties)
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def build_id_remap(edges: list) -> dict:
    """``{old_bare_point_id: new_model_point_id}`` (str -> str) per edge text."""
    return {str(_frozen_bare_id(text)): str(_frozen_model_id(text)) for text in _edge_texts(edges)}


def build_id_remap_reverse(edges: list) -> dict:
    """``{new_model_point_id: old_bare_point_id}`` for the downgrade."""
    return {str(_frozen_model_id(text)): str(_frozen_bare_id(text)) for text in _edge_texts(edges)}


def _is_hybrid_backend() -> bool:
    """Same refusal as the entity migration: on hybrid graph+vector backends
    the "points" are graph nodes — these steps would corrupt them."""
    from cognee.infrastructure.databases.graph.config import get_graph_context_config
    from cognee.infrastructure.databases.unified.get_unified_engine import _is_hybrid_provider
    from cognee.infrastructure.databases.vector.config import get_vectordb_context_config

    return _is_hybrid_provider(get_graph_context_config(), get_vectordb_context_config())


async def _apply_map(context: MigrationContext, id_map: dict, text_by_old: dict) -> None:
    """Move every point in ``id_map`` to its new id (direction-agnostic core).

    Native fast path moves stored vectors; the generic fallback re-embeds —
    and unlike the entity migration there is no failed set: the embeddable
    text is always known (the map was built FROM the texts).
    """
    if not id_map or context.vector_engine is None:
        return

    if await rekey_native(context.vector_engine, _COLLECTION, id_map):
        return

    from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

    try:
        rows = await context.vector_engine.retrieve(_COLLECTION, list(id_map))
    except CollectionNotFoundError:
        return  # collection never created (no edges ever indexed) — nothing to move

    new_points = []
    migrated_old_ids = []
    for row in rows:
        old_id = str(row.id)
        new_id = id_map.get(old_id)
        if new_id is None:
            continue
        payload = dict(row.payload or {})
        new_points.append(
            RekeyedPoint(
                id=UUID(new_id),
                text=payload.get("text") or text_by_old[old_id],
                belongs_to_set=payload.get("belongs_to_set") or [],
            )
        )
        migrated_old_ids.append(old_id)

    if new_points:
        await context.vector_engine.index_data_points(_COLLECTION_KIND, _INDEX_FIELD, new_points)
        await context.vector_engine.delete_data_points(_COLLECTION, migrated_old_ids)


async def migrate(context: MigrationContext) -> None:
    """Move EdgeType points from bare to model-namespaced ids."""
    if _is_hybrid_backend():
        logger.warning(
            "EdgeType id migration skipped: hybrid graph+vector backend stores points "
            "as graph nodes; this migration does not support it."
        )
        return

    _, edges = await context.graph_engine.get_graph_data()
    texts = _edge_texts(edges)
    await _apply_map(
        context,
        {str(_frozen_bare_id(t)): str(_frozen_model_id(t)) for t in texts},
        {str(_frozen_bare_id(t)): t for t in texts},
    )
    logger.info("EdgeType point id migration done (%d edge text(s) considered).", len(texts))


async def downgrade(context: MigrationContext) -> None:
    """Move EdgeType points back to the bare released ids."""
    if _is_hybrid_backend():
        logger.warning("EdgeType id downgrade skipped: hybrid backend (see migrate()).")
        return

    _, edges = await context.graph_engine.get_graph_data()
    texts = _edge_texts(edges)
    await _apply_map(
        context,
        {str(_frozen_model_id(t)): str(_frozen_bare_id(t)) for t in texts},
        {str(_frozen_model_id(t)): t for t in texts},
    )
    logger.info("EdgeType point id downgrade done (%d edge text(s) considered).", len(texts))
