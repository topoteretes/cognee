"""The ordered chain of cognee data migrations (graph + vector + relational ledger).

ONE chain for all stores: a migration is a cross-store transformation (the id
remap rewrites graph nodes, vector points and ledger rows in lockstep), so
splitting chains per database would only force an arbitrary registration
choice. Each migration receives the full :class:`MigrationContext`.

To add a migration, implement it in its own module under ``versions/`` and
APPEND a ``Migration`` here whose ``down_revision`` is the previous entry's
slug. Existing entries are the permanent, immutable history of the chain —
never rename, remove, or reorder them: every deployed database stores the slug
of the last migration it ran, and an unknown stored slug disables the chain
for that database (see ``pending_migrations``). Keep migrations idempotent and
cheap on empty stores; see README.md in this package for the full contract.
"""

from cognee.modules.migrations.migration import Migration, order_migrations
from cognee.modules.migrations.versions.ladybug_graph_provenance_columns import (
    downgrade as ladybug_graph_provenance_columns_down,
)
from cognee.modules.migrations.versions.ladybug_graph_provenance_columns import (
    migrate as ladybug_graph_provenance_columns,
)
from cognee.modules.migrations.versions.namespace_edge_type_point_ids import (
    downgrade as namespace_edge_type_point_ids_down,
)
from cognee.modules.migrations.versions.namespace_edge_type_point_ids import (
    migrate as namespace_edge_type_point_ids,
)
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
    downgrade as namespace_entity_type_node_ids_down,
)
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
    migrate as namespace_entity_type_node_ids,
)
from cognee.modules.migrations.versions.pgvector_hnsw_indexes import (
    downgrade as pgvector_hnsw_indexes_down,
)
from cognee.modules.migrations.versions.pgvector_hnsw_indexes import (
    migrate as pgvector_hnsw_indexes,
)
from cognee.modules.migrations.versions.postgres_graph_provenance_columns import (
    downgrade as postgres_graph_provenance_columns_down,
)
from cognee.modules.migrations.versions.postgres_graph_provenance_columns import (
    migrate as postgres_graph_provenance_columns,
)
from cognee.modules.migrations.versions.postgres_graph_source_ref_tables import (
    downgrade as postgres_graph_source_ref_tables_down,
)
from cognee.modules.migrations.versions.postgres_graph_source_ref_tables import (
    migrate as postgres_graph_source_ref_tables,
)
from cognee.modules.migrations.versions.reembed_local_default_embedding_model import (
    downgrade as reembed_local_default_embedding_model_down,
)
from cognee.modules.migrations.versions.reembed_local_default_embedding_model import (
    migrate as reembed_local_default_embedding_model,
)
from cognee.modules.migrations.versions.rekey_fork_document_ids import (
    downgrade as rekey_fork_document_ids_down,
)
from cognee.modules.migrations.versions.rekey_fork_document_ids import (
    migrate as rekey_fork_document_ids,
)

# The vector adapter's storage-schema sync (e.g. LanceDB adding columns) is NOT
# in this chain: a chain entry runs once per database, but that sync must run on
# every Cognee version change, even a release with no data migration. The runner
# triggers it on a cognee_version mismatch, after the chain (see
# runner._sync_vector_adapter_storage).

MIGRATIONS: list[Migration] = [
    # PR #2515: Entity/EntityType node IDs gained "Entity:" / "EntityType:"
    # namespacing so Entity("x") and EntityType("x") stop colliding on one
    # UUID. Remaps graph nodes, vector points, triplet points and ledger rows.
    Migration(
        slug="namespace_entity_type_node_ids",
        cognee_version="1.2.0",
        up=namespace_entity_type_node_ids,
        down_revision=None,
        down=namespace_entity_type_node_ids_down,
    ),
    # EdgeType vector points moved from the bare hand-rolled uuid5 to the
    # DataPoint identity derivation ("EdgeType:<name>"), unifying the last
    # hand-rolled id with the single DataPoint mechanism.
    Migration(
        slug="namespace_edge_type_point_ids",
        cognee_version="1.2.0",
        up=namespace_edge_type_point_ids,
        down_revision="namespace_entity_type_node_ids",
        down=namespace_edge_type_point_ids_down,
    ),
    # COG-5522: the Postgres graph adapter gained four varchar[] provenance columns
    # (+ GIN indexes) on graph_node/graph_edge for graph-native delete/rollback.
    # Fresh graphs get them from create_all; this backfills graph_node/graph_edge
    # tables left over from a pre-provenance release. No-op on every non-Postgres
    # graph backend.
    Migration(
        slug="postgres_graph_provenance_columns",
        cognee_version="1.2.2",
        up=postgres_graph_provenance_columns,
        down_revision="namespace_edge_type_point_ids",
        down=postgres_graph_provenance_columns_down,
    ),
    # COG-5522: same backfill for the Ladybug/Kuzu backend. Fresh databases get
    # the provenance columns from _ensure_schema's CREATE ... IF NOT EXISTS,
    # which no-ops on tables a pre-provenance release already created — an
    # EMPTY legacy graph then gets marked graph-provenance on first write and
    # every stamped write dies on the missing columns (caught by the
    # backwards-compatibility CI). No-op on every non-Ladybug graph backend.
    Migration(
        slug="ladybug_graph_provenance_columns",
        cognee_version="1.5.0",
        up=ladybug_graph_provenance_columns,
        down_revision="postgres_graph_provenance_columns",
        down=ladybug_graph_provenance_columns_down,
    ),
    # Dataset-scoping backfill split pre-refactor shared Data rows; fork
    # datasets' graph document nodes still carried the pre-fork id. Re-keys
    # them to the canonical relational id (ledger references included), so no
    # graph/vector path needs legacy awareness afterwards.
    Migration(
        slug="rekey_fork_document_ids",
        cognee_version="1.5.0",
        up=rekey_fork_document_ids,
        down_revision="ladybug_graph_provenance_columns",
        down=rekey_fork_document_ids_down,
    ),
    # pgvector collections were created without a vector index, so every search was a
    # full scan of every stored vector. Builds the HNSW index new collections now get.
    Migration(
        slug="pgvector_hnsw_indexes",
        cognee_version="1.6.6",
        up=pgvector_hnsw_indexes,
        down_revision="rekey_fork_document_ids",
        down=pgvector_hnsw_indexes_down,
    ),
    # The Postgres graph kept the source refs of a node or edge in four array columns,
    # so a node shared by many documents had its row rewritten on every attach. Moves
    # them into graph_node_source_ref / graph_edge_source_ref, one row per ref, in
    # batches. No-op on every graph that is not stored in these Postgres tables.
    Migration(
        slug="postgres_graph_source_ref_tables",
        cognee_version="1.6.9",
        up=postgres_graph_source_ref_tables,
        down_revision="pgvector_hnsw_indexes",
        down=postgres_graph_source_ref_tables_down,
    ),
    # The default local embedding model (no embedding setting, no LLM key) changed from
    # BAAI/bge-small-en-v1.5 to perplexity-ai/pplx-embed-v1-0.6b, whose vectors have
    # another width. Re-embeds the datasets the previous default built. No-op wherever
    # an embedding model is configured, and for every dataset another model built.
    Migration(
        slug="reembed_local_default_embedding_model",
        cognee_version="1.6.11",
        up=reembed_local_default_embedding_model,
        down_revision="postgres_graph_source_ref_tables",
        down=reembed_local_default_embedding_model_down,
    ),
]


def _validate_registry() -> None:
    """Invariants enforced at import: linear chain (order_migrations), unique
    slugs, and no slug shadowing the reserved revision keywords."""
    order_migrations(MIGRATIONS)
    slugs = [migration.slug for migration in MIGRATIONS]
    duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
    if duplicates:
        raise ValueError(f"Duplicate migration slugs: {sorted(duplicates)}")
    reserved = {"head", "base"}
    shadowed = set(slugs) & reserved
    if shadowed:
        raise ValueError(f"Migration slugs shadow reserved revision keywords: {sorted(shadowed)}")


_validate_registry()
