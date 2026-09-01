"""Schema definitions for the Postgres graph adapter (graph_node, graph_edge)."""

from sqlalchemy import Table, Column, MetaData, String, DateTime, Index, ForeignKey, func, text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

_meta = MetaData()


def _provenance_columns() -> list:
    """Four graph-provenance columns shared by graph_node and graph_edge.

    Stored as native ``text[]`` arrays (COG-5522 Part 1) so delete/rollback can
    filter by source ref, dataset id, or pipeline run id with an array-membership
    scan. Default ``'{}'`` keeps them non-NULL, so every node/edge always carries
    an (initially empty) provenance vector. See the Ladybug adapter for the
    delimiter-string equivalent; the contract is identical, only the storage
    representation differs (Postgres has first-class arrays, Kuzu does not).
    """
    return [
        Column("source_ref_keys", ARRAY(String), nullable=False, server_default=text("'{}'")),
        Column("source_dataset_ids", ARRAY(String), nullable=False, server_default=text("'{}'")),
        Column("source_run_ids", ARRAY(String), nullable=False, server_default=text("'{}'")),
        Column("source_run_refs", ARRAY(String), nullable=False, server_default=text("'{}'")),
    ]


_node_table = Table(
    "graph_node",
    _meta,
    Column("id", String, primary_key=True),
    Column("name", String),
    Column("type", String),
    Column("properties", JSONB),
    *_provenance_columns(),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), server_default=func.now()),
)

_edge_table = Table(
    "graph_edge",
    _meta,
    Column(
        "source_id",
        String,
        ForeignKey("graph_node.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    ),
    Column(
        "target_id",
        String,
        ForeignKey("graph_node.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    ),
    Column("relationship_name", String, primary_key=True, nullable=False),
    Column("properties", JSONB),
    *_provenance_columns(),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), server_default=func.now()),
)

# Graph-level key/value metadata. Holds the graph-provenance delete-mode marker
# (provenance_version / delete_mode) that routes an empty Postgres graph onto the
# graph-native delete path instead of the relational ledger.
_metadata_table = Table(
    "graph_metadata",
    _meta,
    Column("key", String, primary_key=True),
    Column("value", String, nullable=False),
)

Index("idx_edge_source", _edge_table.c.source_id)
Index("idx_edge_target", _edge_table.c.target_id)
Index("idx_node_type", _node_table.c.type)

# Covering indexes: neighbor lookups without heap reads
Index(
    "idx_edge_source_cover",
    _edge_table.c.source_id,
    postgresql_include=["target_id", "relationship_name"],
)
Index(
    "idx_edge_target_cover",
    _edge_table.c.target_id,
    postgresql_include=["source_id", "relationship_name"],
)

# GIN indexes back the array-membership filters used by graph-provenance delete
# (find_*_by_source_ref / _by_dataset / _by_pipeline_run): `:token = ANY(col)`
# and `col @> ARRAY[:token]` become index scans instead of full table scans.
Index("idx_node_source_ref_keys", _node_table.c.source_ref_keys, postgresql_using="gin")
Index("idx_node_source_dataset_ids", _node_table.c.source_dataset_ids, postgresql_using="gin")
Index("idx_node_source_run_ids", _node_table.c.source_run_ids, postgresql_using="gin")
Index("idx_edge_source_ref_keys", _edge_table.c.source_ref_keys, postgresql_using="gin")
Index("idx_edge_source_dataset_ids", _edge_table.c.source_dataset_ids, postgresql_using="gin")
Index("idx_edge_source_run_ids", _edge_table.c.source_run_ids, postgresql_using="gin")
