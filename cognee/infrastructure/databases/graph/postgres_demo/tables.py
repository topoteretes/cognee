"""Schema definitions for the Postgres graph adapter.

``graph_node`` and ``graph_edge`` hold the graph, ``graph_metadata`` its key/value
marker. Graph provenance, the source refs that say which documents own a node or
an edge, is one row per (node or edge, source ref) in ``graph_node_source_ref``
and ``graph_edge_source_ref``.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    MetaData,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

_meta = MetaData()

_node_table = Table(
    "graph_node",
    _meta,
    Column("id", String, primary_key=True),
    Column("name", String),
    Column("type", String),
    Column("properties", JSONB),
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
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), server_default=func.now()),
)


def _source_ref_columns() -> list:
    """The columns a source-ref row has besides the node or edge it belongs to.

    ``dataset_id`` repeats the dataset id that is part of ``source_ref_key``, so
    deleting a dataset can find its refs through an index. ``pipeline_run_id`` is
    set only by the write that first attached the ref, and is NULL when that write
    had no run; a failed run is rolled back by removing the refs that carry its
    id. ``position`` grows with every inserted row, which gives the refs of one
    node or edge their attach order.
    """
    return [
        Column("source_ref_key", Text, primary_key=True, nullable=False),
        Column("dataset_id", Text, nullable=False),
        Column("pipeline_run_id", Text),
        Column("position", BigInteger, Identity(always=True), nullable=False),
    ]


# A source ref is a row of its own, removed with its node or edge (ON DELETE
# CASCADE). The refs used to be four array columns on graph_node and graph_edge,
# where a node shared by many documents had its row rewritten with a longer list
# on every attach.
#
# The ``source_*`` columns and ``description`` are what the node carried in the
# write that attached the ref: its five source stamps and, for an ``Entity``, its
# description. ``add_nodes`` does not rewrite a stored node that differs from the
# incoming one only in those, so the node keeps the values it was first written
# with and each ref row keeps its own write's. They are NULL where the write did
# not carry the value and for a ref attached without a node payload
# (``attach_node_source_refs``). Edge writes are not compared this way, so edge
# refs have no such columns.
_node_source_ref_table = Table(
    "graph_node_source_ref",
    _meta,
    Column(
        "node_id",
        Text,
        ForeignKey("graph_node.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    ),
    *_source_ref_columns(),
    Column("source_pipeline", Text),
    Column("source_task", Text),
    Column("source_node_set", Text),
    Column("source_user", Text),
    Column("source_content_hash", Text),
    Column("description", Text),
)

_edge_source_ref_table = Table(
    "graph_edge_source_ref",
    _meta,
    Column("source_id", Text, primary_key=True, nullable=False),
    Column("target_id", Text, primary_key=True, nullable=False),
    Column("relationship_name", Text, primary_key=True, nullable=False),
    *_source_ref_columns(),
    ForeignKeyConstraint(
        ["source_id", "target_id", "relationship_name"],
        ["graph_edge.source_id", "graph_edge.target_id", "graph_edge.relationship_name"],
        ondelete="CASCADE",
    ),
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

# Graph-provenance delete and rollback look refs up by source ref, by dataset and
# by pipeline run (find_*_by_source_ref / _by_dataset / _by_pipeline_run).
Index("idx_node_source_ref_key", _node_source_ref_table.c.source_ref_key)
Index("idx_node_source_ref_dataset", _node_source_ref_table.c.dataset_id)
Index("idx_node_source_ref_run", _node_source_ref_table.c.pipeline_run_id)
Index("idx_edge_source_ref_key", _edge_source_ref_table.c.source_ref_key)
Index("idx_edge_source_ref_dataset", _edge_source_ref_table.c.dataset_id)
Index("idx_edge_source_ref_run", _edge_source_ref_table.c.pipeline_run_id)
