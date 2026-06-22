"""Schema definitions for the Postgres graph adapter (graph_node, graph_edge)."""

from sqlalchemy import Table, Column, MetaData, String, DateTime, Index, ForeignKey, func
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
