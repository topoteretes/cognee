"""Schema definitions for the Turso (libSQL/SQLite) graph adapter (graph_node, graph_edge)."""

from sqlalchemy import Table, Column, MetaData, String, Text, DateTime, Index, ForeignKey, func

_meta = MetaData()

_node_table = Table(
    "graph_node",
    _meta,
    Column("id", String, primary_key=True),
    Column("name", String),
    Column("type", String),
    Column("properties", Text),
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
    Column("properties", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), server_default=func.now()),
)

Index("idx_edge_source", _edge_table.c.source_id)
Index("idx_edge_target", _edge_table.c.target_id)
Index("idx_node_type", _node_table.c.type)

# Covering indexes for SQLite
# SQLite doesn't support INCLUDE clauses on indexes like Postgres.
Index(
    "idx_edge_source_cover",
    _edge_table.c.source_id,
    _edge_table.c.target_id,
    _edge_table.c.relationship_name,
)
Index(
    "idx_edge_target_cover",
    _edge_table.c.target_id,
    _edge_table.c.source_id,
    _edge_table.c.relationship_name,
)
