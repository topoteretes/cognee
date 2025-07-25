from datetime import datetime, timezone
from uuid import uuid5, NAMESPACE_OID
from sqlalchemy import UUID, Column, DateTime, String, Index

from cognee.infrastructure.databases.relational import Base


class GraphRelationshipLedger(Base):
    __tablename__ = "graph_relationship_ledger"

    id = Column(
        UUID,
        primary_key=True,
        default=lambda: uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
    )
    source_node_id = Column(UUID, nullable=False)
    destination_node_id = Column(UUID, nullable=False)
    creator_function = Column(String, nullable=False)
    node_label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    user_id = Column(UUID, nullable=True)

    # Create indexes
    __table_args__ = (
        Index("idx_graph_relationship_id", "id"),
        Index("idx_graph_relationship_ledger_source_node_id", "source_node_id"),
        Index("idx_graph_relationship_ledger_destination_node_id", "destination_node_id"),
    )
