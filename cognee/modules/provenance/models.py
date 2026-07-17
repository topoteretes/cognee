"""Compact relational evidence sidecar for graph assertions.

Cognee already has durable representations for revisions (``Data``), segments
(``DocumentChunk``), activities (``PipelineRun``), and assertions (graph edges
with ``edge_object_id``). This table records only the missing many-to-many link.
It is append-only and is never consulted by graph-native deletion.
"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text, UUID

from cognee.infrastructure.databases.relational import Base


class ProvenanceEdgeEvidence(Base):
    __tablename__ = "provenance_edge_evidence"

    id = Column(UUID, primary_key=True, default=uuid4)
    tenant_id = Column(UUID, nullable=True)
    user_id = Column(UUID, nullable=False)
    dataset_id = Column(UUID, nullable=False)
    data_id = Column(UUID, nullable=False)
    pipeline_run_id = Column(UUID, nullable=True)

    chunk_id = Column(UUID, nullable=False)
    chunk_index = Column(Integer, nullable=True)

    # edge_id is the graph's deterministic edge_object_id.
    edge_id = Column(UUID, nullable=False)
    source_node_id = Column(UUID, nullable=False)
    destination_node_id = Column(UUID, nullable=False)
    relationship_name = Column(Text, nullable=False)

    evidence_kind = Column(String(32), nullable=False, default="extracted")
    source_task = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_prov_evidence_edge", "dataset_id", "edge_id"),
        Index("ix_prov_evidence_source", "dataset_id", "data_id", "chunk_id"),
        Index("ix_prov_evidence_run", "pipeline_run_id"),
    )
