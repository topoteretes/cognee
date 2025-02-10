from datetime import datetime, timezone
from sqlalchemy.sql import func

from sqlalchemy import Column, DateTime, Float, Integer, JSON, UUID

from cognee.infrastructure.databases.relational import Base
from uuid import uuid4


class GraphMetrics(Base):
    __tablename__ = "graph_metrics"

    # TODO: Change ID to reflect unique id of graph database
    id = Column(UUID, primary_key=True, default=uuid4)
    num_tokens = Column(Integer, nullable=True)
    num_nodes = Column(Integer, nullable=True)
    num_edges = Column(Integer, nullable=True)
    mean_degree = Column(Float, nullable=True)
    edge_density = Column(Float, nullable=True)
    num_connected_components = Column(Integer, nullable=True)
    sizes_of_connected_components = Column(JSON, nullable=True)
    num_selfloops = Column(Integer, nullable=True)
    diameter = Column(Integer, nullable=True)
    avg_shortest_path_length = Column(Float, nullable=True)
    avg_clustering = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
