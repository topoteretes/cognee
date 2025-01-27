from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, ARRAY, UUID

from cognee.infrastructure.databases.relational import Base
from uuid import uuid4


class GraphMetricData(Base):
    __tablename__ = "graph_metric_table"

    # TODO: Change ID to reflect unique id of graph database
    id = Column(UUID, primary_key=True, default=uuid4)
    num_tokens = Column(Integer)
    num_nodes = Column(Integer)
    num_edges = Column(Integer)
    mean_degree = Column(Float)
    edge_density = Column(Float)
    num_connected_components = Column(Integer)
    sizes_of_connected_components = Column(ARRAY(Integer))
    num_selfloops = Column(Integer, nullable=True)
    diameter = Column(Integer, nullable=True)
    avg_shortest_path_length = Column(Float, nullable=True)
    avg_clustering = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
