from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, ARRAY

from cognee.infrastructure.databases.relational import Base


class GraphMetricData(Base):
    __tablename__ = "graph_metric_table"

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


class InputMetricData(Base):
    __tablename__ = "input_metric_table"

    num_tokens = Column(Integer)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
