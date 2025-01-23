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

    def to_json(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "mean_degree": self.mean_degree,
            "edge_density": self.edge_density,
            "num_connected_components": self.num_connected_components,
            "sizes_of_connected_components": self.sizes_of_connected_components,
            "num_selfloops": self.num_selfloops if self.num_selfloops else None,
            "diameter": self.diameter if self.diameter else None,
            "avg_shortest_path_length": self.avg_shortest_path_length
            if self.avg_shortest_path_length
            else None,
            "avg_clustering": self.avg_clustering if self.avg_clustering else None,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }


class InputMetricData(Base):
    __tablename__ = "input_metric_table"

    num_tokens = Column(Integer)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    def to_json(self) -> dict:
        return {
            "num_tokens": self.num_tokens,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }
