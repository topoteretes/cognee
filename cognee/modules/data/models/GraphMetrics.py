from datetime import datetime, timezone
from sqlalchemy.sql import func

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, UUID, false

from cognee.infrastructure.databases.relational import Base
from uuid import uuid4


class GraphMetrics(Base):
    """One pipeline run's graph metrics, cached under that run's id.

    Rows are written by two paths with different appetites: the full
    computation in ``get_pipeline_run_metrics``, and the node/edge count in
    ``get_datasets_graph_counts``, which only ever fills ``num_nodes`` and
    ``num_edges`` because that is all the graph-summary endpoints read.
    ``has_full_metrics`` tells the two apart — see the column's comment.
    """

    __tablename__ = "graph_metrics"

    # TODO: Change ID to reflect unique id of graph database
    id = Column(UUID, primary_key=True, default=uuid4)
    # False on a row holding only node/edge counts. Without it, the cheap
    # counting path would look to `get_pipeline_run_metrics` like a finished
    # cache entry, and that run's token count and connectivity metrics would
    # read as NULL forever, with nothing reporting they were never computed.
    has_full_metrics = Column(Boolean, nullable=False, default=False, server_default=false())
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
