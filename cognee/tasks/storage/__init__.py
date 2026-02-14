"""
Graph persistence and indexing tasks.

This module provides tasks for adding structured DataPoints to the graph database, 
deduplicating extracted nodes and edges, and managing vector-based indexing for both 
nodes and relationship types.
"""

from .add_data_points import add_data_points
from .index_data_points import index_data_points
from .index_graph_edges import index_graph_edges
