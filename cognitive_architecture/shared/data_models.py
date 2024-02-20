"""Data models for the cognitive architecture."""
from typing import Optional, List
from pydantic import BaseModel, Field


class Node(BaseModel):
    """Node in a knowledge graph."""
    id: int
    description: str
    category: str
    color: str = "blue"
    memory_type: str
    created_at: Optional[float] = None
    summarized: Optional[bool] = None


class Edge(BaseModel):
    """Edge in a knowledge graph."""
    source: int
    target: int
    description: str
    color: str = "blue"
    created_at: Optional[float] = None
    summarized: Optional[bool] = None


class KnowledgeGraph(BaseModel):
    """Knowledge graph."""
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


class GraphQLQuery(BaseModel):
    """GraphQL query."""
    query: str


class MemorySummary(BaseModel):
    """ Memory summary. """
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)
