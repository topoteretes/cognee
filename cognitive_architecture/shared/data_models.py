from typing import Optional, List

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: int
    description: str
    category: str
    color: str = "blue"
    memory_type: str
    created_at: Optional[float] = None
    summarized: Optional[bool] = None


class Edge(BaseModel):
    source: int
    target: int
    description: str
    color: str = "blue"
    created_at: Optional[float] = None
    summarized: Optional[bool] = None


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


class GraphQLQuery(BaseModel):
    query: str


class MemorySummary(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)
