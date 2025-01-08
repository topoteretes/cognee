from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RelationshipModel(BaseModel):
    type: str
    source: str
    target: str


class NodeModel(BaseModel):
    node_id: str
    name: str
    default_relationship: Optional[RelationshipModel] = None
    children: List[Union[Dict[str, Any], "NodeModel"]] = Field(default_factory=list)


NodeModel.model_rebuild()


class OntologyNode(BaseModel):
    id: str = Field(..., description="Unique identifier made from node name.")
    name: str
    description: str


class OntologyEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    relationship_type: str


class GraphOntology(BaseModel):
    nodes: list[OntologyNode]
    edges: list[OntologyEdge]
