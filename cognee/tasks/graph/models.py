from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RelationshipModel(BaseModel):
    """
    Represents a relationship between two entities in a model.

    This class holds the type of the relationship and the identifiers for the source and
    target entities. It includes the following public instance variables:

    - type: A string indicating the type of relationship.
    - source: A string representing the source entity of the relationship.
    - target: A string representing the target entity of the relationship.
    """

    type: str
    source: str
    target: str


class NodeModel(BaseModel):
    """
    Represents a node in a hierarchical model structure with relationships to other nodes.

    Public methods:

    - __init__(self, node_id: str, name: str, default_relationship:
    Optional[RelationshipModel] = None, children: List[Union[Dict[str, Any], NodeModel]] =
    Field(default_factory=list))

    Instance variables:

    - node_id: Unique identifier for the node.
    - name: Name of the node.
    - default_relationship: Default relationship associated with the node, if any.
    - children: List of child nodes or dictionaries representing children for this node.
    """

    node_id: str
    name: str
    default_relationship: Optional[RelationshipModel] = None
    children: List[Union[Dict[str, Any], "NodeModel"]] = Field(default_factory=list)


NodeModel.model_rebuild()


class OntologyNode(BaseModel):
    """
    Represents a node in an ontology with a unique identifier, name, and description.
    """

    id: str = Field(..., description="Unique identifier made from node name.")
    name: str
    description: str


class OntologyEdge(BaseModel):
    """
    Represent an edge in an ontology, connecting a source and target with a specific
    relationship type.

    The class includes the following instance variables:
    - id: A unique identifier for the edge.
    - source_id: The identifier of the source node.
    - target_id: The identifier of the target node.
    - relationship_type: The type of relationship represented by this edge, defining how the
    source and target are related.
    """

    id: str
    source_id: str
    target_id: str
    relationship_type: str


class GraphOntology(BaseModel):
    """
    Represents a graph-based structure of ontology consisting of nodes and edges.

    The GraphOntology class contains a collection of OntologyNode instances representing the
    nodes of the graph and OntologyEdge instances representing the relationships between
    them. Public methods include the management of nodes and edges as well as any relevant
    graph operations. Instance variables include a list of nodes and a list of edges.
    """

    nodes: list[OntologyNode]
    edges: list[OntologyEdge]
