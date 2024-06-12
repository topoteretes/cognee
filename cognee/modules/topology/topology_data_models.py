


from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Type

class Relationship(BaseModel):
    type: str = Field(..., description="The type of relationship, e.g., 'belongs_to'.")
    source: Optional[str] = Field(None, description="The identifier of the source id in the relationship.")
    target: Optional[str] = Field(None, description="The identifier of the target id in the relationship.")
    properties: Optional[Dict[str, Any]] = Field(None, description="A dictionary of additional properties related to the relationship.")

class Document(BaseModel):
    node_id: str
    title: str
    description: Optional[str] = None
    default_relationship: Relationship

class DirectoryModel(BaseModel):
    node_id: str
    path: str
    summary: str
    documents: List[Document] = []
    subdirectories: List['DirectoryModel'] = []
    default_relationship: Relationship

DirectoryModel.update_forward_refs()

class DirMetadata(BaseModel):
    node_id: str
    summary: str
    owner: str
    description: Optional[str] = None
    directories: List[DirectoryModel] = []
    documents: List[Document] = []
    default_relationship: Relationship

class GitHubRepositoryModel(BaseModel):
    node_id: str
    metadata: DirMetadata
    root_directory: DirectoryModel



class RelationshipModel(BaseModel):
    type: str
    source: str
    target: str

class NodeModel(BaseModel):
    node_id: str
    name: str
    default_relationship: Optional[RelationshipModel] = None
    children: List[Union[Dict[str, Any], "NodeModel"]] = Field(default_factory=list)
NodeModel.update_forward_refs()