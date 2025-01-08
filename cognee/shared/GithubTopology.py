from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union


class Relationship(BaseModel):
    type: str
    attributes: Optional[Dict[str, Any]] = {}


class Document(BaseModel):
    name: str
    content: str
    filetype: str


class Directory(BaseModel):
    name: str
    documents: List[Document] = []
    directories: List["Directory"] = []


# Allows recursive Directory Model
Directory.model_rebuild()


class RepositoryProperties(BaseModel):
    custom_properties: Optional[Dict[str, Any]] = None
    location: Optional[str] = None  # Simplified location reference


class RepositoryNode(BaseModel):
    node_id: str
    node_type: str  # 'document' or 'directory'
    properties: RepositoryProperties = RepositoryProperties()
    content: Union[Document, Directory, None] = None
    relationships: List[Relationship] = []


class RepositoryGraphModel(BaseModel):
    root: RepositoryNode
    default_relationships: List[Relationship] = []
