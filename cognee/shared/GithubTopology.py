from typing import Any

from pydantic import BaseModel


class Relationship(BaseModel):
    type: str
    attributes: dict[str, Any] | None = {}


class Document(BaseModel):
    name: str
    content: str
    filetype: str


class Directory(BaseModel):
    name: str
    documents: list[Document] = []
    directories: list["Directory"] = []


# Allows recursive Directory Model
Directory.model_rebuild()


class RepositoryProperties(BaseModel):
    custom_properties: dict[str, Any] | None = None
    location: str | None = None  # Simplified location reference


class RepositoryNode(BaseModel):
    node_id: str
    node_type: str  # 'document' or 'directory'
    properties: RepositoryProperties = RepositoryProperties()
    content: Document | Directory | None = None
    relationships: list[Relationship] = []


class RepositoryGraphModel(BaseModel):
    root: RepositoryNode
    default_relationships: list[Relationship] = []
