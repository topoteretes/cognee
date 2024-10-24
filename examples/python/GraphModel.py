
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


async def add_data_points(collection_name: str, data_points: list):
    pass



class Summary(BaseModel):
    id: UUID
    text: str
    chunk: "Chunk"
    created_at: datetime
    updated_at: Optional[datetime]

    vector_index = ["text"]

class Chunk(BaseModel):
    id: UUID
    text: str
    summary: Summary
    document: "Document"
    created_at: datetime
    updated_at: Optional[datetime]
    word_count: int
    chunk_index: int
    cut_type: str

    vector_index = ["text"]

class Document(BaseModel):
    id: UUID
    chunks: list[Chunk]
    created_at: datetime
    updated_at: Optional[datetime]

class EntityType(BaseModel):
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: Optional[datetime]

    vector_index = ["name"]

class Entity(BaseModel):
    id: UUID
    name: str
    type: EntityType
    description: str
    chunks: list[Chunk]
    created_at: datetime
    updated_at: Optional[datetime]

    vector_index = ["name"]

class OntologyModel(BaseModel):
    chunks: list[Chunk]
