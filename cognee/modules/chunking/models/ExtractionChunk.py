from typing import List
from pydantic import BaseModel
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class ExtractionChunk(BaseModel):
    document_chunk: DocumentChunk
    potential_nodes: List[str] = []
    potential_relationships: List[str] = []
