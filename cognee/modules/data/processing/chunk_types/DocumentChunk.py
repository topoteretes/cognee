from pydantic import BaseModel

class DocumentChunk(BaseModel):
    text: str
    word_count: int
    document_id: str
    chunk_id: str
    chunk_index: int
    cut_type: str
