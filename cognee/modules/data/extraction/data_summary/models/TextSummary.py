from pydantic import BaseModel

class TextSummary(BaseModel):
    text: str
    chunk_id: str
