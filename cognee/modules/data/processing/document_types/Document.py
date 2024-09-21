from uuid import UUID
from typing import Protocol

class Document(Protocol):
    id: UUID
    type: str
    title: str
    raw_data_location: str

    def read(self, chunk_size: int) -> str:
        pass
