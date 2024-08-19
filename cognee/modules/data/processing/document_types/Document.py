from uuid import UUID
from typing import Protocol

class Document(Protocol):
    id: UUID
    type: str
    title: str
    file_path: str

    def read(self) -> str:
        pass
