import json
from typing import List, Literal
from uuid import uuid4, UUID as UUID_t
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone
from fastapi.encoders import jsonable_encoder
from sqlalchemy import Boolean, Column, DateTime, JSON, UUID, String, TypeDecorator
from sqlalchemy.orm import mapped_column, Mapped

from cognee.infrastructure.databases.relational import Base


class NotebookCell(BaseModel):
    id: UUID_t
    type: Literal["markdown", "code"]
    name: str
    content: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class NotebookCellList(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, notebook_cells, dialect):
        if notebook_cells is None:
            return []
        return [
            json.dumps(jsonable_encoder(cell)) if isinstance(cell, NotebookCell) else cell
            for cell in notebook_cells
        ]

    def process_result_value(self, cells_json_list, dialect):
        if cells_json_list is None:
            return []
        return [NotebookCell(**json.loads(json_string)) for json_string in cells_json_list]


class Notebook(Base):
    __tablename__ = "notebooks"

    id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    owner_id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), index=True)

    name: Mapped[str] = mapped_column(String, nullable=False)

    cells: Mapped[List[NotebookCell]] = mapped_column(NotebookCellList, nullable=False)

    deletable: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
