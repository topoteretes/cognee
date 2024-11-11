from typing_extensions import TypedDict
from uuid import UUID, uuid4
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class MetaData(TypedDict):
    index_fields: list[str]

class DataPoint(BaseModel):
    id: UUID = Field(default_factory = uuid4)
    updated_at: Optional[datetime] = datetime.now(timezone.utc)
    _metadata: Optional[MetaData] = {
        "index_fields": []
    }

    # class Config:
    #     underscore_attrs_are_private = True

    def get_embeddable_data(self):
        if self._metadata and len(self._metadata["index_fields"]) > 0 \
            and hasattr(self, self._metadata["index_fields"][0]):

            return getattr(self, self._metadata["index_fields"][0])
