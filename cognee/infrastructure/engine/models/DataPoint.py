from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class MetaData(TypedDict):
    index_fields: list[str]


class DataPoint(BaseModel):
    __tablename__ = "data_point"
    id: UUID = Field(default_factory=uuid4)
    updated_at: Optional[datetime] = datetime.now(timezone.utc)
    topological_rank: Optional[int] = 0
    _metadata: Optional[MetaData] = {"index_fields": [], "type": "DataPoint"}

    # class Config:
    #     underscore_attrs_are_private = True

    @classmethod
    def get_embeddable_data(self, data_point):
        if (
            data_point._metadata
            and len(data_point._metadata["index_fields"]) > 0
            and hasattr(data_point, data_point._metadata["index_fields"][0])
        ):
            attribute = getattr(data_point, data_point._metadata["index_fields"][0])

            if isinstance(attribute, str):
                return attribute.strip()
            else:
                return attribute

    @classmethod
    def get_embeddable_properties(self, data_point):
        if data_point._metadata and len(data_point._metadata["index_fields"]) > 0:
            return [
                getattr(data_point, field, None) for field in data_point._metadata["index_fields"]
            ]

        return []

    @classmethod
    def get_embeddable_property_names(self, data_point):
        return data_point._metadata["index_fields"] or []
