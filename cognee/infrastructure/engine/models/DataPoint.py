

from datetime import datetime, timezone
from typing import Optional, Any, Dict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# Define metadata type
class MetaData(TypedDict):
    index_fields: list[str]


# Updated DataPoint model with versioning and new fields
class DataPoint(BaseModel):
    __tablename__ = "data_point"
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    updated_at: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    version: str = "0.1"  # Default version
    source: Optional[str] = None  # Path to file, URL, etc.
    type: Optional[str] = "text"  # "text", "file", "image", "video"
    topological_rank: Optional[int] = 0
    extra: Optional[Dict[str, Any]] = None  # For additional properties
    _metadata: Optional[MetaData] = Field(
        default={"index_fields": [], "type": "DataPoint"}
    )

    # Override the Pydantic configuration
    class Config:
        underscore_attrs_are_private = True

    @classmethod
    def get_embeddable_data(cls, data_point):
        """Retrieve embeddable data based on metadata's index_fields."""
        if (
            data_point._metadata
            and len(data_point._metadata["index_fields"]) > 0
            and hasattr(data_point, data_point._metadata["index_fields"][0])
        ):
            attribute = getattr(data_point, data_point._metadata["index_fields"][0])

            if isinstance(attribute, str):
                return attribute.strip()
            return attribute

    @classmethod
    def get_embeddable_properties(cls, data_point):
        """Retrieve all embeddable properties."""
        if data_point._metadata and len(data_point._metadata["index_fields"]) > 0:
            return [getattr(data_point, field, None) for field in data_point._metadata["index_fields"]]
        return []

    @classmethod
    def get_embeddable_property_names(cls, data_point):
        """Retrieve names of embeddable properties."""
        return data_point._metadata["index_fields"] or []

    def update_version(self, new_version: str):
        """Update the version and updated_at timestamp."""
        self.version = new_version
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)
