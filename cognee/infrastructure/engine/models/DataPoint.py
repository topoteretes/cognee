from datetime import datetime, timezone
from typing import Optional, Any, Dict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import pickle


# Define metadata type
class MetaData(TypedDict):
    index_fields: list[str]


# Updated DataPoint model with versioning and new fields
class DataPoint(BaseModel):
    __tablename__ = "data_point"
    id: UUID = Field(default_factory=uuid4)
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    updated_at: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    version: int = 1  # Default version
    topological_rank: Optional[int] = 0
    _metadata: Optional[MetaData] = {"index_fields": [], "type": "DataPoint"}

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
            return attribute

    @classmethod
    def get_embeddable_properties(self, data_point):
        """Retrieve all embeddable properties."""
        if data_point._metadata and len(data_point._metadata["index_fields"]) > 0:
            return [
                getattr(data_point, field, None) for field in data_point._metadata["index_fields"]
            ]

        return []

    @classmethod
    def get_embeddable_property_names(self, data_point):
        """Retrieve names of embeddable properties."""
        return data_point._metadata["index_fields"] or []

    def update_version(self):
        """Update the version and updated_at timestamp."""
        self.version += 1
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)

    # JSON Serialization
    def to_json(self) -> str:
        """Serialize the instance to a JSON string."""
        return self.json()

    @classmethod
    def from_json(self, json_str: str):
        """Deserialize the instance from a JSON string."""
        return self.model_validate_json(json_str)

    # Pickle Serialization
    def to_pickle(self) -> bytes:
        """Serialize the instance to pickle-compatible bytes."""
        return pickle.dumps(self.dict())

    @classmethod
    def from_pickle(self, pickled_data: bytes):
        """Deserialize the instance from pickled bytes."""
        data = pickle.loads(pickled_data)
        return self(**data)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Serialize model to a dictionary."""
        return self.model_dump(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """Deserialize model from a dictionary."""
        return cls.model_validate(data)
