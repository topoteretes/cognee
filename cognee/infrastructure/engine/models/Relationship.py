from datetime import datetime, timezone
from typing import Optional, Any, Dict
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import pickle

# Define metadata type
class RelationshipMetaData(TypedDict):
    index_fields: list[str]


class Relationship(BaseModel):
    __tablename__ = "relationship"
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID  # ID of the source node
    target_id: UUID  # ID of the target node
    relationship_type: str  # Type of relationship
    weight: Optional[float] = None  # Weight of the edge (optional)
    created_at: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    updated_at: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    version: str = "0.1"
    _metadata: Optional[RelationshipMetaData] = {
        "index_fields": [],
        "type": "Relationship"
    }

    class Config:
        underscore_attrs_are_private = True

    def update_version(self, new_version: str):
        """Update the version and updated_at timestamp."""
        self.version = new_version
        self.updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)

    def to_json(self) -> str:
        """Serialize the instance to a JSON string."""
        return self.json()

    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize the instance from a JSON string."""
        return cls.model_validate_json(json_str)

    def to_pickle(self) -> bytes:
        """Serialize the instance to pickle-compatible bytes."""
        return pickle.dumps(self.dict())

    @classmethod
    def from_pickle(cls, pickled_data: bytes):
        """Deserialize the instance from pickled bytes."""
        data = pickle.loads(pickled_data)
        return cls(**data)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Serialize model to a dictionary."""
        return self.model_dump(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Deserialize model from a dictionary."""
        return cls.model_validate(data)

    def get_embeddable_properties(self):
        """Retrieve embeddable properties for edge embeddings."""
        return {field: getattr(self, field, None) for field in self._metadata["index_fields"]}

    def get_embeddable_property_names(self):
        """Retrieve names of embeddable properties."""
        return self._metadata["index_fields"]
