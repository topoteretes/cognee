from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType
from typing import Optional
from datetime import datetime, timezone  
from pydantic import BaseModel, Field

class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str
    last_accessed_at: int = Field(  
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000)  
    )
    metadata: dict = {"index_fields": ["name"]}
