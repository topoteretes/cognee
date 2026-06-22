from typing import List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType


class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: Optional[str] = None
    relations: List[tuple] = []
    # identity_fields makes the id deterministic and namespaced by class
    # (``Entity:<name>``) when constructed without an explicit id — the same
    # value ``Entity.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
