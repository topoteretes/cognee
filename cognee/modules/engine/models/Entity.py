from typing import List, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.EntityType import EntityType


class Entity(DataPoint):
    name: str
    is_a: Optional[EntityType] = None
    description: str
    relations: List[tuple] = []
    # Optional truth-alignment fields; never embedded (kept out of index_fields)
    # and not part of id/dedup (kept out of identity_fields).
    truth_alignment: Optional[list[float]] = None
    truth_subspace_signature: Optional[str] = None
    truth_epoch: Optional[int] = None
    # identity_fields makes the id deterministic and namespaced by class
    # (``Entity:<name>``) when constructed without an explicit id — the same
    # value ``Entity.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
