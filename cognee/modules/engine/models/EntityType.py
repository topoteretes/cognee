from typing import List

from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    name: str
    description: str
    relations: List[tuple] = []
    # identity_fields makes the id deterministic and namespaced by class
    # (``EntityType:<name>``) when constructed without an explicit id — the same
    # value ``EntityType.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
