from typing import List

from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    name: str
    description: str
    relations: List[tuple] = []
    metadata: dict = {"index_fields": ["name"]}
