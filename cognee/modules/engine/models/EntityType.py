from typing import Any

from pydantic import field_validator

from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    name: str
    description: str
    relations: list[tuple] = []
    # identity_fields makes the id deterministic and namespaced by class
    # (``EntityType:<name>``) when constructed without an explicit id — the same
    # value ``EntityType.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    @field_validator("name")
    @classmethod
    def normalize_name(cls, name: str) -> str:
        return name.lower().replace(" ", "")

    @classmethod
    def _normalize_identity_value(cls, value: Any) -> str:
        # Raw-name lookups and constructed types must use the same identity.
        if isinstance(value, str):
            value = cls.normalize_name(value)
        return super()._normalize_identity_value(value)
