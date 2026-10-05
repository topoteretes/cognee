from enum import Enum

from pydantic import ConfigDict

from cognee.infrastructure.engine import DataPoint


class EntityTypeCategory(str, Enum):
    """The fixed taxonomy an EntityType is filed under, the same in every dataset.

    Member names equal their values because BAML registers an enum by name but
    validates its answer by value.
    """

    technology = "technology"
    person = "person"
    concept = "concept"
    artifact = "artifact"
    organization = "organization"
    place = "place"
    event = "event"
    work = "work"
    other = "other"

    def __str__(self) -> str:
        # str(member) of a str-mixin Enum is "EntityTypeCategory.place". Code that
        # stringifies a node property, as the schema view and the Postgres adapter
        # do, must get the stored value.
        return self.value


class EntityType(DataPoint):
    # Held to the taxonomy when a task assigns the field on an instance. model_copy
    # does not validate, so such a task must assign, not copy with update=.
    model_config = ConfigDict(validate_assignment=True)

    name: str
    description: str
    # None means not classified yet, "other" means the classifier ran and found no
    # better fit. A value outside the taxonomy is rejected.
    category: EntityTypeCategory | None = None
    relations: list[tuple] = []
    # identity_fields makes the id deterministic and namespaced by class
    # (``EntityType:<name>``) when constructed without an explicit id — the same
    # value ``EntityType.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
