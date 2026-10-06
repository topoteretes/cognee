from enum import Enum

from cognee.infrastructure.engine import DataPoint


class EntityTypeCategory(str, Enum):
    """The fixed taxonomy an EntityType is filed under, the same in every dataset.

    What the classifier's answer is held to. Member names equal their values because
    BAML registers an enum by name but validates its answer by value.
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


class EntityType(DataPoint):
    name: str
    description: str
    # None means not classified yet, "other" means the classifier ran and found no
    # better fit. The classifier answers in EntityTypeCategory, which holds it to the
    # taxonomy. The field is a plain string so a reader that rebuilds EntityType from
    # stored properties never fails on a value it does not know.
    category: str | None = None
    relations: list[tuple] = []
    # identity_fields makes the id deterministic and namespaced by class
    # (``EntityType:<name>``) when constructed without an explicit id — the same
    # value ``EntityType.id_for(name)`` produces. Prevents the random-uuid4 footgun.
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
