from cognee.infrastructure.engine import DataPoint


class EntityType(DataPoint):
    """
    Represents a type of entity with a name and description.

    This class inherits from DataPoint and includes two primary attributes: `name` and
    `description`. Additionally, it contains a metadata dictionary that specifies
    `index_fields` for indexing purposes.
    """

    name: str
    description: str

    metadata: dict = {"index_fields": ["name"]}
