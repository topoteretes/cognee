from collections.abc import Iterable
from uuid import UUID


def coerce_graph_uuid(value: str | UUID, field_name: str) -> UUID:
    """A graph node id as a ``UUID``, whether it arrived as one or as its string form.

    Graph adapters hand ids back as strings, and the relational ledger's id columns are
    ``Uuid(as_uuid=True)``, which SQLite binds through ``value.hex``: a str there fails.
    A value that is not a UUID at all is reported with the field it was meant for.
    """
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field_name} must be a UUID: {value!r}.") from error


def coerce_graph_uuid_set(values: Iterable[str | UUID], field_name: str) -> set[UUID]:
    return {coerce_graph_uuid(value, field_name) for value in values}
