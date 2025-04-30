from cognee.infrastructure.engine import DataPoint


class NodeSet(DataPoint):
    """NodeSet data point."""

    name: str
    metadata: dict = {"index_fields": ["name"]}
