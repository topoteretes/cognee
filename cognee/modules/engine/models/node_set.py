from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id


class NodeSet(DataPoint):
    """NodeSet data point."""

    name: str


def node_sets_from_names(names: list[str]) -> list[NodeSet]:
    """One NodeSet per name, with the shared id scheme (``NodeSet:<name>``).

    Every caller that tags nodes with a node_set (documents, chunks, the code
    graph, ...) must build its ``NodeSet`` objects through this function, so
    the same name always resolves to the same node id and tags from
    different ingestions merge onto one NodeSet instead of duplicating it.
    """
    return [NodeSet(id=generate_node_id(f"NodeSet:{name}"), name=name) for name in names]
