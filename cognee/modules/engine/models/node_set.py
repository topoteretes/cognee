from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id


class NodeSet(DataPoint):
    """A named group of nodes (``belongs_to_set``) used to organize and filter the graph."""

    name: str
    # identity_fields: ``NodeSet(name=...)`` without an explicit id derives the
    # same value ``NodeSet.id_for(name)`` returns. index_fields stays empty on
    # purpose: node-set names are filters, not embedded content.
    metadata: dict = {"index_fields": [], "identity_fields": ["name"]}

    @classmethod
    def id_for(cls, name: str) -> UUID:
        """The id of the node set called ``name``, the one place that formula lives.

        Kept on the legacy ``generate_node_id("NodeSet:<name>")`` derivation rather
        than DataPoint's default, which does not lower-case the class prefix: every
        node set already stored was created with this formula, and changing it would
        orphan them. Two spellings that normalize alike (case, spaces, apostrophes)
        are one node set.
        """
        return generate_node_id(f"NodeSet:{name}")
