from typing import Any
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id


class Triplet(DataPoint):
    text: str
    from_node_id: str
    to_node_id: str

    metadata: dict = {"index_fields": ["text"]}

    @staticmethod
    def id_for_edge(source_id: Any, relationship_name: Any, target_id: Any) -> UUID:
        """Return the point id of the triplet built from one graph edge.

        The id derives from the edge alone, so writers, deletes, and node-set-scoped
        retrieval can all address a triplet from its edge without reading it back.
        """
        return generate_node_id(str(source_id) + str(relationship_name) + str(target_id))
