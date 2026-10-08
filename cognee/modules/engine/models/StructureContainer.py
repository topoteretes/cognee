import json
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine import DataPoint

# The edge a document-source row (or a container) has to the thing it sits under.
CHILD_OF = "child_of"


class StructureContainer(DataPoint):
    """Something a document source files rows under that has no content of its own
    (a Notion database or data source), built from what the rows say about it."""

    name: str
    kind: str
    external_id: str
    source: str
    table_name: str
    dataset_id: str
    # Containers are structure, not content: nothing about them is embedded.
    metadata: dict = {"index_fields": []}

    @classmethod
    def container_id(
        cls, dataset_id: UUID, source: str, table_name: str, kind: str, external_id: str
    ) -> UUID:
        """Scoped by dataset, since with access control off two datasets syncing one
        workspace share a graph."""
        parts = ["cognee:structure", str(dataset_id), source, table_name, kind, external_id]
        return uuid5(NAMESPACE_OID, json.dumps(parts))
