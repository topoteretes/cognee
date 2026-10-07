from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine import DataPoint

# The edge a document-source row (or a container) has to the thing it sits under.
CHILD_OF = "child_of"


class StructureContainer(DataPoint):
    """A node for something a document source organizes its rows under (a Notion
    database or data source, a folder) that has no content of its own.

    Built by the structure pass from what the rows say about their ancestors
    (see ``dlt_utils.STRUCTURE_COLUMN``), never extracted by an LLM. It keeps its
    source's id and kind so a reader can tell what it stands for.
    """

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
        """The id of a container, scoped by dataset so two datasets syncing the
        same workspace never share a node (the graph is shared when backend
        access control is off)."""
        return uuid5(
            NAMESPACE_OID,
            f"cognee:structure:{dataset_id}:{source}:{table_name}:{kind}:{external_id}",
        )
