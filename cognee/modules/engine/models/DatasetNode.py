from cognee.infrastructure.engine import DataPoint


class DatasetNode(DataPoint):
    """Provenance lineage node representing a dataset in the knowledge graph.

    Emitted by the provenance lineage layer (see
    ``cognee.tasks.storage.provenance_lineage``) so every extracted node has an
    in-graph path up to the dataset it originated from. Its id is derived
    deterministically from the dataset id, so a single DatasetNode is shared and
    deduplicated across every data item in the same dataset.

    Like ``NodeSet`` it declares no ``index_fields``, so it is a structural
    graph-only node and is not embedded into the vector store.
    """

    name: str
