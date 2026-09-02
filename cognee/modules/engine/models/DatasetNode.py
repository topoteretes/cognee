from cognee.infrastructure.engine import DataPoint


class DatasetNode(DataPoint):
    """Graph node that represents a dataset.

    The provenance lineage layer (see ``cognee.tasks.storage.provenance_lineage``)
    creates one of these per dataset so every extracted node has a path in the
    graph up to its dataset. The id is derived from the dataset id, so a single
    node is shared across every data item in the same dataset.

    It declares no ``index_fields``, so like ``NodeSet`` it is a structural node
    that lives only in the graph and is not embedded into the vector store.
    """

    name: str
