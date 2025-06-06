from cognee.infrastructure.engine import DataPoint
from typing import Optional


class GraphitiNode(DataPoint):
    """
    Represent a node in a graph with optional content, name, and summary attributes.

    This class extends DataPoint and includes a metadata dictionary that specifies the index
    fields for the node's data. The public instance variables are:

    - content: an optional string representing the content of the node.
    - name: an optional string representing the name of the node.
    - summary: an optional string providing a summary of the node.
    - metadata: a dictionary outlining the fields used for indexing.
    """

    content: Optional[str] = None
    name: Optional[str] = None
    summary: Optional[str] = None

    metadata: dict = {"index_fields": ["name", "summary", "content"]}
