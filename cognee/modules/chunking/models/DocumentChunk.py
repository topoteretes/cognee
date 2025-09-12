from typing import List, Union, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity
from cognee.tasks.temporal_graph.models import Event


class DocumentChunk(DataPoint):
    """
    Represents a chunk of text from a document with associated metadata.

    Public methods include:

    - No public methods defined in the provided code.

    Instance variables include:

    - text: The textual content of the chunk.
    - chunk_size: The size of the chunk.
    - chunk_index: The index of the chunk in the original document.
    - cut_type: The type of cut that defined this chunk.
    - contains: Optional list of entities or events contained within the chunk (default: None).
    - is_part_of: Optional document to which this chunk belongs (default: None).
    - metadata: A dictionary to hold meta information related to the chunk, including index
    fields.
    """

    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    contains: Optional[List[Union[Entity, Event]]] = None
    is_part_of: Optional[Document] = None

    metadata: dict = {"index_fields": ["text"]}
