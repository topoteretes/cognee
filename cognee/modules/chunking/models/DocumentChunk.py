from typing import List, Union, Optional

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
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
    - is_part_of: The document to which this chunk belongs.
    - contains: A list of entities or events contained within the chunk (default is None).
    - document_id: Flat string id of the source document, for reference rendering.
    - document_name: Display name (basename) of the source document, for reference rendering.
    - metadata: A dictionary to hold meta information related to the chunk, including index
    fields.
    """

    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    is_part_of: Document
    contains: List[Union[Entity, Event, tuple[Edge, Entity]]] = None
    importance_weight: Optional[float] = 0.5
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    # Optional truth-alignment fields; never embedded (kept out of index_fields)
    # and not part of id/dedup.
    truth_alignment: Optional[list[float]] = None
    truth_subspace_signature: Optional[str] = None
    metadata: dict = {"index_fields": ["text"]}
