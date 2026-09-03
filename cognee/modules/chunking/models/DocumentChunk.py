from typing import List, Union, Optional

from pydantic import PrivateAttr

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
    # Hex digest of `text` (see chunk_id.chunk_content_hash); the chunk id is
    # derived from it, so identity survives edits that shift chunk positions.
    content_hash: Optional[str] = None
    # Token budget this chunk was cut against. Incremental updates re-chunk a
    # region with the budget recorded on the chunks it replaces, so a document
    # stays self-consistent even when the global configuration changes.
    max_chunk_tokens: Optional[int] = None
    # Which chunker produced this chunk (Chunker.chunker_id). Chunkers disagree
    # on boundaries — an overlapping chunker's output cannot tile its input —
    # so a document may only be updated by the chunker that built it. None on
    # chunks written before the field existed, which reads as "unknown" and
    # falls through to the tiling check.
    chunker_id: Optional[str] = None
    is_part_of: Document
    contains: List[Union[Entity, Event, tuple[Edge, Entity]]] = None
    importance_weight: Optional[float] = 0.5
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    # Optional truth-alignment fields; never embedded (kept out of index_fields)
    # and not part of id/dedup.
    truth_alignment: Optional[list[float]] = None
    truth_epoch: Optional[int] = None
    metadata: dict = {"index_fields": ["text"]}

    # Two records of the same extraction, kept apart because their readers
    # need different shapes. Both are filled by
    # ``construct_data_points_and_edges`` before edge deduplication, so both
    # see edges the graph already held. Private: never persisted as node
    # properties.
    #
    # Relationship edges this chunk's OWN extraction yielded, as
    # (source_id, target_id, relationship_name), deduplicated. Chunk ownership
    # is derived from this record, not from the entities' ``relations`` lists
    # (which accumulate edges from every chunk in a batch).
    _produced_edge_identities: list = PrivateAttr(default_factory=list)
    # Per-chunk semantic graph identities used by the provenance sidecar,
    # carrying the edge text and every occurrence rather than a unique set.
    _provenance_edges: list = PrivateAttr(default_factory=list)
