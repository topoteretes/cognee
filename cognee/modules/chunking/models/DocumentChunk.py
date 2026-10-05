from pydantic import PrivateAttr, model_validator

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.external_metadata import normalize_external_metadata
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
    - external_metadata: The parent document's external_metadata as JSON text, copied onto
    every new chunk so hybrid retrieval can surface allowlisted keys; None when the document
    carries none.
    - metadata: A dictionary to hold meta information related to the chunk, including index
    fields.
    """

    text: str
    chunk_size: int
    chunk_index: int
    cut_type: str
    # Hex digest of `text` (see chunk_id.chunk_content_hash); the chunk id is
    # derived from it, so identity survives edits that shift chunk positions.
    content_hash: str | None = None
    # Token budget this chunk was cut against. Incremental updates re-chunk a
    # region with the budget recorded on the chunks it replaces, so a document
    # stays self-consistent even when the global configuration changes.
    max_chunk_tokens: int | None = None
    # Which chunker produced this chunk (Chunker.chunker_id). Chunkers disagree
    # on boundaries — an overlapping chunker's output cannot tile its input —
    # so a document may only be updated by the chunker that built it. None on
    # chunks written before the field existed, which reads as "unknown" and
    # falls through to the tiling check.
    chunker_id: str | None = None
    is_part_of: Document
    contains: list[Entity | Event | tuple[Edge, Entity]] = None
    importance_weight: float | None = 0.5
    document_id: str | None = None
    document_name: str | None = None
    # The parent document's external_metadata as JSON text (see
    # chunking/external_metadata.py for why text, not dict). Filled from
    # ``is_part_of`` by the validator below unless a caller passes it; never
    # embedded, never part of the chunk id.
    external_metadata: str | None = None
    # Optional truth-alignment fields; never embedded (kept out of index_fields)
    # and not part of id/dedup.
    truth_alignment: list[float] | None = None
    truth_epoch: int | None = None
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

    @model_validator(mode="after")
    def _inherit_document_external_metadata(self):
        """Carry the parent document's external_metadata unless the caller set one.

        Every place that builds a chunk (the chunkers, the incremental assembler,
        rehydrate) already passes ``is_part_of``, so deriving here means no
        construction site can forget the copy. An explicit value, such as the one
        rehydrate copies from a stored node, always wins.
        """
        if self.external_metadata is None:
            self.external_metadata = normalize_external_metadata(
                getattr(self.is_part_of, "external_metadata", None)
            )
        return self
