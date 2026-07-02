"""Structured citations for the retrieval surface.

A :class:`Citation` is the machine-readable counterpart to the text
``Evidence:`` blocks produced by :mod:`cognee.modules.retrieval.utils.references`.
Both live side by side: the text version stays as an inline evidence
footer for human-facing completions, and the structured version is
what agents consume through
:class:`cognee.modules.recall.types.SearchResultItem`.

Two kinds are supported today:

* :attr:`CitationKind.CHUNK`: the answer is grounded in a specific
  document chunk. Populates ``document_name``, ``chunk_number``,
  ``chunk_id``, ``data_id``, and ``snippet``.
* :attr:`CitationKind.GRAPH`: the answer is grounded in a subgraph.
  Populates ``node_ids`` and ``edge_ids``, optionally with a rendered
  ``snippet`` describing the traversal.

Every field except ``kind`` is optional so a retriever that only knows
some of the provenance can still emit a valid citation. Callers that
need "you must be able to cite this back to a real thing" should check
that at least one identifier field is populated.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class CitationKind(str, Enum):
    """Which provenance shape a :class:`Citation` carries."""

    CHUNK = "chunk"
    GRAPH = "graph"


class Citation(BaseModel):
    """A single unit of provenance an agent can cite.

    Chunk citations answer "which passage of which document did this
    come from?". Graph citations answer "which subgraph traversal did
    this come from?". A retriever populates whichever fields are known
    and leaves the rest unset.
    """

    model_config = ConfigDict(use_enum_values=True)

    kind: CitationKind

    # Human-readable snippet from the cited source, if extractable. For
    # chunk citations this is the truncated text; for graph citations
    # this is optional (e.g. a rendered triplet).
    snippet: Optional[str] = None

    # Chunk-specific provenance. document_name + chunk_number is the
    # minimum contract for a chunk citation to be useful to a reader.
    document_name: Optional[str] = None
    chunk_number: Optional[int] = None
    chunk_id: Optional[str] = None
    data_id: Optional[str] = None

    # Graph-specific provenance. Node and edge ids are strings so any
    # backend representation (uuid, integer, cypher key) can be carried
    # without a coercion layer.
    node_ids: List[str] = []
    edge_ids: List[str] = []

    # Which dataset this citation belongs to, when known. Duplicated
    # across items in a multi-dataset recall so callers don't need to
    # cross-reference the response envelope.
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
