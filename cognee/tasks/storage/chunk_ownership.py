"""Chunk-scoped ownership of graph output (SDK-6 proposal, Phase 2).

Graph merging stores equal output once; ownership records every chunk that
produced it. Each DocumentChunk's model subtree is re-expanded in isolation
(the same traversal ``add_data_points`` uses for the combined forest), giving
the exact node ids and edge identities that chunk owns. Output reachable from
several chunks gets several owners; output reachable from none — the document
node itself, NodeSet tags — stays document-scoped.

The grouping contract for writes: every node/edge is written in the batch of
its FIRST owner (so its provenance stamp folds into the same atomic
statement), and the remaining owners are attached afterwards.
"""

from typing import Dict, List, Tuple
from uuid import UUID

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import NodeSet
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model

EdgeKey = Tuple[str, str, str]


class ChunkOwnership:
    """node id / edge identity -> ordered owning chunk ref keys."""

    def __init__(self) -> None:
        self.node_owners: Dict[str, List[str]] = {}
        self.edge_owners: Dict[EdgeKey, List[str]] = {}
        self.chunk_ref_keys: List[str] = []

    @property
    def has_chunks(self) -> bool:
        return bool(self.chunk_ref_keys)


def _edge_key(edge) -> EdgeKey:
    return (str(edge[0]), str(edge[1]), str(edge[2]))


def _document_scoped_type_names() -> set:
    """Names of node types that stay document-scoped even when a chunk's
    expansion reaches them: the document IS the document (v1 ref), and NodeSet
    tags outlive any chunk.

    Derived from the real type hierarchy — every ``Document`` subclass
    (CsvDocument, UnstructuredDocument, DltRowDocument, user-defined types) is
    covered automatically, so a new document type can never silently get its
    document node chunk-owned. It must be a NAME set because
    ``get_graph_from_model`` returns synthetic pydantic copies (minted in
    ``cognee.modules.storage.utils``) that keep the class NAME but not the
    class hierarchy — ``isinstance`` never matches them. Computed per call so
    document subclasses imported after this module (plugins, user models) are
    still seen; the walk is a handful of classes.
    """
    names = {Document.__name__, NodeSet.__name__}
    frontier = [Document]
    while frontier:
        cls = frontier.pop()
        for subclass in cls.__subclasses__():
            if subclass.__name__ not in names:
                names.add(subclass.__name__)
                frontier.append(subclass)
    return names


async def collect_chunk_ownership(
    data_points: List[DataPoint],
    dataset_id: UUID,
    data_id: UUID,
) -> ChunkOwnership:
    """Map every node/edge in the batch to the chunks that produced it.

    Attribution is ROOT-wise: each input data point (a summary, a chunk, …)
    is expanded in isolation with fresh tracking dicts, and every chunk found
    inside that expansion owns everything the expansion produced. This
    captures both directions of the model: a chunk's own subtree (entities it
    contains) AND its parents (the summary pointing at it via ``made_from``)
    land in the same expansion. Documents and NodeSet tags are excluded —
    they stay document-scoped regardless of which chunk's expansion reached
    them.
    """
    ownership = ChunkOwnership()
    seen_chunk_keys: dict = {}
    document_scoped_names = _document_scoped_type_names()

    for root in data_points:
        sub_nodes, sub_edges = await get_graph_from_model(
            root, added_nodes={}, added_edges={}, visited_properties={}
        )
        owner_keys = []
        for sub_node in sub_nodes:
            if type(sub_node).__name__ == "DocumentChunk":
                key = make_chunk_source_ref_key(dataset_id, data_id, sub_node.id)
                owner_keys.append(key)
                if key not in seen_chunk_keys:
                    seen_chunk_keys[key] = True
                    ownership.chunk_ref_keys.append(key)
        if not owner_keys:
            continue
        for sub_node in sub_nodes:
            if type(sub_node).__name__ in document_scoped_names:
                continue
            owners = ownership.node_owners.setdefault(str(sub_node.id), [])
            for key in owner_keys:
                if key not in owners:
                    owners.append(key)
        for sub_edge in sub_edges:
            owners = ownership.edge_owners.setdefault(_edge_key(sub_edge), [])
            for key in owner_keys:
                if key not in owners:
                    owners.append(key)

    return ownership
