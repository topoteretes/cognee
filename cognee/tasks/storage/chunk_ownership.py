"""Chunk-scoped ownership of graph output (SDK-6 proposal, Phase 2).

Graph merging stores equal output once; ownership records every chunk that
produced it. A chunk owns EXACTLY what its own extraction produced: itself,
the summary made from it, the entities (or events) it ``contains`` and their
types, the structural edges among those (``is_part_of``, ``contains``,
``is_a``, ``made_from``, ``belongs_to_set``), and the relationship edges its
extraction yielded — whether or not the graph already held them. Output
reachable from several chunks gets several owners; output no chunk produced —
the document node, NodeSet tags, ontology enrichment — stays document-scoped.

Exactness is what chunk-scoped deletion relies on: an artifact is hard-deleted
when its last owner dies and kept while any owner lives. Relationship edges
hang off their SOURCE entity (``Entity.relations``), so a walk that followed
them would hand a chunk everything downstream of any entity it mentions — a
chunk containing only ``hole`` owned the whole ``hole -> rabbit -> alice ->
queen`` chain, and deleting the chunk that contained ``queen`` left it alive as
a ghost. The walk therefore stops at entity boundaries, and relationship edges
are attributed from the chunk's own record (``_produced_edge_identities``),
which also covers edges the graph already held and that were never attached
to the model (the loss case: a fact deleted with its first producer while a
later chunk still stated it).

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


def _is_chunk(data_point) -> bool:
    return type(data_point).__name__ == "DocumentChunk"


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


def _without_relations(item):
    """A contained data point with its extracted relationships detached.

    ``Entity.relations`` accumulates the edges of EVERY chunk in a batch, so
    walking it would attribute other chunks' facts to this one. The chunk's
    own relationships come from its produced-edge record instead.
    """
    if isinstance(item, tuple) and len(item) == 2:
        edge, target = item
        return (edge, _without_relations(target))
    if isinstance(item, DataPoint) and hasattr(item, "relations"):
        return item.model_copy(update={"relations": []})
    return item


def _scoped_chunk(chunk):
    """The chunk with its contained entities' relations detached.

    Only a real ``contains`` list is scoped. A custom graph model stores its
    whole extracted model there (a pydantic object, not a list) and a chunk
    rebuilt from an export (COGX import) may carry no ``contains`` at all —
    both are walked exactly as they are.
    """
    contained = getattr(chunk, "contains", None)
    if not isinstance(contained, list) or not contained:
        return chunk
    return chunk.model_copy(update={"contains": [_without_relations(item) for item in contained]})


def _scoped_root(root):
    """The root with the ownership walk cut at entity boundaries.

    A chunk root, or a summary root pointing at its chunk via ``made_from``,
    is walked through a copy whose contained entities carry no relations.
    Any other root is walked as-is (it reaches no chunk, so it attributes
    nothing to chunk scope).
    """
    if _is_chunk(root):
        return _scoped_chunk(root)
    made_from = getattr(root, "made_from", None)
    if made_from is not None and _is_chunk(made_from):
        return root.model_copy(update={"made_from": _scoped_chunk(made_from)})
    return root


def _chunks_of(root) -> List[DataPoint]:
    """The ORIGINAL chunk objects a root is built from (they carry the record)."""
    if _is_chunk(root):
        return [root]
    made_from = getattr(root, "made_from", None)
    if made_from is not None and _is_chunk(made_from):
        return [made_from]
    return []


async def collect_chunk_ownership(
    data_points: List[DataPoint],
    dataset_id: UUID,
    data_id: UUID,
) -> ChunkOwnership:
    """Map every node/edge in the batch to the chunks that produced it.

    Attribution is ROOT-wise: each input data point (a summary, a chunk, …)
    is walked in isolation with fresh tracking dicts — through a copy that
    stops at entity boundaries — and the chunk found inside that walk owns
    everything the walk produced: the chunk itself, its summary and the
    ``made_from`` edge, the entities it contains with their types and tags,
    and the structural edges among them. The chunk's extracted relationships
    are added from its own record. Documents and NodeSet tags are excluded —
    they stay document-scoped regardless of which chunk's walk reached them.
    """
    ownership = ChunkOwnership()
    seen_chunk_keys: dict = {}
    document_scoped_names = _document_scoped_type_names()
    # Ids of every document-scoped node seen so far, accumulated across roots so
    # an edge can be classified even when its endpoints surfaced in different
    # walks. Node types do not change between roots.
    document_scoped_ids: set = set()

    for root in data_points:
        sub_nodes, sub_edges = await get_graph_from_model(
            _scoped_root(root), added_nodes={}, added_edges={}, visited_properties={}
        )
        owner_keys = []
        for sub_node in sub_nodes:
            if _is_chunk(sub_node):
                key = make_chunk_source_ref_key(dataset_id, data_id, sub_node.id)
                owner_keys.append(key)
                if key not in seen_chunk_keys:
                    seen_chunk_keys[key] = True
                    ownership.chunk_ref_keys.append(key)
            elif type(sub_node).__name__ in document_scoped_names:
                document_scoped_ids.add(str(sub_node.id))
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
            # An edge BETWEEN two document-scoped nodes is document-scoped too:
            # `document -[belongs_to_set]-> NodeSet` outlives every chunk, so
            # chunk-owning it would hand it the chunk's v2 key as its write
            # group — it would never carry the document's v1 key, and stripping
            # the dead chunk's refs would leave it with none at all. An edge
            # with no refs is invisible to delete_by_document (which resolves
            # artifacts through the dataset's ref maps), so it would leak.
            # `chunk -[is_part_of]-> document` keeps its chunk owner: that edge
            # exists only because the chunk does, and must die with it.
            if str(sub_edge[0]) in document_scoped_ids and str(sub_edge[1]) in document_scoped_ids:
                continue
            owners = ownership.edge_owners.setdefault(_edge_key(sub_edge), [])
            for key in owner_keys:
                if key not in owners:
                    owners.append(key)
        # The relationships this chunk's extraction yielded — including ones
        # the graph already held, which never enter the model at all.
        for chunk in _chunks_of(root):
            key = make_chunk_source_ref_key(dataset_id, data_id, chunk.id)
            for identity in getattr(chunk, "_produced_edge_identities", []):
                owners = ownership.edge_owners.setdefault(tuple(identity), [])
                if key not in owners:
                    owners.append(key)

    return ownership
