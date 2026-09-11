"""A chunk that yields an already-stored relationship edge must still own it.

``integrate_chunk_graphs`` asks the graph which candidate edges already exist
and does not attach those to the data points again. Ownership is collected
from the attached model, so the second chunk to state a fact never becomes an
owner of its edge — the edge stays owned by whichever chunk first created it.

With chunk-scoped deletion that is a data-loss path: when the first producer's
chunk is deleted (an incremental edit, or ``delete_data`` on the first document)
the edge has no owner left and is hard-deleted, although a live chunk still
states the fact. Reproduced end to end in
``tests/e2e/incremental_update/test_chunk_ownership_semantics.py``; this pins
the seam in isolation.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.shared.data_models import Edge, KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.tasks.storage.chunk_ownership import collect_chunk_ownership

# The package re-exports ``extract_graph_from_data`` (the function) under the
# same name as its submodule, so a dotted patch target can resolve to the
# function instead of the module. Take the module object from sys.modules.
extract_module = sys.modules["cognee.tasks.graph.extract_graph_from_data"]


def _chunk(document, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=4,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
    )


def _extracted_fact() -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(id="alice", name="alice", type="kinda", description="alice"),
            Node(id="queen", name="queen", type="kindq", description="queen"),
        ],
        edges=[Edge(source_node_id="alice", target_node_id="queen", relationship_name="precedes")],
    )


async def _integrate(edge_already_stored: bool):
    document = TextDocument(
        id=uuid4(),
        name="doc",
        raw_data_location="doc.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    chunk = _chunk(document, "Alice met Queen again today.")

    # **kwargs, not a bare positional: the real helper is called with ctx and
    # chunk_owned. A stub that mismatched its signature made both tests below
    # die on a TypeError instead of checking the seam they were written for.
    async def existing(edge_identities, **_kwargs):
        return set(edge_identities) if edge_already_stored else set()

    with patch.object(
        extract_module, "find_existing_edge_identities", AsyncMock(side_effect=existing)
    ):
        (chunk,) = await integrate_chunk_graphs([chunk], [_extracted_fact()], KnowledgeGraph, None)

    dataset_id, data_id = uuid4(), uuid4()
    ownership = await collect_chunk_ownership([chunk], dataset_id, data_id)
    # ``contains`` entries are ``(Edge, Entity)`` tuples once edge text is attached.
    entities = {}
    for item in chunk.contains:
        entity = item[1] if isinstance(item, tuple) else item
        entities[entity.name] = entity
    edge_key = (str(entities["alice"].id), str(entities["queen"].id), "precedes")
    chunk_key = make_chunk_source_ref_key(dataset_id, data_id, chunk.id)
    return ownership.edge_owners.get(edge_key), chunk_key


def test_new_edge_is_owned_by_its_producing_chunk():
    owners, chunk_key = asyncio.run(_integrate(edge_already_stored=False))
    assert owners == [chunk_key], (
        "a freshly created edge must be owned by the chunk that yielded it"
    )


def test_already_stored_edge_is_still_owned_by_its_producing_chunk():
    owners, chunk_key = asyncio.run(_integrate(edge_already_stored=True))
    assert owners == [chunk_key], (
        "the chunk's extraction yielded alice -[precedes]-> queen, but because the edge "
        "already exists in the graph it was not attached and the chunk owns nothing for it "
        f"(owners recorded: {owners}). When the edge's first producer is deleted, the edge "
        "is hard-deleted while this chunk still states the fact."
    )


def test_a_chunk_produced_edge_gets_no_document_scoped_ref():
    """The document-scoped attach must skip what a chunk already owns.

    dev attaches the document's v1 ref to every already-stored relationship, so
    a later source keeps ownership of a fact it restates. Chunk-scoped
    ownership does that job at chunk granularity, and the two ran together: the
    edge ended up with both a v2 chunk ref and a v1 document ref. Chunk-level
    deletion only retires v2 keys, so once every chunk that stated the fact was
    gone the v1 ref kept it alive — a ghost fact that only a whole-document
    delete could remove. Caught by the stress e2e at step 06.
    """
    from cognee.infrastructure.databases.provenance import EdgeIdentity
    from cognee.modules.graph.utils.retrieve_existing_edges import (
        find_existing_edge_identities,
    )

    retrieve_module = sys.modules["cognee.modules.graph.utils.retrieve_existing_edges"]

    chunk_owned = EdgeIdentity("alice-id", "queen-id", "precedes")
    enrichment = EdgeIdentity("queen-id", "royalty-id", "is_a")

    def _triple(identity):
        return (identity.source_id, identity.target_id, identity.relationship_name)

    graph_engine = AsyncMock()
    graph_engine.has_edges = AsyncMock(
        return_value=[_triple(chunk_owned), _triple(enrichment)],
    )

    with (
        patch.object(retrieve_module, "get_graph_engine", AsyncMock(return_value=graph_engine)),
        patch.object(
            retrieve_module,
            "graph_provenance_write_kwargs",
            AsyncMock(
                return_value={"source_ref_key": "source_ref:v1:ds:doc", "pipeline_run_id": "run"}
            ),
        ),
    ):
        existing = asyncio.run(
            find_existing_edge_identities(
                [chunk_owned, enrichment], ctx=object(), chunk_owned=[chunk_owned]
            )
        )

    # Both still report as existing — only the ref attach is narrowed.
    assert existing == {chunk_owned, enrichment}

    attached = graph_engine.attach_edge_source_refs.await_args.args[0]
    assert chunk_owned not in attached, (
        "a chunk-produced edge must not also carry the document-scoped ref: chunk "
        "deletion cannot retire it, so the fact outlives every chunk that stated it"
    )
    assert enrichment in attached, (
        "an edge no chunk produced has no other owner and must keep the document ref"
    )
