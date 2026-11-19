"""Metadata-only update: move a cognified document to other node sets, no reprocessing.

``update(data_id, None, dataset_id, node_set=[...])`` must make every node-set
filter treat the document as belonging to the new sets only, while its chunk
ids, vectors, text, positions, extraction results and non-membership edges stay
exactly as they were, and no LLM, embedding, loader or chunker runs. Runs on
the mocked-provider incremental fixture (default Ladybug + LanceDB + SQLite
stack, access control on), no keys needed.
"""

import asyncio
import contextvars
import hashlib
from contextlib import contextmanager
from uuid import uuid4

import pytest

from cognee.tests.e2e.incremental_update.backend_env import reset_backend_state
from cognee.tests.e2e.incremental_update.test_incremental_update import (
    CHUNK_TOKENS,
    _doc_chunk_nodes,
    _stored_text,
)
from cognee.tests.e2e.incremental_update.test_incremental_update import (
    incremental_env as _incremental_env,
)

# Reuse the module-scoped fixture (scratch roots, env, LLM mock) under its own name.
incremental_env = _incremental_env


def _document(tag: str, paragraphs: int = 4) -> str:
    """A multi-chunk text whose paragraphs name ``ENT<tag><i>`` and the shared ``ENTSHARED``."""
    return "".join(
        f"Paragraph {i} of {tag} ENT{tag}{i} ENTSHARED. "
        + " ".join(f"{tag.lower()}{i}{j:02d}" for j in range(38))
        + ".\n"
        for i in range(paragraphs)
    )


@contextmanager
def searchable_embeddings():
    """Nonzero, text-derived mock vectors for the block.

    The fixture's mock embeds every text as a zero vector, which no similarity
    search can rank, so CHUNKS searches would come back empty whatever the filter.
    """
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
        EmbeddingResult,
    )

    engine_class = type(get_embedding_engine())
    original = engine_class.__dict__["embed_text"]

    async def embed_text(self, text):
        size = self.get_vector_size()
        vectors = []
        for item in text:
            digest = hashlib.sha256(item.encode("utf-8")).digest()
            vectors.append([1.0 + digest[i % len(digest)] / 255 for i in range(size)])
        return EmbeddingResult(vectors, tokens=0)

    engine_class.embed_text = embed_text
    try:
        yield
    finally:
        engine_class.embed_text = original


@contextmanager
def no_processing():
    """Fail on any LLM, embedding, loader or chunker call made inside the block."""
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine
    from cognee.modules.chunking.TextChunker import TextChunker

    calls = []

    def forbidden(name):
        async def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was called during a node-set reassignment")

        return call

    def forbidden_generator(*args, **kwargs):
        calls.append("chunker")
        raise AssertionError("the chunker was called during a node-set reassignment")

    targets = [
        (LLMGateway, "acreate_structured_output", staticmethod(forbidden("LLM"))),
        (type(get_embedding_engine()), "embed_text", forbidden("embedding")),
        (LoaderEngine, "load_file", forbidden("loader")),
        (TextChunker, "read", forbidden_generator),
    ]
    originals = [(owner, name, owner.__dict__[name]) for owner, name, _ in targets]
    for owner, name, replacement in targets:
        setattr(owner, name, replacement)
    try:
        yield calls
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
    assert not calls, f"processing ran during the reassignment: {calls}"


def _ids_in(value) -> set[str]:
    """Every ``id`` found anywhere in a search result, whatever its shape."""
    found: set[str] = set()
    if isinstance(value, dict):
        if "id" in value:
            found.add(str(value["id"]))
        for item in value.values():
            found |= _ids_in(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found |= _ids_in(item)
    elif hasattr(value, "model_dump"):
        found |= _ids_in(value.model_dump())
    elif hasattr(value, "__dict__"):
        found |= _ids_in(vars(value))
    return found


@pytest.mark.asyncio
async def test_node_set_reassignment_moves_memory_without_reprocessing(incremental_env):
    with searchable_embeddings():
        await _reassignment_scenario()


async def _reassignment_scenario():
    await reset_backend_state()
    import cognee
    from cognee.api.v1.exceptions import UpdateRequestInvalidError, UpdateTargetNotFoundError
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.models import Data
    from cognee.modules.engine.models.node_set import NodeSet
    from cognee.modules.graph.exceptions import NodeSetReassignmentNotSupportedError
    from cognee.modules.search.types import SearchType
    from cognee.modules.users.methods import create_user, get_default_user

    pristine = contextvars.copy_context()

    async def in_fresh_context(coroutine):
        """Like an API request: no dataset context vars leak between calls."""
        return await asyncio.get_running_loop().create_task(coroutine, context=pristine.copy())

    async def update(*args, **kwargs):
        return await in_fresh_context(cognee.update(*args, **kwargs))

    text_a, text_b = _document("A"), _document("B")
    await cognee.add(text_a, dataset_name="ns_move", node_set=["alpha"])
    await cognee.add(text_b, dataset_name="ns_move", node_set=["alpha"])
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "ns_move")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)

    rows = await get_dataset_data(dataset.id)
    stored = {row.id: await _stored_text(user, row.id) for row in rows}
    doc_a = next(data_id for data_id, text in stored.items() if text == text_a)
    doc_b = next(data_id for data_id, text in stored.items() if text == text_b)

    async with set_database_global_context_variables(dataset.id, user.id):
        pass  # graph/vector configs persist for this task's inspections
    graph = await get_graph_engine()
    vector = await get_vector_engine_async()

    chunks_a = [node["id"] for node in await _doc_chunk_nodes(doc_a, text_a)]
    chunks_b = [node["id"] for node in await _doc_chunk_nodes(doc_b, text_b)]
    assert len(chunks_a) >= 2 and len(chunks_b) >= 2, "expected multi-chunk documents"

    async def chunk_search(node_name=None) -> set[str]:
        results = await in_fresh_context(
            cognee.search(
                "Paragraph",
                query_type=SearchType.CHUNKS,
                dataset_ids=[dataset.id],
                top_k=100,
                node_name=node_name,
            )
        )
        return _ids_in(results) & set(chunks_a + chunks_b)

    async def entity(name: str) -> dict:
        nodes, _ = await graph.get_graph_data()
        return next(
            props | {"id": node_id}
            for node_id, props in nodes
            if props.get("type") == "Entity" and props.get("name") == name
        )

    async def member(node_id: str, set_name: str) -> bool:
        return await graph.has_edge(str(node_id), str(NodeSet.id_for(set_name)), "belongs_to_set")

    async def memory_snapshot():
        """Everything a reassignment must leave untouched."""
        nodes, edges = await graph.get_graph_data()
        chunk_ids = [node["id"] for node in await _doc_chunk_nodes(doc_a, text_a)]
        rows = await vector.retrieve(
            "DocumentChunk_text", chunk_ids + chunks_b, include_vector=True
        )
        return {
            "chunk_ids": sorted(chunk_ids),
            "vectors": {str(row.id): list(row.payload["vector"]) for row in rows},
            "chunks": {
                node_id: (props.get("text"), props.get("chunk_index"))
                for node_id, props in nodes
                if props.get("type") == "DocumentChunk"
            },
            "entities": sorted(
                props["name"] for _, props in nodes if props.get("type") == "Entity"
            ),
            "edges": sorted(
                (str(source), str(target), relationship)
                for source, target, relationship, _ in edges
                if relationship != "belongs_to_set"
            ),
        }

    async def relational(data_id):
        async with get_relational_engine().get_async_session() as session:
            row = await session.get(Data, data_id)
            return row.node_set, row.external_metadata.get("node_set")

    before = await memory_snapshot()
    assert await chunk_search(["alpha"]) == set(chunks_a + chunks_b)

    # --- Move A from alpha to beta: memory follows, nothing is reprocessed ---- #
    with no_processing():
        moved = await update(doc_a, None, dataset.id, user=user, node_set=["beta"])
    assert moved["status"] == "metadata_updated", moved
    assert moved["pipeline_run_id"] is None and moved["fallback"] is None
    assert await memory_snapshot() == before, "chunks, vectors and extraction must not change"

    assert await chunk_search(["beta"]) == set(chunks_a)
    assert await chunk_search(["alpha"]) == set(chunks_b), "B keeps alpha, A left it"
    assert await relational(doc_a) == ('["beta"]', ["beta"])
    assert await relational(doc_b) == ('["alpha"]', ["alpha"])

    document = await graph.get_node(str(doc_a))
    assert document["belongs_to_set"] == ["beta"] and document["source_node_set"] == "beta"
    assert await member(doc_a, "beta") and not await member(doc_a, "alpha")
    payloads = await vector.retrieve("DocumentChunk_text", chunks_a)
    assert all(row.payload["belongs_to_set"] == ["beta"] for row in payloads)
    assert all('"beta"' in row.payload["external_metadata"] for row in payloads)

    # An entity only A extracted moves; the one both documents share keeps B's alpha.
    only_a = await entity("enta0")
    assert only_a["belongs_to_set"] == ["beta"]
    assert await member(only_a["id"], "beta") and not await member(only_a["id"], "alpha")
    shared = await entity("entshared")
    assert await member(shared["id"], "alpha") and await member(shared["id"], "beta")
    assert set(shared["belongs_to_set"]) == {"alpha", "beta"}
    assert await graph.get_node(str(NodeSet.id_for("alpha"))) is not None, "B still owns alpha"

    # --- Repeating the assignment changes nothing ----------------------------- #
    _, edges_before_repeat = await graph.get_graph_data()
    with no_processing():
        repeated = await update(doc_a, None, dataset.id, user=user, node_set=["beta"])
    assert repeated["status"] == "unchanged", repeated
    _, edges_after_repeat = await graph.get_graph_data()
    assert len(edges_after_repeat) == len(edges_before_repeat), "no duplicate memberships"

    # --- Several sets at once -------------------------------------------------- #
    with no_processing():
        several = await update(doc_a, None, dataset.id, user=user, node_set=["beta", "gamma"])
    assert several["status"] == "metadata_updated"
    assert await chunk_search(["gamma"]) == set(chunks_a)
    assert await chunk_search(["beta"]) == set(chunks_a)
    assert (await entity("enta0"))["belongs_to_set"] == ["beta", "gamma"]

    # --- Clearing: no set filter finds A, unfiltered search and vectors stay --- #
    with no_processing():
        cleared = await update(doc_a, None, dataset.id, user=user, node_set=[])
    assert cleared["status"] == "metadata_updated"
    assert await chunk_search(["beta", "gamma"]) == set()
    assert set(chunks_a) <= await chunk_search()
    assert await memory_snapshot() == before
    assert await relational(doc_a) == (None, None)
    assert await graph.get_node(str(NodeSet.id_for("gamma"))) is None, "an unowned set goes"
    assert not await member(only_a["id"], "beta")

    # --- Preconditions fail before anything is written ------------------------- #
    with pytest.raises(UpdateRequestInvalidError):
        await update(doc_a, None, dataset.id, user=user)
    with pytest.raises(UpdateTargetNotFoundError):
        await update(uuid4(), None, dataset.id, user=user, node_set=["delta"])
    intruder = await create_user(f"intruder_{uuid4().hex[:8]}@example.com", "pw")
    with pytest.raises(Exception) as denied:
        await update(doc_a, None, dataset.id, user=intruder, node_set=["delta"])
    assert "Permission" in type(denied.value).__name__
    vector_class = vector.__class__  # the adapter's class; the engine is a handle
    vector_class.supports_payload_update = False
    try:
        with pytest.raises(NodeSetReassignmentNotSupportedError):
            await update(doc_a, None, dataset.id, user=user, node_set=["delta"])
    finally:
        vector_class.supports_payload_update = True
    assert await relational(doc_a) == (None, None)
    assert await chunk_search(["delta"]) == set()

    # --- A failed write is completed by an identical retry --------------------- #
    original_update_payload = vector_class.update_payload

    async def failing_update_payload(self, collection_name, payload_updates):
        raise RuntimeError("simulated vector write failure")

    vector_class.update_payload = failing_update_payload
    try:
        with pytest.raises(RuntimeError, match="simulated vector write failure"):
            await update(doc_a, None, dataset.id, user=user, node_set=["delta"])
    finally:
        vector_class.update_payload = original_update_payload
    assert await relational(doc_a) == (None, None), "the row flips only after storage"

    with no_processing():
        retried = await update(doc_a, None, dataset.id, user=user, node_set=["delta"])
    assert retried["status"] == "metadata_updated"
    assert await chunk_search(["delta"]) == set(chunks_a)
    assert await relational(doc_a) == ('["delta"]', ["delta"])
    # Membership comes back after a clear: the extracted entity is tagged again.
    assert await member((await entity("enta0"))["id"], "delta")
    assert await memory_snapshot() == before
    assert await chunk_search(["alpha"]) == set(chunks_b)
