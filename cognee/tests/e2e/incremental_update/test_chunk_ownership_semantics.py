"""Deletion semantics of chunk-scoped ownership, end to end.

Chunk-level updates retire replaced chunks through ``delete_by_source_ref``:
an artifact is hard-deleted when its last owning chunk dies and kept while any
owner lives. These scenarios pin what that must mean for the graph a user sees:

* an entity no live chunk contains is gone after the update (no ghosts);
* a relationship every live chunk still states survives the deletion of the
  chunk that first produced it (no lost facts) — within one document through
  ``update()``, and across documents through ``datasets.delete_data``.

Each scenario is three sentences long with one chunk per paragraph, so every
assertion message can print, for the offending artifact, which chunks CONTAIN
it next to which chunks OWN it. Extraction is deterministic: capitalised words
are entities (typed by first letter) and consecutive entities in a chunk get a
``precedes`` edge. Runs on kuzu + lancedb + sqlite in a scratch root with
ENABLE_BACKEND_ACCESS_CONTROL=true — each dataset gets its own graph, so the
scenarios are independent — with mocked LLM and embeddings.
"""

import asyncio
import hashlib
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

CHUNK_TOKENS = 8
NOUN = re.compile(r"\b[A-Z][a-z]{3,}\b")


def _nouns(text: str) -> list:
    return [word.lower() for word in NOUN.findall(text)]


def _pairs(text: str) -> set:
    names = _nouns(text)
    return {(a, b) for a, b in zip(names, names[1:]) if a != b}


@pytest.fixture(scope="module")
def event_loop():
    """One event loop for the module: cognee's cached engines (asyncpg on the
    Postgres backend) bind to the loop that created them."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture(scope="module")
def ownership_env():
    """Scratch roots, env overrides, config-cache resets, and the extraction mock."""
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_ownership_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TRIPLET_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        # Isolated per-dataset DBs need a backend that supports them; Neo4j
        # Community cannot CREATE DATABASE, so its run sets INCR_TEST_ACL=false.
        ENABLE_BACKEND_ACCESS_CONTROL=os.environ.get("INCR_TEST_ACL", "true"),
    )
    os.environ.setdefault("LLM_API_KEY", "mock-key")

    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
        ("cognee.modules.cognify.config", "get_cognify_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            text = str(text_input)
            names = sorted(set(_nouns(text)))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type=f"kind{n[0]}", description=n) for n in names],
                edges=[
                    Edge(source_node_id=a, target_node_id=b, relationship_name="precedes")
                    for a, b in sorted(_pairs(text))
                ],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            digest = hashlib.sha256(str(text_input).encode()).hexdigest()[:12]
            return SummarizedContent(summary=f"Summary {digest}.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original_acreate = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    import cognee.api.v1.update.incremental as incremental_module

    async def _fixed_chunk_tokens():
        return CHUNK_TOKENS

    original_budget = incremental_module.get_max_chunk_tokens
    incremental_module.get_max_chunk_tokens = _fixed_chunk_tokens

    yield root

    LLMGateway.acreate_structured_output = original_acreate
    incremental_module.get_max_chunk_tokens = original_budget
    shutil.rmtree(root, ignore_errors=True)


async def _ingest(dataset_name: str, *texts: str):
    """Add and cognify the given documents into a fresh dataset; return (user, dataset, data ids)."""
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    for text in texts:
        await cognee.add(text, dataset_name=dataset_name)
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == dataset_name)
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    rows = await get_dataset_data(dataset.id)
    ids_by_text = {}
    for row in rows:
        stored = Path(row.raw_data_location.replace("file://", "")).read_text(encoding="utf-8")
        ids_by_text[stored] = row.id
    return user, dataset, [ids_by_text[text] for text in texts]


class GraphView:
    """The dataset's graph, read back by meaning: names instead of ids."""

    def __init__(self, chunks, entities, precedes, contains, owners, edge_owners):
        self.chunks = chunks  # chunk id -> text
        self.entities = entities  # entity name -> node id
        self.precedes = precedes  # {(source name, target name)}
        self.contains = contains  # entity name -> {chunk ids that contain it}
        self.owners = owners  # entity name -> {owning chunk ids}
        self.edge_owners = edge_owners  # (source, target) -> {owning chunk ids}

    def label(self, chunk_id: str) -> str:
        text = self.chunks.get(chunk_id)
        return repr(text.strip()[:24]) if text is not None else "<deleted chunk>"

    def supporting(self, source: str, target: str) -> set:
        return {cid for cid, text in self.chunks.items() if (source, target) in _pairs(text)}

    def describe_entity(self, name: str) -> str:
        return (
            f"entity {name!r}: contained by {sorted(map(self.label, self.contains.get(name, ())))}, "
            f"owned by {sorted(map(self.label, self.owners.get(name, ())))}"
        )

    def describe_edge(self, source: str, target: str) -> str:
        return (
            f"edge {source}->{target}: stated by {sorted(map(self.label, self.supporting(source, target)))}, "
            f"owned by {sorted(map(self.label, self.edge_owners.get((source, target), ())))}"
        )


async def _view(user, dataset) -> GraphView:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.provenance import parse_source_ref_key

    async with set_database_global_context_variables(dataset.id, user.id):
        graph = await get_graph_engine()
        nodes, edges = await graph.get_graph_data()
        node_refs = await graph.find_node_source_refs_by_dataset(str(dataset.id))
        edge_refs = await graph.find_edge_source_refs_by_dataset(str(dataset.id))

    props = {str(node_id): p for node_id, p in nodes}
    chunks = {n: p["text"] for n, p in props.items() if p.get("type") == "DocumentChunk"}
    entities = {p["name"]: n for n, p in props.items() if p.get("type") == "Entity"}
    name_of = {node_id: name for name, node_id in entities.items()}
    precedes = {
        (name_of[str(s)], name_of[str(t)])
        for s, t, r, _ in edges
        if r == "precedes" and str(s) in name_of and str(t) in name_of
    }
    contains = {}
    for s, t, r, _ in edges:
        if r == "contains" and str(t) in name_of:
            contains.setdefault(name_of[str(t)], set()).add(str(s))

    def chunk_owners(refs):
        return {
            str(parse_source_ref_key(ref).chunk_id)
            for ref in refs
            if parse_source_ref_key(ref).version == 2
        }

    owners = {name: chunk_owners(node_refs.get(node_id, [])) for name, node_id in entities.items()}
    edge_owners = {
        (name_of[str(e.source_id)], name_of[str(e.target_id)]): chunk_owners(refs)
        for e, refs in edge_refs.items()
        if e.relationship_name == "precedes" and str(e.source_id) in name_of
    }
    return GraphView(chunks, entities, precedes, contains, owners, edge_owners)


async def _vector_has(user, dataset, collection: str, node_id: str) -> bool:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    async with set_database_global_context_variables(dataset.id, user.id):
        vector = await get_vector_engine_async()
        if not await vector.has_collection(collection):
            return False
        return bool(await vector.retrieve(collection, [node_id]))


CHAIN = "Hole met Rabbit.\n\nRabbit met Alice.\n\nAlice met Queen."


@pytest.mark.asyncio
async def test_chunk_owns_exactly_what_it_contains(ownership_env):
    """At cognify time, every entity's owners are the chunks that contain it and every
    relationship's owners are the chunks that state it — the invariant deletion relies on."""
    await reset_backend_state()
    user, dataset, _ = await _ingest("ownership_exact", CHAIN)
    view = await _view(user, dataset)
    assert len(view.chunks) == 3, "each paragraph must be its own chunk for this scenario"

    wrong_entities = [
        view.describe_entity(name)
        for name in sorted(view.entities)
        if view.owners.get(name) != view.contains.get(name, set())
    ]
    wrong_edges = [
        view.describe_edge(a, b)
        for (a, b) in sorted(view.precedes)
        if view.edge_owners.get((a, b)) != view.supporting(a, b)
    ]
    assert not wrong_entities and not wrong_edges, (
        "ownership must be exact — a chunk that owns what it did not produce keeps ghosts "
        "alive; a producing chunk that is not an owner lets live facts be deleted:\n  "
        + "\n  ".join(wrong_entities + wrong_edges)
    )


@pytest.mark.asyncio
async def test_entity_orphaned_by_update_is_deleted(ownership_env):
    """Replace the only paragraph that mentions Queen: the entity, its relationship and its
    vector row must be gone, because no live chunk contains it any more."""
    await reset_backend_state()
    import cognee

    user, dataset, (data_id,) = await _ingest("ownership_ghost", CHAIN)
    before = await _view(user, dataset)
    assert before.contains["queen"] and all(
        "Queen" in before.chunks[cid] for cid in before.contains["queen"]
    )
    queen_node_id = before.entities["queen"]

    result = await cognee.update(
        data_id, "Hole met Rabbit.\n\nRabbit met Alice.\n\nNothing remains.", dataset.id, user=user
    )
    assert result["status"] == "incremental", result

    after = await _view(user, dataset)
    assert "queen" not in after.entities, (
        "no live chunk contains Queen after the update, yet the entity survived — "
        f"{after.describe_entity('queen')}"
    )
    assert ("alice", "queen") not in after.precedes, (
        "no live chunk states alice->queen after the update, yet the edge survived — "
        f"{after.describe_edge('alice', 'queen')}"
    )
    assert not await _vector_has(user, dataset, "Entity_name", queen_node_id), (
        "the orphaned entity's Entity_name vector row must be deleted with it"
    )


@pytest.mark.asyncio
async def test_fact_survives_loss_of_its_first_producer(ownership_env):
    """Two paragraphs state the same fact; delete the one that produced it first. The fact
    is still stated by a live chunk, so it must survive."""
    await reset_backend_state()
    import cognee

    user, dataset, (data_id,) = await _ingest(
        "ownership_loss", "Alice met Queen.\n\nBoris saw Carol."
    )

    result = await cognee.update(
        data_id, "Alice met Queen.\n\nAlice met Queen again today.", dataset.id, user=user
    )
    assert result["status"] == "incremental", result
    stated_twice = await _view(user, dataset)
    assert len(stated_twice.supporting("alice", "queen")) == 2, "both paragraphs state the fact"
    ownership_before = stated_twice.describe_edge("alice", "queen")

    result = await cognee.update(
        data_id, "Nothing here.\n\nAlice met Queen again today.", dataset.id, user=user
    )
    assert result["status"] == "incremental", result
    after = await _view(user, dataset)
    assert after.supporting("alice", "queen"), "a live chunk still states the fact"
    assert ("alice", "queen") in after.precedes, (
        "the fact alice->queen was deleted although a live chunk still states it. Before the "
        f"deletion: {ownership_before} — a chunk that states a fact but does not own its "
        "edge cannot keep it alive"
    )
    assert after.edge_owners.get(("alice", "queen")) == after.supporting("alice", "queen"), (
        f"every chunk that states a fact must own its edge — {after.describe_edge('alice', 'queen')}"
    )


@pytest.mark.asyncio
async def test_fact_survives_deleting_one_of_two_documents(ownership_env):
    """Two documents, ingested at different times, state the same fact; delete the first.
    The other still states it, so the edge must survive — this is the ``delete_data``
    path, not the incremental one, and it exists on ``dev`` independently of updates."""
    await reset_backend_state()
    import cognee
    from cognee.api.v1.datasets import datasets as datasets_api
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    user, dataset, (doc_a,) = await _ingest("ownership_cross_doc", "Alice met Queen.")
    # A later ingestion into the same dataset: its extraction yields an edge the
    # graph already holds from document A.
    await cognee.add("Alice met Queen at the Garden.", dataset_name="ownership_cross_doc")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    assert len(await get_dataset_data(dataset.id)) == 2
    both = await _view(user, dataset)
    assert len(both.supporting("alice", "queen")) == 2, "both documents state the fact"
    ownership_before = both.describe_edge("alice", "queen")

    await datasets_api.delete_data(dataset_id=dataset.id, data_id=doc_a, user=user)

    after = await _view(user, dataset)
    assert after.supporting("alice", "queen"), "document B still states the fact"
    assert ("alice", "queen") in after.precedes, (
        "deleting document A removed alice->queen although document B still states it. "
        f"Before the deletion: {ownership_before} — a chunk that states a fact but does not "
        "own its edge cannot keep it alive"
    )
    assert "alice" in after.entities and "queen" in after.entities, (
        "entities shared with the surviving document must survive"
    )
    assert after.edge_owners.get(("alice", "queen")) == after.supporting("alice", "queen"), (
        f"every chunk that states a fact must own its edge — {after.describe_edge('alice', 'queen')}"
    )
