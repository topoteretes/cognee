"""The ``from_entities`` root written again after a document deletion.

Every test builds a ``DatasetContent`` by hand and fakes the stores, so no
database is involved. The deletion that precedes the rebuild has already
removed the deleted document's chunks, its bucket and the root; the content
here is what ``load_dataset_content`` reads after that.
"""

from importlib import import_module
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cognee.tasks.memify.global_context_index.build_document_index import build_context_summaries
from cognee.tasks.memify.global_context_index.ids import (
    create_bucket_id,
    create_document_bucket_id,
    create_root_summary_id,
)
from cognee.tasks.memify.global_context_index.models import DatasetContent, DatasetEntity
from cognee.tasks.memify.global_context_index.rebuild_root import (
    index_is_from_entities,
    rebuild_document_index_root,
    stored_document_buckets,
)

rebuild_module = import_module("cognee.tasks.memify.global_context_index.rebuild_root")
build_module = import_module("cognee.tasks.memify.global_context_index.build_document_index")

DATASET_ID = str(uuid4())
USER = SimpleNamespace(id=uuid4(), tenant_id=uuid4())


def content_after_deleting_gamma(*, index_shape="from_entities"):
    """alpha.md and beta.md remain; gamma.md's chunks and bucket are gone.

    ``existing_summary_texts`` is the index as the deletion left it: the two
    surviving buckets (``from_entities``), two clustering buckets (``llm``), or
    nothing (``emptied``: gamma.md held the only entities, so its bucket was the
    only one, and the survivors have chunks and no entity). The root is absent
    in every shape, as after a deletion.
    """
    entities = {
        "e_bob": DatasetEntity(name="Bob", type_name="Person", chunk_ids=frozenset({"c1"})),
        "e_cara": DatasetEntity(name="Cara", type_name="Person", chunk_ids=frozenset({"c2"})),
    }
    if index_shape == "from_entities":
        existing = {
            str(create_document_bucket_id(DATASET_ID, "d1")): "alpha.md\nPerson: Bob",
            str(create_document_bucket_id(DATASET_ID, "d2")): "beta.md\nPerson: Cara",
        }
    elif index_shape == "llm":
        existing = {
            str(create_bucket_id(DATASET_ID, 0, ["s1", "s2"])): "An LLM summary of a cluster",
            str(create_bucket_id(DATASET_ID, 0, ["s3"])): "Another cluster summary",
        }
    else:
        existing, entities = {}, {}
    return DatasetContent(
        dataset_id=DATASET_ID,
        document_names={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={"c1": "d1", "c2": "d2"},
        entities=entities,
        relations={},
        existing_summary_texts=existing,
    )


class FakeStores:
    """Records every write the rebuild makes."""

    def __init__(self, monkeypatch, content):
        self.persisted = []
        self.edges = []
        self.synced = []
        self.vector_deletes = []
        self.node_deletes = []
        self.engine = SimpleNamespace(graph="graph", vector="vector")

        async def load(data=None, ctx=None):
            self.ctx = ctx
            return content

        async def engine():
            return self.engine

        async def persist(summaries, ctx):
            self.persisted.extend(summaries)

        async def edges(edges_by_relationship, engine, ctx=None):
            self.edges.append(edges_by_relationship)

        async def sync(dataset_id, desired, index_engine):
            self.synced.append((dataset_id, list(desired), index_engine))
            return len(desired)

        async def delete_vectors(vector_engine, ids):
            self.vector_deletes.append(list(ids))

        async def delete_nodes(engine, ids):
            self.node_deletes.append(list(ids))

        monkeypatch.setattr(rebuild_module, "load_dataset_content", load)
        monkeypatch.setattr(rebuild_module, "get_unified_engine", engine)
        monkeypatch.setattr(rebuild_module, "persist_context_summaries", persist)
        monkeypatch.setattr(rebuild_module, "persist_context_index_edges", edges, raising=False)
        monkeypatch.setattr(
            import_module("cognee.tasks.memify.global_context_index.build_document_index"),
            "persist_context_index_edges",
            edges,
        )
        monkeypatch.setattr(rebuild_module, "sync_dataset_routes", sync)
        monkeypatch.setattr(rebuild_module, "safe_delete_context_vectors", delete_vectors)
        monkeypatch.setattr(rebuild_module, "delete_context_index_nodes", delete_nodes)

    def wrote_nothing(self):
        return not (
            self.persisted or self.edges or self.synced or self.vector_deletes or self.node_deletes
        )


@pytest.mark.asyncio
async def test_root_is_written_from_the_surviving_documents_and_mirrored(monkeypatch):
    content = content_after_deleting_gamma()
    stores = FakeStores(monkeypatch, content)

    root = await rebuild_document_index_root(DATASET_ID, USER)

    # The root is the one the next build would compute: same id, same text.
    expected_root = next(s for s in build_context_summaries(content) if s.is_root)
    assert root is not None
    assert root.id == create_root_summary_id(DATASET_ID)
    assert root.text == expected_root.text
    assert "gamma" not in root.text
    assert "alpha.md" in root.text and "beta.md" in root.text
    assert stores.persisted == [root]
    # The write is stamped like the build's: a context carrying the user and the dataset.
    assert stores.ctx.user is USER
    assert stores.ctx.dataset.id == UUID(DATASET_ID)
    # No root was stored, so no stale vector had to go first.
    assert stores.vector_deletes == []


@pytest.mark.asyncio
async def test_surviving_buckets_are_linked_to_the_new_root_and_mirrored_as_stored(monkeypatch):
    content = content_after_deleting_gamma()
    stores = FakeStores(monkeypatch, content)
    alpha_id = str(create_document_bucket_id(DATASET_ID, "d1"))
    beta_id = str(create_document_bucket_id(DATASET_ID, "d2"))
    root_id = str(create_root_summary_id(DATASET_ID))

    root = await rebuild_document_index_root(DATASET_ID, USER)

    # The root took every bucket's summarized_in edge with it; the buckets get
    # that edge back and nothing else (their document edge was never touched).
    assert stores.edges == [{"summarized_in": [(alpha_id, root_id), (beta_id, root_id)]}]
    # The mirror reconciles against the full index, so the buckets go in with the
    # text the index stores now: the mirror leaves their rows alone and copies
    # the root's vector only. Their recomputed text waits for the next build.
    [(synced_dataset, desired, index_engine)] = stores.synced
    assert synced_dataset == DATASET_ID
    assert index_engine == "vector"
    assert {str(s.id): s.text for s in desired} == {
        alpha_id: "alpha.md\nPerson: Bob",
        beta_id: "beta.md\nPerson: Cara",
        root_id: root.text,
    }


@pytest.mark.asyncio
async def test_a_root_still_stored_loses_its_vector_before_the_write(monkeypatch):
    """The id never changes and the stores keep a conflicting row's vector."""
    content = content_after_deleting_gamma()
    root_id = str(create_root_summary_id(DATASET_ID))
    content.existing_summary_texts[root_id] = "the root a failed removal left"
    stores = FakeStores(monkeypatch, content)

    await rebuild_document_index_root(DATASET_ID, USER)

    assert stores.vector_deletes == [[root_id]]


@pytest.mark.asyncio
async def test_llm_shaped_index_gets_no_root_and_no_call(monkeypatch):
    """Clustering ids are not derivable from the documents: that index's root is
    an LLM summary and stays dropped until its own build."""
    stores = FakeStores(monkeypatch, content_after_deleting_gamma(index_shape="llm"))

    assert await rebuild_document_index_root(DATASET_ID, USER) is None
    assert stores.wrote_nothing()


@pytest.mark.asyncio
async def test_an_emptied_index_gets_its_root_from_the_documents_that_remain(monkeypatch):
    """The deleted document held the only entities, so its bucket was the only
    one. The build gives a dataset its root whether or not any document has a
    bucket, and so does the rebuild: the root names the survivors by document."""
    content = content_after_deleting_gamma(index_shape="emptied")
    stores = FakeStores(monkeypatch, content)

    root = await rebuild_document_index_root(DATASET_ID, USER)

    assert root is not None
    assert root.text == next(s for s in build_context_summaries(content) if s.is_root).text
    assert "alpha.md" in root.text and "beta.md" in root.text
    assert stores.persisted == [root]
    # No bucket to relink, and the mirror gets the root alone.
    assert stores.edges == []
    [(_, desired, _)] = stores.synced
    assert [s.id for s in desired] == [root.id]


@pytest.mark.asyncio
async def test_the_rebuild_renders_no_bucket(monkeypatch):
    """Only the root is written, so the per-document tf-idf ranking that the
    build spends on every bucket is not run; the deletion holds the dataset lock."""
    stores = FakeStores(monkeypatch, content_after_deleting_gamma())

    def no_bucket_ranking(content):
        raise AssertionError("the rebuild ranked the buckets")

    monkeypatch.setattr(build_module, "_rank_document_entities", no_bucket_ranking)

    assert await rebuild_document_index_root(DATASET_ID, USER) is not None
    assert len(stores.persisted) == 1


@pytest.mark.asyncio
async def test_deleting_the_last_document_leaves_no_root(monkeypatch):
    """No surviving chunk: the dataset is empty (or its remaining documents have
    no memory), so there is nothing a root could name."""
    content = DatasetContent(
        dataset_id=DATASET_ID,
        document_names={},
        document_by_chunk={},
        entities={},
        relations={},
        existing_summary_texts={},
    )
    stores = FakeStores(monkeypatch, content)

    assert await rebuild_document_index_root(DATASET_ID, USER) is None
    assert stores.wrote_nothing()


@pytest.mark.asyncio
async def test_a_failure_is_a_warning_not_an_error(monkeypatch, caplog):
    """The forget is done by the time the rebuild runs; an embedding provider
    that cannot be reached leaves the root to the next improve."""
    content = content_after_deleting_gamma()
    stores = FakeStores(monkeypatch, content)

    async def embedding_failed(summaries, ctx):
        raise ConnectionError("embedding endpoint unreachable")

    monkeypatch.setattr(rebuild_module, "persist_context_summaries", embedding_failed)

    with caplog.at_level("WARNING", logger="global_context_index"):
        assert await rebuild_document_index_root(DATASET_ID, USER) is None

    assert any("could not be rebuilt" in record.message for record in caplog.records)
    # Nothing after the failed write ran: no edges, no mirror.
    assert stores.edges == [] and stores.synced == []
    # The write stores the node before it embeds the text, so the root node may
    # be there without a vector. It goes again: the build compares text, and
    # would otherwise keep that root as it is, never embed it, and fail at the
    # mirror on every improve, since the mirror copies the vector it lacks.
    assert stores.node_deletes == [[str(create_root_summary_id(DATASET_ID))]]


@pytest.mark.asyncio
async def test_a_failure_after_the_root_write_takes_the_root_out_again(monkeypatch):
    """The edges and the mirror follow the node write. A root left without
    its buckets' edges would also pass as complete at the next build, which
    relinks every bucket only to a root it found missing."""
    content = content_after_deleting_gamma()
    stores = FakeStores(monkeypatch, content)
    root_id = str(create_root_summary_id(DATASET_ID))

    async def edge_write_failed(edges_by_relationship, engine, ctx=None):
        raise RuntimeError("graph write failed")

    monkeypatch.setattr(build_module, "persist_context_index_edges", edge_write_failed)

    assert await rebuild_document_index_root(DATASET_ID, USER) is None

    assert [s.id for s in stores.persisted] == [UUID(root_id)]
    assert stores.synced == []
    assert stores.node_deletes == [[root_id]]


@pytest.mark.asyncio
async def test_a_rebuild_that_succeeds_removes_nothing(monkeypatch):
    stores = FakeStores(monkeypatch, content_after_deleting_gamma())

    assert await rebuild_document_index_root(DATASET_ID, USER) is not None

    assert stores.node_deletes == []


def test_stored_document_buckets_are_the_surviving_documents_with_a_bucket():
    content = content_after_deleting_gamma()
    # A third survivor with chunks and no entity never had a bucket; the root
    # names it, the buckets do not.
    content.document_by_chunk["c3"] = "d3"
    content.document_names["d3"] = "delta.md"

    buckets = stored_document_buckets(content)

    assert [(b.document_id, b.text, b.level, b.is_root) for b in buckets] == [
        ("d1", "alpha.md\nPerson: Bob", 0, False),
        ("d2", "beta.md\nPerson: Cara", 0, False),
    ]
    assert index_is_from_entities(content)


@pytest.mark.asyncio
async def test_an_index_node_no_surviving_document_derives_stops_the_rebuild(monkeypatch):
    """A node whose id is neither the root's nor a surviving document's bucket
    is not this shape, a clustering bucket or a bucket a failed removal left
    behind for a document whose chunks are gone, and the rebuild steps aside."""
    content = content_after_deleting_gamma()
    content.existing_summary_texts[str(create_document_bucket_id(DATASET_ID, "d9"))] = "stale"
    stores = FakeStores(monkeypatch, content)

    assert not index_is_from_entities(content)
    assert await rebuild_document_index_root(DATASET_ID, USER) is None
    assert stores.wrote_nothing()
