"""Task 2 of the ``from_entities`` global context index: rank, render, write.

Every test builds a ``DatasetContent`` by hand, so no database is involved.
"""

from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    make_chunk_source_ref_key,
    make_source_ref_key,
)
from cognee.tasks.memify.global_context_index.build_document_index import build_context_summaries
from cognee.tasks.memify.global_context_index.constants import ROOT_MAX_DOCUMENTS
from cognee.tasks.memify.global_context_index.models import DatasetContent, DatasetEntity

build_document_index_module = import_module(
    "cognee.tasks.memify.global_context_index.build_document_index"
)
persist_module = import_module("cognee.tasks.memify.global_context_index.persist")


def make_content(
    *,
    documents,
    document_by_chunk,
    entities,
    relations=(),
    existing_summary_texts=None,
    dataset_id="ds",
):
    """``entities`` maps entity id -> (name, type name, chunk ids); ``relations`` are triples."""
    return DatasetContent(
        dataset_id=dataset_id,
        document_names=dict(documents),
        document_by_chunk=dict(document_by_chunk),
        entities={
            entity_id: DatasetEntity(name=name, type_name=type_name, chunk_ids=frozenset(chunk_ids))
            for entity_id, (name, type_name, chunk_ids) in entities.items()
        },
        relations={(source_id, target_id): (text,) for source_id, target_id, text in relations},
        existing_summary_texts=dict(existing_summary_texts or {}),
    )


def two_document_content(**overrides):
    """Two documents sharing one dataset-wide entity, which therefore has idf 0."""
    return make_content(
        documents={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={"c1": "d1", "c2": "d1", "c3": "d2"},
        entities={
            "e_shared": ("Shared", "Thing", {"c1", "c2", "c3"}),
            "e_acme": ("Acme", "Organization", {"c1", "c2"}),
            "e_bob": ("Bob", "Person", {"c1"}),
            "e_cara": ("Cara", "Person", {"c3"}),
        },
        relations=[
            ("e_bob", "e_acme", "Bob works at Acme"),
            ("e_shared", "e_acme", "Shared relates to Acme"),
            ("e_acme", "e_cara", "Acme knows Cara"),
        ],
        **overrides,
    )


def index_texts(content):
    """Each bucket's text by document id, and the root's under ``"root"``."""
    return {
        summary.document_id or "root": summary.text for summary in build_context_summaries(content)
    }


# --- selection -------------------------------------------------------------


def test_entity_in_every_chunk_ranks_last_in_each_bucket_and_tops_the_root():
    texts = index_texts(two_document_content())

    # tf-idf scores it 0 -- it cannot tell the two documents apart -- so it comes
    # last, but it is still named while the bucket has room.
    assert texts["d1"].startswith("alpha.md\nPerson: Bob\nOrganization: Acme\nThing: Shared\n")
    assert texts["d2"] == "beta.md\nPerson: Cara\nThing: Shared"
    # The root ranks the same name the opposite way, by document frequency.
    assert "\nEntities: Shared, " in texts["root"]


def test_bucket_entities_are_ranked_by_term_frequency_times_idf():
    # alpha.md holds three of the seven chunks. Term frequency alone would put
    # Common second, idf alone would put Rare first and name order Common first;
    # only their product gives Often (3 x log 7/3), Rare (1 x log 7/1), Common
    # (2 x log 7/5).
    content = make_content(
        documents={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={
            "c1": "d1",
            "c2": "d1",
            "c3": "d1",
            "c4": "d2",
            "c5": "d2",
            "c6": "d2",
            "c7": "d2",
        },
        entities={
            "e_often": ("Often", "Thing", {"c1", "c2", "c3"}),
            "e_rare": ("Rare", "Thing", {"c1"}),
            "e_common": ("Common", "Thing", {"c1", "c2", "c4", "c5", "c6"}),
        },
    )

    assert index_texts(content)["d1"] == "alpha.md\nThing: Often, Rare, Common"


def test_equal_scores_always_render_in_the_same_order():
    content = make_content(
        documents={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={"c1": "d1", "c2": "d2"},
        entities={
            "a-id": ("Bruno", "Person", {"c1"}),
            "z-id": ("Alice", "Person", {"c1"}),
        },
    )

    # Same score, so the name breaks the tie - never the dict, set or id order.
    assert index_texts(content)["d1"] == "alpha.md\nPerson: Alice, Bruno"


def test_document_without_any_entities_gets_no_bucket():
    content = make_content(
        documents={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={"c1": "d1", "c2": "d2"},
        entities={"e_acme": ("Acme", "Organization", {"c1"})},
    )

    summaries = build_context_summaries(content)

    assert [summary.document_id for summary in summaries] == ["d1", None]


def test_empty_dataset_produces_no_summaries_at_all_not_even_a_root():
    content = make_content(
        documents={},
        document_by_chunk={},
        entities={},
        existing_summary_texts={"stale-bucket": "old", "stale-root": "old"},
    )

    assert build_context_summaries(content) == []


def test_one_chunk_dataset_still_names_every_entity():
    # One chunk means every entity has idf log(1/1) = 0, so nothing scores -- but the
    # bucket has room, so every name is still stated rather than dropped.
    names = [f"Name{index}" for index in range(8)]
    content = make_content(
        documents={"d1": "only.md"},
        document_by_chunk={"c1": "d1"},
        entities={f"e{index}": (name, "Person", {"c1"}) for index, name in enumerate(names)},
    )

    bucket, root = build_context_summaries(content)

    assert bucket.text == "only.md\nPerson: " + ", ".join(names)
    assert root.text.startswith("Topics: Person (8)\n")


# --- relations -------------------------------------------------------------


def test_relation_is_dropped_when_an_endpoint_is_not_in_this_bucket():
    texts = index_texts(two_document_content())

    # Cara belongs to the other document, so the line naming her cannot be stated here.
    assert texts["d1"].endswith("Thing: Shared\nBob works at Acme\nShared relates to Acme")


def test_relations_are_ranked_by_their_weaker_endpoint():
    # Ann is in one chunk, Bea and Cy in two, Dan in three, so in alpha.md Ann
    # scores highest, Bea and Cy in the middle, Dan lowest.
    content = make_content(
        documents={"d1": "alpha.md", "d2": "beta.md"},
        document_by_chunk={"c1": "d1", "c2": "d2", "c3": "d2", "c4": "d2"},
        entities={
            "e_ann": ("Ann", "Person", {"c1"}),
            "e_bea": ("Bea", "Person", {"c1", "c2"}),
            "e_cy": ("Cy", "Person", {"c1", "c2"}),
            "e_dan": ("Dan", "Person", {"c1", "c2", "c3"}),
        },
        relations=[
            ("e_ann", "e_dan", "Ann met Dan"),
            ("e_bea", "e_cy", "Bea met Cy"),
        ],
    )

    # The middle pair comes first. By sum or max, or by text, Ann's line would.
    assert index_texts(content)["d1"].endswith("\nBea met Cy\nAnn met Dan")


# --- rendering -------------------------------------------------------------


def test_bucket_text_leads_with_the_document_name_and_orders_types_by_best_entity():
    text = index_texts(two_document_content())["d1"]

    assert text == (
        "alpha.md\nPerson: Bob\nOrganization: Acme\nThing: Shared\n"
        "Bob works at Acme\nShared relates to Acme"
    )


def test_root_text_is_three_name_dense_lines():
    text = index_texts(two_document_content())["root"]

    assert text == (
        "Topics: Person (2), Organization (1), Thing (1)\n"
        "Entities: Shared, Acme, Bob, Cara\n"
        "Documents: alpha.md, beta.md"
    )


def test_root_document_line_reports_how_many_documents_it_left_out():
    document_count = ROOT_MAX_DOCUMENTS + 3
    content = make_content(
        documents={f"d{index}": f"doc{index:03d}.md" for index in range(document_count)},
        document_by_chunk={f"c{index}": f"d{index}" for index in range(document_count)},
        entities={},
    )

    assert index_texts(content)["root"].endswith(" (+3 more)")


def test_root_document_line_falls_back_to_the_id_of_a_document_with_no_name():
    # A chunk may carry no document_name: it is optional on DocumentChunk, and a
    # document whose name is blank leaves none behind either.
    content = make_content(documents={}, document_by_chunk={"c1": "d1"}, entities={})

    assert index_texts(content)["root"] == "Documents: d1"


# --- write path ------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_changed_nodes_are_written_and_everything_else_is_deleted(monkeypatch):
    """The whole point of deterministic text: a re-run over unchanged data writes nothing."""
    alpha, beta, root = build_context_summaries(two_document_content())
    content = two_document_content(
        existing_summary_texts={
            str(alpha.id): alpha.text,  # recomputed identically
            str(beta.id): "what the last run wrote",  # recomputed differently
            str(root.id): root.text,  # recomputed identically
            "bucket-of-a-deleted-document": "orphan",
        }
    )
    deleted, persisted, edges, routed, stamped = [], [], [], [], []

    async def fake_engine():
        return SimpleNamespace(graph="graph")

    async def fake_delete(engine, node_ids):
        deleted.extend(node_ids)

    async def fake_persist(summaries, ctx):
        persisted.extend(summaries)

    async def fake_edges(edges_by_relationship, engine, ctx=None):
        edges.append(edges_by_relationship)

    async def fake_sync_routes(dataset_id, desired):
        routed.append((dataset_id, [summary.id for summary in desired]))

    async def fake_stamp(graph_engine, dataset_id, buckets):
        stamped.append((graph_engine, dataset_id, [bucket.id for bucket in buckets]))

    monkeypatch.setattr(build_document_index_module, "stamp_bucket_owners", fake_stamp)
    monkeypatch.setattr(build_document_index_module, "get_unified_engine", fake_engine)
    monkeypatch.setattr(build_document_index_module, "delete_context_index_nodes", fake_delete)
    monkeypatch.setattr(build_document_index_module, "persist_context_summaries", fake_persist)
    monkeypatch.setattr(build_document_index_module, "persist_context_index_edges", fake_edges)
    monkeypatch.setattr(build_document_index_module, "sync_dataset_routes", fake_sync_routes)

    changed = await build_document_index_module.build_document_index(content)

    assert [summary.id for summary in changed] == [beta.id]
    assert [summary.id for summary in persisted] == [beta.id]
    assert deleted == ["bucket-of-a-deleted-document"]
    # The routing collection reconciles against the whole desired index, not the diff.
    assert routed == [("ds", [alpha.id, beta.id, root.id])]
    # So does the ownership stamp, over every bucket and never the root.
    assert stamped == [("graph", "ds", [alpha.id, beta.id])]
    # One write carrying both edge sets, for the changed bucket only; the
    # summarizes edge targets the document.
    assert edges == [
        {
            "summarized_in": [(str(beta.id), str(root.id))],
            "summarizes": [(str(beta.id), "d2")],
        }
    ]


# --- ownership ---------------------------------------------------------------


class FakeProvenanceGraph:
    """A graph-provenance graph: the refs each node holds, and the calls made."""

    def __init__(self, refs_by_node):
        self.refs = {node_id: list(refs) for node_id, refs in refs_by_node.items()}
        self.calls = []

    async def get_node_delete_data(self, node_ids):
        return {
            node_id: SimpleNamespace(source_ref_keys=list(self.refs[node_id]))
            for node_id in node_ids
            if node_id in self.refs
        }

    async def attach_node_source_refs(self, node_ids, source_ref_keys, pipeline_run_id=None):
        self.calls.append(("attach", list(node_ids), list(source_ref_keys)))
        for node_id in node_ids:
            self.refs.setdefault(node_id, []).extend(source_ref_keys)

    async def remove_node_source_refs(self, node_ids, source_ref_keys):
        self.calls.append(("remove", list(node_ids), list(source_ref_keys)))
        for node_id in node_ids:
            self.refs[node_id] = [ref for ref in self.refs[node_id] if ref not in source_ref_keys]


def _provenance(monkeypatch, stored: bool):
    async def stores_provenance_in_graph(_graph_engine):
        return stored

    monkeypatch.setattr(persist_module, "stores_provenance_in_graph", stores_provenance_in_graph)


@pytest.mark.asyncio
async def test_each_bucket_ends_up_owned_by_its_document_alone(monkeypatch):
    """The write folds the dataset-level ref in; the stamp swaps it for the document's,
    so forget() removes the bucket with the document. Refs the pass does not own
    (another dataset's, a chunk's) are left alone."""
    _provenance(monkeypatch, stored=True)
    dataset_id, doc_a, doc_b, other_dataset = uuid4(), uuid4(), uuid4(), uuid4()
    content = make_content(
        dataset_id=str(dataset_id),
        documents={str(doc_a): "a.md", str(doc_b): "b.md"},
        document_by_chunk={"c1": str(doc_a), "c2": str(doc_b)},
        entities={"e1": ("One", "Thing", {"c1"}), "e2": ("Two", "Thing", {"c2"})},
    )
    bucket_a, bucket_b, _root = build_context_summaries(content)
    dataset_level = make_source_ref_key(dataset_id, uuid4())  # the write's sentinel ref
    foreign = make_source_ref_key(other_dataset, doc_a)
    chunk_level = make_chunk_source_ref_key(dataset_id, doc_a, uuid4())
    graph = FakeProvenanceGraph(
        {
            str(bucket_a.id): [dataset_level, foreign, chunk_level],
            str(bucket_b.id): [make_source_ref_key(dataset_id, doc_b)],  # already right
        }
    )

    await persist_module.stamp_bucket_owners(graph, str(dataset_id), [bucket_a, bucket_b])

    assert graph.refs[str(bucket_a.id)] == [
        foreign,
        chunk_level,
        make_source_ref_key(dataset_id, doc_a),
    ]
    assert graph.refs[str(bucket_b.id)] == [make_source_ref_key(dataset_id, doc_b)]
    assert graph.calls == [
        ("attach", [str(bucket_a.id)], [make_source_ref_key(dataset_id, doc_a)]),
        ("remove", [str(bucket_a.id)], [dataset_level]),
    ]


@pytest.mark.asyncio
async def test_buckets_sharing_a_stale_ref_give_it_back_in_one_call(monkeypatch):
    _provenance(monkeypatch, stored=True)
    dataset_id, doc_a, doc_b = uuid4(), uuid4(), uuid4()
    content = make_content(
        dataset_id=str(dataset_id),
        documents={str(doc_a): "a.md", str(doc_b): "b.md"},
        document_by_chunk={"c1": str(doc_a), "c2": str(doc_b)},
        entities={"e1": ("One", "Thing", {"c1"}), "e2": ("Two", "Thing", {"c2"})},
    )
    bucket_a, bucket_b, _root = build_context_summaries(content)
    dataset_level = make_source_ref_key(dataset_id, uuid4())
    graph = FakeProvenanceGraph(
        {str(bucket_a.id): [dataset_level], str(bucket_b.id): [dataset_level]}
    )

    await persist_module.stamp_bucket_owners(graph, str(dataset_id), [bucket_a, bucket_b])

    assert [call for call in graph.calls if call[0] == "remove"] == [
        ("remove", [str(bucket_a.id), str(bucket_b.id)], [dataset_level])
    ]


@pytest.mark.asyncio
async def test_a_ledger_graph_is_not_stamped(monkeypatch):
    _provenance(monkeypatch, stored=False)
    dataset_id, doc = uuid4(), uuid4()
    content = make_content(
        dataset_id=str(dataset_id),
        documents={str(doc): "a.md"},
        document_by_chunk={"c1": str(doc)},
        entities={"e1": ("One", "Thing", {"c1"})},
    )
    bucket, _root = build_context_summaries(content)
    graph = FakeProvenanceGraph({str(bucket.id): []})

    await persist_module.stamp_bucket_owners(graph, str(dataset_id), [bucket])

    assert graph.calls == []


@pytest.mark.asyncio
async def test_removing_the_root_deletes_its_node_and_its_vector_row(monkeypatch):
    from cognee.tasks.memify.global_context_index.constants import (
        GLOBAL_CONTEXT_SUMMARY_COLLECTION,
    )
    from cognee.tasks.memify.global_context_index.ids import create_root_summary_id

    dataset_id = uuid4()
    calls = []

    class Graph:
        async def delete_nodes(self, node_ids):
            calls.append(("graph", list(node_ids)))

    class Vector:
        async def delete_data_points(self, collection, ids):
            calls.append(("vector", collection, list(ids)))

    async def fake_engine():
        return SimpleNamespace(graph=Graph(), vector=Vector())

    monkeypatch.setattr(persist_module, "get_unified_engine", fake_engine)

    await persist_module.remove_dataset_root(dataset_id)

    root = create_root_summary_id(str(dataset_id))
    assert calls == [
        ("graph", [str(root)]),
        ("vector", GLOBAL_CONTEXT_SUMMARY_COLLECTION, [root]),
    ]
