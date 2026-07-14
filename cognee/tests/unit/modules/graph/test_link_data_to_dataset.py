"""Unit tests for link_data_to_dataset (cross-dataset reuse).

Linking must reproduce exactly what a full pipeline run for the target
dataset would add on top of the already-materialized artifacts: NodeSet
tags on the taggable nodes/rows, belongs_to_set edges to the target's
NodeSet anchors, and provenance ledger rows keyed by (target_dataset, data).
"""

import importlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.modules.data.models import Data
from cognee.modules.graph.models import Edge, Node

link_module = importlib.import_module("cognee.modules.graph.methods.link_data_to_dataset")
link_data_to_dataset = link_module.link_data_to_dataset


class _FakeScalarResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    """Serves queued scalar results (node rows, then edge rows) and records writes."""

    def __init__(self, scalar_results):
        self._scalar_results = list(scalar_results)
        self.executed = []
        self.committed = False

    async def scalars(self, _statement):
        return _FakeScalarResult(self._scalar_results.pop(0))

    async def execute(self, statement, *_args, **_kwargs):
        self.executed.append(statement)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeRelationalEngine:
    def __init__(self, node_rows, edge_rows):
        self._results = [node_rows, edge_rows]
        self.sessions = []

    def get_async_session(self):
        session = _FakeSession([list(self._results[0]), list(self._results[1])])
        self.sessions.append(session)
        return session


class _SupportedGraphEngine:
    def __init__(self):
        self.added_nodes = []
        self.added_edges = []
        self.tag_calls = []

    async def add_nodes(self, nodes, **_kwargs):
        self.added_nodes.extend(nodes)

    async def add_edges(self, edges, **_kwargs):
        self.added_edges.extend(edges)

    async def add_belongs_to_set_tags(self, tags, node_ids):
        self.tag_calls.append((list(tags), list(node_ids)))


class _SupportedVectorEngine:
    def __init__(self, error=None):
        self.tag_calls = []
        self._error = error

    async def add_belongs_to_set_tags(self, tags, node_ids):
        if self._error is not None:
            raise self._error
        self.tag_calls.append((list(tags), list(node_ids)))


class _UnsupportedEngine:
    """No add_belongs_to_set_tags at all — must be detected as unsupported."""


class _EngineHandle:
    """Mimics the deployed engine handles: the adapter's methods never appear
    on the handle's class — attribute access forwards to the live adapter via
    ``__getattr__``, and callables come back wrapped in a plain closure (as the
    leased-cache proxy does), hiding the bound method's ``__func__``."""

    __slots__ = ("_engine",)

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        attr = getattr(self._engine, name)
        if not callable(attr):
            return attr

        def call_with_lease(*args, **kwargs):
            return attr(*args, **kwargs)

        return call_with_lease


class _ProvenanceGraphEngine(_SupportedGraphEngine):
    """Graph engine exposing the graph-native provenance API."""

    def __init__(self, node_properties, edge_identities):
        super().__init__()
        self._node_properties = node_properties
        self._edge_identities = edge_identities
        self.node_ref_attachments = []
        self.edge_ref_attachments = []

    async def find_nodes_by_source_ref(self, source_ref_key):
        self.queried_ref = source_ref_key
        return [properties["id"] for properties in self._node_properties]

    async def extract_nodes(self, node_ids):
        wanted = set(node_ids)
        return [p for p in self._node_properties if str(p["id"]) in wanted]

    async def find_edges_by_source_ref(self, source_ref_key):
        return list(self._edge_identities)

    async def attach_node_source_refs(self, node_ids, source_ref_keys, pipeline_run_id=None):
        self.node_ref_attachments.append((list(node_ids), list(source_ref_keys), pipeline_run_id))

    async def attach_edge_source_refs(self, edges, source_ref_keys, pipeline_run_id=None):
        self.edge_ref_attachments.append((list(edges), list(source_ref_keys), pipeline_run_id))


def _ledger_node(data_id, dataset_id, node_type, attributes, label="node"):
    return Node(
        id=uuid4(),
        slug=uuid4(),
        user_id=uuid4(),
        data_id=data_id,
        dataset_id=dataset_id,
        pipeline_run_id=None,
        label=label,
        type=node_type,
        indexed_fields=["name"],
        attributes=attributes,
    )


def _ledger_edge(data_id, dataset_id, source, destination, relationship_name):
    return Edge(
        id=uuid4(),
        slug=uuid4(),
        user_id=uuid4(),
        data_id=data_id,
        dataset_id=dataset_id,
        pipeline_run_id=None,
        source_node_id=source,
        destination_node_id=destination,
        relationship_name=relationship_name,
        label=relationship_name,
        attributes={"relationship_name": relationship_name},
    )


def _insert_rows(statement):
    """Row dicts of a multi-values INSERT statement, keyed by column name."""
    return [
        {getattr(column, "name", column): value for column, value in row.items()}
        for row in statement._multi_values[0]
    ]


@pytest.fixture
def upsert_recorder(monkeypatch):
    recorder = SimpleNamespace(nodes=[], edges=[])

    async def _fake_upsert_nodes(nodes, **kwargs):
        recorder.nodes.append((nodes, kwargs))

    async def _fake_upsert_edges(edges, **kwargs):
        recorder.edges.append((edges, kwargs))

    monkeypatch.setattr(link_module, "upsert_nodes", _fake_upsert_nodes)
    monkeypatch.setattr(link_module, "upsert_edges", _fake_upsert_edges)
    return recorder


def _setup(
    monkeypatch,
    graph_engine,
    vector_engine,
    relational_engine,
    access_control=False,
    graph_provenance=False,
):
    monkeypatch.setattr(link_module, "backend_access_control_enabled", lambda: access_control)

    async def _get_graph_engine():
        return graph_engine

    async def _get_vector_engine():
        return vector_engine

    async def _stores_provenance(_graph_engine):
        return graph_provenance

    monkeypatch.setattr(link_module, "get_graph_engine", _get_graph_engine)
    monkeypatch.setattr(link_module, "get_vector_engine_async", _get_vector_engine)
    monkeypatch.setattr(link_module, "get_relational_engine", lambda: relational_engine)
    monkeypatch.setattr(link_module, "stores_provenance_in_graph", _stores_provenance)


def _make_case(tags=("dataset_b",)):
    """A source extraction: document + chunk (taggable), entity (untagged),
    the source dataset's NodeSet row, and their edges."""
    data_id = uuid4()
    source_dataset_id = uuid4()

    document = _ledger_node(
        data_id,
        source_dataset_id,
        "TextDocument",
        {"belongs_to_set": ["dataset_a"], "source_node_set": "dataset_a", "name": "doc"},
    )
    chunk = _ledger_node(
        data_id, source_dataset_id, "DocumentChunk", {"belongs_to_set": ["dataset_a"]}
    )
    entity = _ledger_node(data_id, source_dataset_id, "Entity", {"belongs_to_set": None})
    source_node_set = _ledger_node(
        data_id, source_dataset_id, "NodeSet", {"belongs_to_set": None, "name": "dataset_a"}
    )

    edges = [
        _ledger_edge(data_id, source_dataset_id, chunk.slug, document.slug, "is_part_of"),
        _ledger_edge(data_id, source_dataset_id, chunk.slug, entity.slug, "contains"),
        _ledger_edge(
            data_id, source_dataset_id, chunk.slug, source_node_set.slug, "belongs_to_set"
        ),
    ]

    data = Data(id=data_id)
    data.node_set = json.dumps(list(tags)) if tags else None

    return SimpleNamespace(
        data=data,
        source_dataset_id=source_dataset_id,
        target_dataset=SimpleNamespace(id=uuid4(), name="dataset-b"),
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        document=document,
        chunk=chunk,
        entity=entity,
        source_node_set=source_node_set,
        node_rows=[document, chunk, entity, source_node_set],
        edge_rows=edges,
    )


@pytest.mark.asyncio
async def test_links_artifacts_to_target_dataset(monkeypatch, upsert_recorder):
    case = _make_case()
    graph_engine = _SupportedGraphEngine()
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine(case.node_rows, case.edge_rows)
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
        pipeline_run_id=uuid4(),
    )

    assert linked is True

    taggable_slugs = {str(case.document.slug), str(case.chunk.slug)}

    # Vector + graph tagging targets exactly the nodes the pipeline tagged.
    assert (
        vector_engine.tag_calls == [(["dataset_b"], sorted(taggable_slugs))]
        or set(vector_engine.tag_calls[0][1]) == taggable_slugs
    )
    assert vector_engine.tag_calls[0][0] == ["dataset_b"]
    assert graph_engine.tag_calls[0][0] == ["dataset_b"]
    assert set(graph_engine.tag_calls[0][1]) == taggable_slugs

    # The target's NodeSet anchor is created and wired with belongs_to_set edges.
    assert [node.name for node in graph_engine.added_nodes] == ["dataset_b"]
    assert {
        (source, relationship_name) for source, _, relationship_name, _ in graph_engine.added_edges
    } == {(slug, "belongs_to_set") for slug in taggable_slugs}

    # Ledger copies: nodes exclude the source NodeSet row; edges exclude
    # the source belongs_to_set edge; everything is re-keyed to the target.
    write_session = relational_engine.sessions[-1]
    node_inserts = [s for s in write_session.executed if s.table.name == "nodes"]
    edge_inserts = [s for s in write_session.executed if s.table.name == "edges"]
    assert len(node_inserts) == 1 and len(edge_inserts) == 1

    copied_nodes = _insert_rows(node_inserts[0])
    assert {row["slug"] for row in copied_nodes} == {
        case.document.slug,
        case.chunk.slug,
        case.entity.slug,
    }
    assert all(row["dataset_id"] == case.target_dataset.id for row in copied_nodes)

    by_slug = {row["slug"]: row for row in copied_nodes}
    assert by_slug[case.document.slug]["attributes"]["belongs_to_set"] == ["dataset_b"]
    assert by_slug[case.document.slug]["attributes"]["source_node_set"] == "dataset_b"
    assert by_slug[case.chunk.slug]["attributes"]["belongs_to_set"] == ["dataset_b"]
    assert by_slug[case.entity.slug]["attributes"] == {"belongs_to_set": None}

    copied_edges = _insert_rows(edge_inserts[0])
    assert {row["relationship_name"] for row in copied_edges} == {"is_part_of", "contains"}
    assert all(row["dataset_id"] == case.target_dataset.id for row in copied_edges)

    # The target's NodeSet node and belongs_to_set edges get their own
    # ledger rows through the regular upsert helpers.
    assert len(upsert_recorder.nodes) == 1
    assert [node.name for node in upsert_recorder.nodes[0][0]] == ["dataset_b"]
    assert len(upsert_recorder.edges) == 1
    assert {edge[2] for edge in upsert_recorder.edges[0][0]} == {"belongs_to_set"}

    assert write_session.committed


@pytest.mark.asyncio
async def test_no_node_set_still_copies_ledger_without_tagging(monkeypatch, upsert_recorder):
    case = _make_case(tags=())
    graph_engine = _SupportedGraphEngine()
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine(case.node_rows, case.edge_rows)
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is True
    assert vector_engine.tag_calls == []
    assert graph_engine.tag_calls == []
    assert graph_engine.added_nodes == []
    assert upsert_recorder.nodes == []

    write_session = relational_engine.sessions[-1]
    node_inserts = [s for s in write_session.executed if s.table.name == "nodes"]
    assert len(node_inserts) == 1
    # A run without node_set would serialize belongs_to_set as None.
    copied_nodes = _insert_rows(node_inserts[0])
    by_slug = {row["slug"]: row for row in copied_nodes}
    assert by_slug[case.document.slug]["attributes"]["belongs_to_set"] is None
    assert by_slug[case.document.slug]["attributes"]["source_node_set"] is None


@pytest.mark.asyncio
async def test_backend_access_control_disables_linking(monkeypatch, upsert_recorder):
    case = _make_case()
    graph_engine = _SupportedGraphEngine()
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine(case.node_rows, case.edge_rows)
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine, access_control=True)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is False
    assert relational_engine.sessions == []
    assert vector_engine.tag_calls == []


@pytest.mark.asyncio
async def test_unsupported_adapter_disables_linking(monkeypatch, upsert_recorder):
    case = _make_case()
    relational_engine = _FakeRelationalEngine(case.node_rows, case.edge_rows)
    _setup(monkeypatch, _UnsupportedEngine(), _SupportedVectorEngine(), relational_engine)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is False
    assert relational_engine.sessions == [], "no store may be touched when unsupported"


def test_supports_tag_add_resolves_through_engine_handles():
    """Regression: deployed engines are __getattr__-forwarding handles, so the
    adapter's methods never appear on the handle's class; class-level
    inspection reported no tagging support and silently disabled reuse."""
    assert link_module._supports_tag_add(
        _EngineHandle(_SupportedGraphEngine()), _EngineHandle(_SupportedVectorEngine())
    )


def test_supports_tag_add_rejects_interface_defaults():
    """An adapter that inherits the interface's raising default is provably
    unsupported and must still be detected through the instance."""

    class _DefaultTagGraphEngine:
        add_belongs_to_set_tags = GraphDBInterface.add_belongs_to_set_tags

    class _DefaultTagVectorEngine:
        add_belongs_to_set_tags = VectorDBInterface.add_belongs_to_set_tags

    assert not link_module._supports_tag_add(_DefaultTagGraphEngine(), _SupportedVectorEngine())
    assert not link_module._supports_tag_add(_SupportedGraphEngine(), _DefaultTagVectorEngine())


@pytest.mark.asyncio
async def test_links_via_graph_provenance_through_engine_handles(monkeypatch, upsert_recorder):
    """The full link path must work when the engines only expose their API
    through __getattr__-forwarding handles (as get_graph_engine() returns)."""
    case = _make_case()
    doc_id = str(uuid4())
    graph_engine = _ProvenanceGraphEngine(
        [{"id": doc_id, "type": "TextDocument", "belongs_to_set": ["dataset_a"]}], []
    )
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine([], [])
    _setup(
        monkeypatch,
        _EngineHandle(graph_engine),
        _EngineHandle(vector_engine),
        relational_engine,
        graph_provenance=True,
    )

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is True
    assert vector_engine.tag_calls[0][1] == [doc_id]
    assert graph_engine.node_ref_attachments, "target refs must reach the adapter via the handle"


@pytest.mark.asyncio
async def test_missing_ledger_rows_disables_linking(monkeypatch, upsert_recorder):
    case = _make_case()
    graph_engine = _SupportedGraphEngine()
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine([], [])
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is False
    assert vector_engine.tag_calls == []
    assert graph_engine.added_nodes == []


@pytest.mark.asyncio
async def test_links_artifacts_via_graph_provenance(monkeypatch, upsert_recorder):
    """No ledger rows + graph marked as graph-provenance → link by source refs."""
    case = _make_case()

    doc_id, chunk_id, entity_id, node_set_id = (str(uuid4()) for _ in range(4))
    node_properties = [
        {"id": doc_id, "type": "TextDocument", "belongs_to_set": ["dataset_a"]},
        {"id": chunk_id, "type": "DocumentChunk", "belongs_to_set": ["dataset_a"]},
        {"id": entity_id, "type": "Entity", "belongs_to_set": None},
        {"id": node_set_id, "type": "NodeSet", "belongs_to_set": None},
    ]
    edge_identities = [
        EdgeIdentity(source_id=chunk_id, target_id=doc_id, relationship_name="is_part_of"),
        EdgeIdentity(source_id=chunk_id, target_id=entity_id, relationship_name="contains"),
        EdgeIdentity(source_id=chunk_id, target_id=node_set_id, relationship_name="belongs_to_set"),
    ]

    graph_engine = _ProvenanceGraphEngine(node_properties, edge_identities)
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine([], [])  # ledger empty
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine, graph_provenance=True)

    pipeline_run_id = uuid4()
    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
        pipeline_run_id=pipeline_run_id,
    )

    assert linked is True
    assert graph_engine.queried_ref == make_source_ref_key(case.source_dataset_id, case.data.id)

    # Tagging targets the nodes the pipeline tagged (belongs_to_set list),
    # never the source dataset's NodeSet anchor.
    assert vector_engine.tag_calls[0][0] == ["dataset_b"]
    assert set(vector_engine.tag_calls[0][1]) == {doc_id, chunk_id}
    assert set(graph_engine.tag_calls[0][1]) == {doc_id, chunk_id}

    # The target's ref is attached to every artifact except the source
    # anchor / its belongs_to_set edges.
    target_ref = make_source_ref_key(case.target_dataset.id, case.data.id)
    assert len(graph_engine.node_ref_attachments) == 1
    attached_ids, attached_refs, attached_run = graph_engine.node_ref_attachments[0]
    assert set(attached_ids) == {doc_id, chunk_id, entity_id}
    assert attached_refs == [target_ref]
    assert attached_run == str(pipeline_run_id)

    assert len(graph_engine.edge_ref_attachments) == 1
    attached_edges, edge_refs, _ = graph_engine.edge_ref_attachments[0]
    assert {edge.relationship_name for edge in attached_edges} == {"is_part_of", "contains"}
    assert edge_refs == [target_ref]

    # New NodeSet anchor + belongs_to_set edges are created and no
    # relational ledger rows are written in this mode.
    assert [node.name for node in graph_engine.added_nodes] == ["dataset_b"]
    assert {relationship for _, _, relationship, _ in graph_engine.added_edges} == {
        "belongs_to_set"
    }
    assert upsert_recorder.nodes == []
    assert all(not session.executed for session in relational_engine.sessions)


@pytest.mark.asyncio
async def test_graph_provenance_without_stamps_disables_linking(monkeypatch, upsert_recorder):
    case = _make_case()
    graph_engine = _ProvenanceGraphEngine([], [])
    vector_engine = _SupportedVectorEngine()
    relational_engine = _FakeRelationalEngine([], [])
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine, graph_provenance=True)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is False
    assert vector_engine.tag_calls == []
    assert graph_engine.node_ref_attachments == []


@pytest.mark.asyncio
async def test_tagging_failure_returns_false_for_fallback(monkeypatch, upsert_recorder):
    case = _make_case()
    graph_engine = _SupportedGraphEngine()
    vector_engine = _SupportedVectorEngine(error=RuntimeError("vector store down"))
    relational_engine = _FakeRelationalEngine(case.node_rows, case.edge_rows)
    _setup(monkeypatch, graph_engine, vector_engine, relational_engine)

    linked = await link_data_to_dataset(
        data=case.data,
        source_dataset_id=case.source_dataset_id,
        target_dataset=case.target_dataset,
        user=case.user,
    )

    assert linked is False, "any failure must signal fallback to full processing"
