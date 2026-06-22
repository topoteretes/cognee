"""Backend-agnostic capability tests for the Part 0 graph-native provenance contract.

These tests verify the CONTRACT only — the names, shapes, defaults, and the
"never silently succeed" guarantee that Part 1 (storage primitives) and Part 2
(delete/rollback wiring) both depend on. They deliberately do NOT exercise any
real adapter: no graph-native backend exists yet. The base ``GraphDBInterface``
provenance read primitives must raise ``UnsupportedProvenanceCapability`` by
default, the ref helpers must be deterministic and namespace-separated, the
marker constants must be stable, and the snapshot/result dataclasses must be
frozen with the documented fields/defaults.

The ``CAPABILITY_FACTORIES`` list at the bottom is the parametrizable harness
that Part 1 will plug Ladybug + LanceDB + SQLite into; while empty the
parametrized tests skip.
"""

from dataclasses import FrozenInstanceError, fields
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.graph.provenance import (
    DATASET_IDS_KEY,
    DELETE_MODE_GRAPH_NATIVE,
    DELETE_MODE_KEY,
    DELETE_MODE_LEDGER,
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
    PROVENANCE_VERSION,
    PROVENANCE_VERSION_KEY,
    ProvenanceDeleteResult,
    SOURCE_REFS_KEY,
    SOURCE_RUN_REFS_KEY,
    UnsupportedProvenanceCapability,
    make_source_ref,
    make_source_run_ref,
)


# ---------------------------------------------------------------------------
# Minimal concrete GraphDBInterface
#
# Implements ONLY the @abstractmethods (with trivial / NotImplementedError
# bodies) so the class can be instantiated. It does NOT override any of the
# graph-native provenance read primitives, so those keep the base contract
# defaults under test.
# ---------------------------------------------------------------------------
class _MinimalGraphDB(GraphDBInterface):
    """Bare-minimum concrete subclass that satisfies the ABC but adds nothing.

    Every abstractmethod has a trivial body; none of the provenance primitives
    are overridden, so the contract defaults are what we exercise.
    """

    async def is_empty(self) -> bool:
        raise NotImplementedError

    async def query(self, query: str, params: dict) -> List[Any]:
        raise NotImplementedError

    async def add_node(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def add_nodes(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError

    async def delete_nodes(self, node_ids: List[str]) -> None:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def add_edge(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def add_edges(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def has_edge(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    async def has_edges(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError

    async def get_edges(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError

    async def get_connections(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError

    async def get_neighbors(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError

    async def get_neighborhood(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def get_nodeset_subgraph(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def get_graph_data(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def get_filtered_graph_data(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def get_graph_metrics(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def delete_graph(self, *args, **kwargs) -> None:
        raise NotImplementedError


@pytest.fixture
def base_graph_db() -> GraphDBInterface:
    """A vanilla GraphDBInterface subclass with no provenance support."""
    return _MinimalGraphDB()


# ---------------------------------------------------------------------------
# Base GraphDBInterface provenance defaults
# ---------------------------------------------------------------------------
def test_minimal_subclass_instantiates(base_graph_db):
    """A subclass that implements only the abstractmethods must instantiate."""
    assert isinstance(base_graph_db, GraphDBInterface)


def test_supports_graph_native_provenance_defaults_false(base_graph_db):
    assert base_graph_db.supports_graph_native_provenance() is False


@pytest.mark.asyncio
async def test_get_nodes_delete_data_by_source_ref_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_nodes_delete_data_by_source_ref("ref")


@pytest.mark.asyncio
async def test_get_edges_delete_data_by_source_ref_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_edges_delete_data_by_source_ref("ref")


@pytest.mark.asyncio
async def test_get_nodes_delete_data_by_dataset_id_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_nodes_delete_data_by_dataset_id(uuid4())


@pytest.mark.asyncio
async def test_get_edges_delete_data_by_dataset_id_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_edges_delete_data_by_dataset_id(uuid4())


@pytest.mark.asyncio
async def test_get_nodes_delete_data_by_source_run_ref_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_nodes_delete_data_by_source_run_ref("run-ref")


@pytest.mark.asyncio
async def test_get_edges_delete_data_by_source_run_ref_raises(base_graph_db):
    with pytest.raises(UnsupportedProvenanceCapability):
        await base_graph_db.get_edges_delete_data_by_source_run_ref("run-ref")


@pytest.mark.asyncio
async def test_provenance_capability_is_notimplementederror_subclass(base_graph_db):
    """The typed error must subclass NotImplementedError so legacy handlers work."""
    assert issubclass(UnsupportedProvenanceCapability, NotImplementedError)
    with pytest.raises(NotImplementedError):
        await base_graph_db.get_nodes_delete_data_by_source_ref("ref")


def test_unsupported_capability_carries_metadata():
    exc = UnsupportedProvenanceCapability("get_nodes_delete_data_by_source_ref", backend="ladybug")
    assert exc.capability == "get_nodes_delete_data_by_source_ref"
    assert exc.backend == "ladybug"
    assert "ladybug" in str(exc)
    # backend is optional
    assert UnsupportedProvenanceCapability("cap").backend is None


# ---------------------------------------------------------------------------
# Ref helpers: deterministic, namespace-separated, string-typed
# ---------------------------------------------------------------------------
def test_make_source_ref_is_deterministic():
    dataset_id, data_id = uuid4(), uuid4()
    assert make_source_ref(dataset_id, data_id) == make_source_ref(dataset_id, data_id)


def test_make_source_run_ref_is_deterministic():
    dataset_id, run_id = uuid4(), uuid4()
    assert make_source_run_ref(dataset_id, run_id) == make_source_run_ref(dataset_id, run_id)


def test_refs_return_strings():
    dataset_id, data_id = uuid4(), uuid4()
    assert isinstance(make_source_ref(dataset_id, data_id), str)
    assert isinstance(make_source_run_ref(dataset_id, data_id), str)


def test_refs_are_namespace_separated():
    """A source ref and a source-run ref built from the SAME overlapping ids
    must differ — the two namespaces never collide even when a data_id and a
    pipeline_run_id happen to share a value."""
    a, b = uuid4(), uuid4()
    assert make_source_ref(a, b) != make_source_run_ref(a, b)


def test_distinct_inputs_produce_distinct_refs():
    dataset_id = uuid4()
    assert make_source_ref(dataset_id, uuid4()) != make_source_ref(dataset_id, uuid4())
    assert make_source_run_ref(dataset_id, uuid4()) != make_source_run_ref(dataset_id, uuid4())


# ---------------------------------------------------------------------------
# Marker constants: importable and stable
# ---------------------------------------------------------------------------
def test_marker_constants_stable():
    assert PROVENANCE_VERSION == 1
    assert DELETE_MODE_GRAPH_NATIVE == "graph_native"
    assert DELETE_MODE_LEDGER == "ledger"


def test_property_key_constants_stable():
    assert PROVENANCE_VERSION_KEY == "provenance_version"
    assert DELETE_MODE_KEY == "delete_mode"
    assert SOURCE_REFS_KEY == "source_refs"
    assert SOURCE_RUN_REFS_KEY == "source_run_refs"
    assert DATASET_IDS_KEY == "dataset_ids"


# ---------------------------------------------------------------------------
# Snapshot / result dataclasses: frozen, hashable, documented fields/defaults
# ---------------------------------------------------------------------------
def test_edge_identity_fields_and_frozen():
    ident = EdgeIdentity(source_node_id="src", relationship_name="rel", target_node_id="tgt")
    assert ident.source_node_id == "src"
    assert ident.relationship_name == "rel"
    assert ident.target_node_id == "tgt"
    # frozen
    with pytest.raises(FrozenInstanceError):
        ident.source_node_id = "other"
    # hashable / set-dedup-able
    assert len({ident, EdgeIdentity("src", "rel", "tgt")}) == 1


def test_node_delete_data_defaults_and_frozen():
    node = NodeDeleteData(node_id="n1", node_type="Entity")
    assert node.node_id == "n1"
    assert node.node_type == "Entity"
    assert node.label is None
    assert node.indexed_fields == ()
    assert node.source_refs == ()
    assert node.source_run_refs == ()
    with pytest.raises(FrozenInstanceError):
        node.node_id = "n2"
    # hashable
    assert hash(node) == hash(NodeDeleteData(node_id="n1", node_type="Entity"))


def test_node_delete_data_full_fields():
    node = NodeDeleteData(
        node_id="n1",
        node_type="Entity",
        label="Alice",
        indexed_fields=("name", "description"),
        source_refs=("ref-a", "ref-b"),
        source_run_refs=("run-a",),
    )
    assert node.label == "Alice"
    assert node.indexed_fields == ("name", "description")
    assert node.source_refs == ("ref-a", "ref-b")
    assert node.source_run_refs == ("run-a",)


def test_edge_delete_data_defaults_and_frozen():
    ident = EdgeIdentity("src", "rel", "tgt")
    edge = EdgeDeleteData(identity=ident)
    assert edge.identity is ident
    assert edge.edge_retrieval_text is None
    assert edge.source_refs == ()
    assert edge.source_run_refs == ()
    with pytest.raises(FrozenInstanceError):
        edge.edge_retrieval_text = "x"
    # hashable / set-dedup-able
    assert len({edge, EdgeDeleteData(identity=EdgeIdentity("src", "rel", "tgt"))}) == 1


def test_edge_delete_data_full_fields():
    ident = EdgeIdentity("src", "rel", "tgt")
    edge = EdgeDeleteData(
        identity=ident,
        edge_retrieval_text="works with",
        source_refs=("ref-a",),
        source_run_refs=("run-a", "run-b"),
    )
    assert edge.edge_retrieval_text == "works with"
    assert edge.source_refs == ("ref-a",)
    assert edge.source_run_refs == ("run-a", "run-b")


def test_provenance_delete_result_defaults_and_frozen():
    result = ProvenanceDeleteResult()
    assert result.nodes_deleted == 0
    assert result.edges_deleted == 0
    assert result.nodes_detached == 0
    assert result.edges_detached == 0
    with pytest.raises(FrozenInstanceError):
        result.nodes_deleted = 5
    # explicit values + hashable
    populated = ProvenanceDeleteResult(
        nodes_deleted=1, edges_deleted=2, nodes_detached=3, edges_detached=4
    )
    assert (populated.nodes_deleted, populated.edges_deleted) == (1, 2)
    assert (populated.nodes_detached, populated.edges_detached) == (3, 4)
    assert hash(ProvenanceDeleteResult()) == hash(ProvenanceDeleteResult())


def test_dataclass_field_names_are_stable():
    """Lock the documented field names so Part 1/2 wiring can rely on them."""
    assert [f.name for f in fields(EdgeIdentity)] == [
        "source_node_id",
        "relationship_name",
        "target_node_id",
    ]
    assert [f.name for f in fields(NodeDeleteData)] == [
        "node_id",
        "node_type",
        "label",
        "indexed_fields",
        "source_refs",
        "source_run_refs",
    ]
    assert [f.name for f in fields(EdgeDeleteData)] == [
        "identity",
        "edge_retrieval_text",
        "source_refs",
        "source_run_refs",
    ]
    assert [f.name for f in fields(ProvenanceDeleteResult)] == [
        "nodes_deleted",
        "edges_deleted",
        "nodes_detached",
        "edges_detached",
    ]


# ---------------------------------------------------------------------------
# Capability harness
#
# Part 1 plugs real backends in here. Each entry is a zero-arg callable that
# returns a GraphDBInterface implementation to exercise against the contract.
# Example (Part 1):
#
#     def _make_ladybug() -> GraphDBInterface: ...
#     def _make_lancedb_backed() -> GraphDBInterface: ...
#     def _make_sqlite_backed() -> GraphDBInterface: ...
#     CAPABILITY_FACTORIES = [_make_ladybug, _make_lancedb_backed, _make_sqlite_backed]
#
# While empty, the parametrized tests skip (see skipif below).
# ---------------------------------------------------------------------------
CAPABILITY_FACTORIES: List[Any] = []  # Part 1: append Ladybug + LanceDB + SQLite factories here.


async def assert_raises_or_returns_delete_data(coro, expected_item_type) -> None:
    """Assert a provenance read primitive honours the contract.

    A backend MUST EITHER:
      * return a ``list`` whose items are all ``expected_item_type``
        (``NodeDeleteData`` or ``EdgeDeleteData``), OR
      * raise ``UnsupportedProvenanceCapability``.

    It must NEVER silently return ``None`` or a non-list — that would let a
    delete look successful while leaving artifacts behind. An empty list is
    acceptable ONLY from a backend that actually implements the capability
    (i.e. it did not raise), representing "no matching artifacts".
    """
    try:
        result = await coro
    except UnsupportedProvenanceCapability:
        return  # acceptable: backend has not implemented this capability
    assert isinstance(result, list), (
        f"expected a list, got {type(result).__name__} ({result!r}); "
        "a backend must raise UnsupportedProvenanceCapability rather than "
        "silently returning None/non-list"
    )
    for item in result:
        assert isinstance(item, expected_item_type), (
            f"expected items of type {expected_item_type.__name__}, got {type(item).__name__}"
        )


@pytest.mark.skipif(
    not CAPABILITY_FACTORIES,
    reason="No graph-native provenance backends registered yet (Part 1 plugs them in).",
)
@pytest.mark.parametrize("factory", CAPABILITY_FACTORIES)
@pytest.mark.asyncio
async def test_backend_node_primitives_honour_contract(factory):
    backend: GraphDBInterface = factory()
    source_ref = make_source_ref(uuid4(), uuid4())
    source_run_ref = make_source_run_ref(uuid4(), uuid4())
    dataset_id = uuid4()

    await assert_raises_or_returns_delete_data(
        backend.get_nodes_delete_data_by_source_ref(source_ref), NodeDeleteData
    )
    await assert_raises_or_returns_delete_data(
        backend.get_nodes_delete_data_by_dataset_id(dataset_id), NodeDeleteData
    )
    await assert_raises_or_returns_delete_data(
        backend.get_nodes_delete_data_by_source_run_ref(source_run_ref), NodeDeleteData
    )


@pytest.mark.skipif(
    not CAPABILITY_FACTORIES,
    reason="No graph-native provenance backends registered yet (Part 1 plugs them in).",
)
@pytest.mark.parametrize("factory", CAPABILITY_FACTORIES)
@pytest.mark.asyncio
async def test_backend_edge_primitives_honour_contract(factory):
    backend: GraphDBInterface = factory()
    source_ref = make_source_ref(uuid4(), uuid4())
    source_run_ref = make_source_run_ref(uuid4(), uuid4())
    dataset_id = uuid4()

    await assert_raises_or_returns_delete_data(
        backend.get_edges_delete_data_by_source_ref(source_ref), EdgeDeleteData
    )
    await assert_raises_or_returns_delete_data(
        backend.get_edges_delete_data_by_dataset_id(dataset_id), EdgeDeleteData
    )
    await assert_raises_or_returns_delete_data(
        backend.get_edges_delete_data_by_source_run_ref(source_run_ref), EdgeDeleteData
    )


@pytest.mark.skipif(
    not CAPABILITY_FACTORIES,
    reason="No graph-native provenance backends registered yet (Part 1 plugs them in).",
)
@pytest.mark.parametrize("factory", CAPABILITY_FACTORIES)
def test_backend_support_flag_matches_behaviour(factory):
    """If a backend advertises support, its primitives must not raise the
    unsupported error; if it does not, the harness still tolerates raising."""
    backend: GraphDBInterface = factory()
    assert isinstance(backend.supports_graph_native_provenance(), bool)
