"""Backend-neutral graph-provenance adapter contract tests.

Add a provider to ``graph_provenance_adapter`` when implementing graph
provenance for a new graph backend. These tests use only public graph adapter
methods and provenance helpers; backend storage details belong in adapter-
specific tests.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.infrastructure.engine import DataPoint

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = pytest.mark.asyncio


def _postgres_graph_url_from_env():
    """Postgres graph connection URL when the configured backend is postgres, else None.

    Driven entirely by ``.env`` (which CI sets): the postgres graph adapter talks
    to the relational engine's database, so the postgres contract params run
    exactly when ``GRAPH_DATABASE_PROVIDER=postgres`` and ``DB_PROVIDER=postgres``
    are configured, and skip on any other stack (keeps kuzu/sqlite CI green).
    """
    from cognee.infrastructure.databases.graph.config import get_graph_config

    if get_graph_config().graph_database_provider != "postgres":
        return None
    from cognee.infrastructure.databases.relational import get_relational_engine

    return get_relational_engine().db_uri


async def _make_postgres_adapter():
    """Fresh-schema Postgres graph adapter, or skip when ``.env`` isn't postgres."""
    url = _postgres_graph_url_from_env()
    if not url:
        pytest.skip("postgres graph backend not configured (set GRAPH_DATABASE_PROVIDER=postgres)")

    from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter
    from cognee.infrastructure.databases.graph.postgres.tables import _meta

    adapter = PostgresAdapter(url)
    try:
        async with adapter.engine.begin() as conn:
            await conn.run_sync(_meta.drop_all)
        await adapter.initialize()
    except Exception as exc:  # pragma: no cover - environment dependent
        await adapter.close()
        pytest.skip(f"postgres graph backend not reachable: {exc}")
    return adapter


async def _make_neo4j_adapter():
    """Fresh (fully wiped) Neo4j graph adapter, or skip when ``.env`` isn't neo4j.

    Driven entirely by ``.env`` (which CI sets): runs exactly when
    ``GRAPH_DATABASE_PROVIDER=neo4j`` is configured and skips on any other stack.
    Each case starts from an empty graph, so the target Neo4j must be a
    disposable test instance — the fixture wipes every node before yielding.
    """
    from cognee.infrastructure.databases.graph.config import get_graph_config

    config = get_graph_config()
    if config.graph_database_provider.lower() != "neo4j":
        pytest.skip("neo4j graph backend not configured (set GRAPH_DATABASE_PROVIDER=neo4j)")
    if not config.graph_database_url:
        pytest.skip("neo4j graph backend URL not configured")

    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    adapter = Neo4jAdapter(
        graph_database_url=config.graph_database_url,
        graph_database_username=config.graph_database_username or None,
        graph_database_password=config.graph_database_password or None,
        graph_database_name=config.graph_database_name or None,
    )
    try:
        await adapter.initialize()
        await adapter.query("MATCH (n) DETACH DELETE n")
    except Exception as exc:  # pragma: no cover - environment dependent
        await adapter.close()
        pytest.skip(f"neo4j graph backend not reachable: {exc}")
    return adapter


class _Ent(DataPoint):
    """Minimal entity DataPoint with an indexed field and tag membership."""

    name: str
    metadata: dict = {"index_fields": ["name"]}


class _Structural(DataPoint):
    """Structural node that declares no index fields."""

    metadata: dict = {"index_fields": []}


@pytest_asyncio.fixture(params=["ladybug", "postgres", "neo4j"])
async def graph_provenance_adapter(request, tmp_path):
    if request.param == "ladybug":
        if not HAS_LADYBUG:
            pytest.skip("ladybug not installed")
        adapter = LadybugAdapter(str(tmp_path / "graph_db"))
    elif request.param == "postgres":
        adapter = await _make_postgres_adapter()
    elif request.param == "neo4j":
        adapter = await _make_neo4j_adapter()
    else:
        raise AssertionError(f"Unknown graph provenance provider: {request.param}")

    try:
        yield adapter
    finally:
        await adapter.close()


async def _seed_two_entities(adapter):
    germany = _Ent(id=uuid4(), name="Germany")
    alice = _Ent(id=uuid4(), name="Alice")
    await adapter.add_nodes([germany, alice])
    return str(germany.id), str(alice.id)


async def test_graph_metadata_round_trip(graph_provenance_adapter):
    adapter = graph_provenance_adapter

    assert await adapter.get_graph_metadata() == {}

    await adapter.set_graph_metadata(
        {
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        }
    )
    meta = await adapter.get_graph_metadata()
    assert meta[GRAPH_DELETE_MODE_KEY] == GRAPH_DELETE_MODE_GRAPH_PROVENANCE
    assert meta[GRAPH_PROVENANCE_VERSION_KEY] == GRAPH_PROVENANCE_VERSION

    assert await adapter.is_empty() is True

    await adapter.set_graph_metadata({"extra": "value"})
    meta = await adapter.get_graph_metadata()
    assert meta[GRAPH_DELETE_MODE_KEY] == GRAPH_DELETE_MODE_GRAPH_PROVENANCE
    assert meta["extra"] == "value"


async def test_attach_node_source_refs_materializes_all_fields(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, d2 = uuid4(), uuid4()
    r1, r2 = uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d2, uuid4())
    germany_id, _ = await _seed_two_entities(adapter)

    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == []
    assert snap.source_dataset_ids == []
    assert snap.source_run_ids == []
    assert snap.source_run_refs == []

    await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key_a]
    assert snap.source_dataset_ids == [str(d1)]
    assert snap.source_run_ids == [str(r1)]
    assert snap.source_run_refs == [make_source_run_ref(r1, key_a)]

    await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert set(snap.source_ref_keys) == {key_a, key_b}
    assert snap.source_dataset_ids == sorted({str(d1), str(d2)})
    assert snap.source_run_ids == sorted({str(r1), str(r2)})

    await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert sorted(snap.source_run_refs) == sorted(
        {make_source_run_ref(r1, key_a), make_source_run_ref(r2, key_b)}
    )

    await adapter.attach_node_source_refs([germany_id], [key_a], str(r2))
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert sorted(snap.source_run_refs) == sorted(
        {make_source_run_ref(r1, key_a), make_source_run_ref(r2, key_b)}
    )
    assert snap.source_run_ids == sorted({str(r1), str(r2)})


async def test_attach_without_pipeline_run_is_not_rollbackable_by_run(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    germany_id, _ = await _seed_two_entities(adapter)

    await adapter.attach_node_source_refs([germany_id], [key], None)
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key]
    assert snap.source_dataset_ids == [str(d1)]
    assert snap.source_run_ids == []
    assert snap.source_run_refs == []
    assert await adapter.find_node_source_refs_by_pipeline_run(str(r1)) == {}


async def test_remove_node_source_refs_updates_derived_and_is_idempotent(
    graph_provenance_adapter,
):
    adapter = graph_provenance_adapter
    d1, d2, r1, r2 = uuid4(), uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d2, uuid4())
    germany_id, _ = await _seed_two_entities(adapter)

    await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
    await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))

    await adapter.remove_node_source_refs([germany_id], [key_a])
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key_b]
    assert snap.source_dataset_ids == [str(d2)]
    assert snap.source_run_ids == [str(r2)]
    assert snap.source_run_refs == [make_source_run_ref(r2, key_b)]
    assert await adapter.has_node(germany_id) is True

    await adapter.remove_node_source_refs([germany_id], [key_a])
    await adapter.remove_node_source_refs(["does-not-exist"], [key_a])
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key_b]


async def test_node_source_ref_lookups_are_exact(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d1, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)

    await adapter.attach_node_source_refs([germany_id], [key_a, key_b], str(r1))
    await adapter.attach_node_source_refs([alice_id], [key_a], str(r1))

    assert sorted(await adapter.find_nodes_by_source_ref(key_b)) == [germany_id]
    assert await adapter.find_nodes_by_source_ref(make_source_ref_key(uuid4(), uuid4())) == []

    by_dataset = await adapter.find_node_source_refs_by_dataset(str(d1))
    assert set(by_dataset[germany_id]) == {key_a, key_b}
    assert by_dataset[alice_id] == [key_a]

    by_run = await adapter.find_node_source_refs_by_pipeline_run(str(r1))
    assert set(by_run[germany_id]) == {key_a, key_b}
    assert by_run[alice_id] == [key_a]


async def test_remove_keeps_dataset_id_when_sibling_ref_shares_dataset(
    graph_provenance_adapter,
):
    adapter = graph_provenance_adapter
    d1, r1, r2 = uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d1, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)

    await adapter.attach_node_source_refs([germany_id], [key_a], str(r1))
    await adapter.attach_node_source_refs([germany_id], [key_b], str(r2))
    await adapter.remove_node_source_refs([germany_id], [key_a])
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key_b]
    assert snap.source_dataset_ids == [str(d1)]

    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
    edge = EdgeIdentity(germany_id, alice_id, "knows")
    await adapter.attach_edge_source_refs([edge], [key_a], str(r1))
    await adapter.attach_edge_source_refs([edge], [key_b], str(r2))
    await adapter.remove_edge_source_refs([edge], [key_a])
    esnap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert esnap.source_ref_keys == [key_b]
    assert esnap.source_dataset_ids == [str(d1)]


async def test_remove_keeps_run_id_when_sibling_ref_shares_run(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, d2, r1 = uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d2, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)

    await adapter.attach_node_source_refs([germany_id], [key_a, key_b], str(r1))
    await adapter.remove_node_source_refs([germany_id], [key_a])
    snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert snap.source_ref_keys == [key_b]
    assert snap.source_run_ids == [str(r1)]
    assert snap.source_run_refs == [make_source_run_ref(r1, key_b)]

    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
    edge = EdgeIdentity(germany_id, alice_id, "knows")
    await adapter.attach_edge_source_refs([edge], [key_a, key_b], str(r1))
    await adapter.remove_edge_source_refs([edge], [key_a])
    esnap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert esnap.source_ref_keys == [key_b]
    assert esnap.source_run_ids == [str(r1)]
    assert esnap.source_run_refs == [make_source_run_ref(r1, key_b)]


async def test_node_delete_data_snapshot_fields(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    germany = _Ent(id=uuid4(), name="Germany")
    ts = _Structural(id=uuid4())
    await adapter.add_nodes([germany, ts])
    germany_id, ts_id = str(germany.id), str(ts.id)

    snaps = await adapter.get_node_delete_data([germany_id, ts_id, "ghost"])
    assert set(snaps.keys()) == {germany_id, ts_id}
    assert snaps[germany_id].node_type == "_Ent"
    assert snaps[germany_id].indexed_fields == ["name"]
    assert snaps[germany_id].node_properties.get("name") == "Germany"
    assert snaps[ts_id].indexed_fields == []

    assert await adapter.get_node_delete_data([]) == {}


async def test_edge_provenance_snapshot_and_lookups(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)

    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "Germany knows Alice"})])
    await adapter.add_edges([(alice_id, germany_id, "likes", {})])
    edge = EdgeIdentity(germany_id, alice_id, "knows")
    edge_no_text = EdgeIdentity(alice_id, germany_id, "likes")

    await adapter.attach_edge_source_refs([edge], [key], str(r1))

    snap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert snap.source_ref_keys == [key]
    assert snap.source_dataset_ids == [str(d1)]
    assert snap.edge_text == "Germany knows Alice"

    snap_fb = (await adapter.get_edge_delete_data([edge_no_text]))[edge_no_text]
    assert snap_fb.edge_text == "likes"

    assert await adapter.find_edges_by_source_ref(key) == [edge]
    assert await adapter.find_edge_source_refs_by_dataset(str(d1)) == {edge: [key]}
    assert await adapter.find_edge_source_refs_by_pipeline_run(str(r1)) == {edge: [key]}

    await adapter.remove_edge_source_refs([edge], [key])
    assert await adapter.find_edges_by_source_ref(key) == []


async def test_delete_edge_triples_preserves_endpoints(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    germany_id, alice_id = await _seed_two_entities(adapter)
    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
    await adapter.add_edges([(germany_id, alice_id, "trusts", {"edge_text": "t2"})])

    await adapter.delete_edge_triples([EdgeIdentity(germany_id, alice_id, "knows")])

    remaining = await adapter.get_edge_delete_data(
        [
            EdgeIdentity(germany_id, alice_id, "knows"),
            EdgeIdentity(germany_id, alice_id, "trusts"),
        ]
    )
    assert EdgeIdentity(germany_id, alice_id, "knows") not in remaining
    assert EdgeIdentity(germany_id, alice_id, "trusts") in remaining
    assert await adapter.has_node(germany_id) is True
    assert await adapter.has_node(alice_id) is True


async def test_remove_belongs_to_set_tags_scoped_and_unscoped(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    shared = _Ent(id=uuid4(), name="Shared", belongs_to_set=["Dev", "DevMirror"])
    same_tag = _Ent(id=uuid4(), name="SameTag", belongs_to_set=["Dev"])
    await adapter.add_nodes([shared, same_tag])
    shared_id, same_tag_id = str(shared.id), str(same_tag.id)

    await adapter.remove_belongs_to_set_tags(["Dev"], node_ids=[shared_id])
    s = (await adapter.get_node_delete_data([shared_id]))[shared_id]
    t = (await adapter.get_node_delete_data([same_tag_id]))[same_tag_id]
    assert s.node_properties.get("belongs_to_set") == ["DevMirror"]
    assert t.node_properties.get("belongs_to_set") == ["Dev"]

    await adapter.remove_belongs_to_set_tags(["Dev"])
    t = (await adapter.get_node_delete_data([same_tag_id]))[same_tag_id]
    assert t.node_properties.get("belongs_to_set") == []

    await adapter.remove_belongs_to_set_tags([])
    s = (await adapter.get_node_delete_data([shared_id]))[shared_id]
    assert s.node_properties.get("belongs_to_set") == ["DevMirror"]


async def test_rewrite_preserves_provenance(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    germany = _Ent(id=uuid4(), name="Germany")
    alice = _Ent(id=uuid4(), name="Alice")
    await adapter.add_nodes([germany, alice])
    germany_id, alice_id = str(germany.id), str(alice.id)

    await adapter.attach_node_source_refs([germany_id], [key], str(r1))
    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
    edge = EdgeIdentity(germany_id, alice_id, "knows")
    await adapter.attach_edge_source_refs([edge], [key], str(r1))

    await adapter.add_nodes([_Ent(id=germany.id, name="Germany UPDATED")])
    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t2"})])

    node_snap = (await adapter.get_node_delete_data([germany_id]))[germany_id]
    assert node_snap.source_ref_keys == [key]
    assert node_snap.node_properties.get("name") == "Germany UPDATED"

    edge_snap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert edge_snap.source_ref_keys == [key]
    assert edge_snap.edge_text == "t2"


async def test_get_edges_created_since_orders_limits_and_returns_endpoint_metadata(
    graph_provenance_adapter,
):
    adapter = graph_provenance_adapter
    germany_id, alice_id = await _seed_two_entities(adapter)
    berlin = _Ent(id=uuid4(), name="Berlin")
    await adapter.add_nodes([berlin])
    berlin_id = str(berlin.id)

    await adapter.add_edges([(germany_id, alice_id, "knows", {"edge_text": "t"})])
    await asyncio.sleep(0.01)
    await adapter.add_edges([(alice_id, berlin_id, "visits", {"edge_text": "t2"})])

    edges, node_map = await adapter.get_edges_created_since(None, 10)
    assert [(edge[0], edge[1], edge[2]) for edge in edges] == [
        (germany_id, alice_id, "knows"),
        (alice_id, berlin_id, "visits"),
    ]
    assert node_map[germany_id]["name"] == "Germany"
    assert node_map[alice_id]["name"] == "Alice"
    assert node_map[berlin_id]["name"] == "Berlin"

    limited_edges, _ = await adapter.get_edges_created_since(None, 1)
    assert [(edge[0], edge[1], edge[2]) for edge in limited_edges] == [
        (germany_id, alice_id, "knows")
    ]

    second_batch, _ = await adapter.get_edges_created_since(edges[0][3], 10)
    assert [(edge[0], edge[1], edge[2]) for edge in second_batch] == [
        (alice_id, berlin_id, "visits")
    ]

    none_after_latest, _ = await adapter.get_edges_created_since(edges[-1][3], 10)
    assert none_after_latest == []


async def test_get_edges_created_since_resumes_inside_same_timestamp_batch(
    graph_provenance_adapter,
):
    """One add_edges batch stamps every edge with the same created_at, so a page
    boundary can land inside the tie group. The keyset cursor (created_at plus
    edge identity) must resume inside the group instead of skipping the rest."""
    adapter = graph_provenance_adapter
    germany_id, _alice_id = await _seed_two_entities(adapter)

    hubs = [_Ent(id=uuid4(), name=f"Hub {i}") for i in range(5)]
    await adapter.add_nodes(hubs)
    hub_ids = [str(hub.id) for hub in hubs]
    # A single batch -> a single created_at stamp shared by all five edges.
    await adapter.add_edges(
        [(germany_id, hub_id, "links", {"edge_text": "t"}) for hub_id in hub_ids]
    )

    collected = []
    since, after_key = None, None
    for _ in range(10):  # bounded: a correct walk needs 4 pages (2+2+1+empty)
        page, _node_map = await adapter.get_edges_created_since(since, 2, after_key=after_key)
        if not page:
            break
        collected.extend(page)
        last_source, last_target, last_relationship, since = page[-1]
        after_key = (last_source, last_target, last_relationship)
    else:
        pytest.fail("keyset pagination did not terminate")

    assert len(collected) == 5, "tie-group edges were skipped or duplicated"
    assert {(edge[0], edge[1], edge[2]) for edge in collected} == {
        (germany_id, hub_id, "links") for hub_id in hub_ids
    }
    assert len({edge[3] for edge in collected}) == 1, "batch should share one created_at"


async def test_add_nodes_folds_provenance_in_one_write(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    node = _Ent(id=uuid4(), name="Germany")

    await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r1))

    node_id = str(node.id)
    snap = (await adapter.get_node_delete_data([node_id]))[node_id]
    assert snap.source_ref_keys == [key]
    assert snap.source_dataset_ids == [str(d1)]
    assert snap.source_run_ids == [str(r1)]
    assert snap.source_run_refs == [make_source_run_ref(r1, key)]
    assert await adapter.find_nodes_by_source_ref(key) == [node_id]


async def test_add_edges_folds_provenance_in_one_write(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, r1 = uuid4(), uuid4()
    key = make_source_ref_key(d1, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)

    await adapter.add_edges(
        [(germany_id, alice_id, "knows", {"edge_text": "t"})],
        source_ref_key=key,
        pipeline_run_id=str(r1),
    )

    edge = EdgeIdentity(germany_id, alice_id, "knows")
    snap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert snap.source_ref_keys == [key]
    assert snap.source_dataset_ids == [str(d1)]
    assert snap.source_run_ids == [str(r1)]
    assert snap.source_run_refs == [make_source_run_ref(r1, key)]
    assert await adapter.find_edges_by_source_ref(key) == [edge]


async def test_add_edges_folds_multiple_owners(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    d1, d2, r1, r2 = uuid4(), uuid4(), uuid4(), uuid4()
    key_a = make_source_ref_key(d1, uuid4())
    key_b = make_source_ref_key(d2, uuid4())
    germany_id, alice_id = await _seed_two_entities(adapter)
    edge = EdgeIdentity(germany_id, alice_id, "knows")

    await adapter.add_edges(
        [(germany_id, alice_id, "knows", {"edge_text": "t"})],
        source_ref_key=key_a,
        pipeline_run_id=str(r1),
    )
    await adapter.add_edges(
        [(germany_id, alice_id, "knows", {"edge_text": "t2"})],
        source_ref_key=key_b,
        pipeline_run_id=str(r2),
    )

    snap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert snap.source_ref_keys == [key_a, key_b]
    assert sorted(snap.source_dataset_ids) == sorted({str(d1), str(d2)})
    assert sorted(snap.source_run_ids) == sorted({str(r1), str(r2)})
    assert sorted(snap.source_run_refs) == sorted(
        {make_source_run_ref(r1, key_a), make_source_run_ref(r2, key_b)}
    )


async def test_folded_attach_omitted_when_no_source_ref(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    node = _Ent(id=uuid4(), name="Germany")
    await adapter.add_nodes([node])
    node_id = str(node.id)
    snap = (await adapter.get_node_delete_data([node_id]))[node_id]
    assert snap.source_ref_keys == []
    assert snap.source_run_refs == []


async def test_folded_attach_preserves_model_a_on_node_and_edge_reattach(
    graph_provenance_adapter,
):
    adapter = graph_provenance_adapter
    key = make_source_ref_key(uuid4(), uuid4())
    r1, r2 = uuid4(), uuid4()
    node = _Ent(id=uuid4(), name="Shared")
    node_id = str(node.id)

    await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r1))
    await adapter.add_nodes([node], source_ref_key=key, pipeline_run_id=str(r2))

    snap = (await adapter.get_node_delete_data([node_id]))[node_id]
    assert snap.source_ref_keys == [key]
    assert snap.source_run_refs == [make_source_run_ref(r1, key)]
    assert snap.source_run_ids == [str(r1)]

    germany_id, alice_id = await _seed_two_entities(adapter)
    edge = EdgeIdentity(germany_id, alice_id, "knows")
    await adapter.add_edges(
        [(germany_id, alice_id, "knows", {"edge_text": "t"})],
        source_ref_key=key,
        pipeline_run_id=str(r1),
    )
    await adapter.add_edges(
        [(germany_id, alice_id, "knows", {"edge_text": "t2"})],
        source_ref_key=key,
        pipeline_run_id=str(r2),
    )

    edge_snap = (await adapter.get_edge_delete_data([edge]))[edge]
    assert edge_snap.source_ref_keys == [key]
    assert edge_snap.source_run_refs == [make_source_run_ref(r1, key)]
    assert edge_snap.source_run_ids == [str(r1)]


async def test_concurrent_folded_attach_keeps_all_keys(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    ds = uuid4()
    run = str(uuid4())
    key1 = make_source_ref_key(ds, uuid4())
    key2 = make_source_ref_key(ds, uuid4())
    node = _Ent(id=uuid4(), name="Alice")
    node_id = str(node.id)

    await asyncio.gather(
        adapter.add_nodes([node], source_ref_key=key1, pipeline_run_id=run),
        adapter.add_nodes([node], source_ref_key=key2, pipeline_run_id=run),
    )

    snap = (await adapter.get_node_delete_data([node_id]))[node_id]
    assert sorted(snap.source_ref_keys) == sorted([key1, key2])
    assert sorted(snap.source_run_refs) == sorted(
        [make_source_run_ref(UUID(run), key1), make_source_run_ref(UUID(run), key2)]
    )


async def test_concurrent_explicit_attach_keeps_all_keys(graph_provenance_adapter):
    adapter = graph_provenance_adapter
    ds = uuid4()
    run = str(uuid4())
    key1 = make_source_ref_key(ds, uuid4())
    key2 = make_source_ref_key(ds, uuid4())
    node = _Ent(id=uuid4(), name="Alice")
    node_id = str(node.id)

    await adapter.add_nodes([node])
    await asyncio.gather(
        adapter.attach_node_source_refs([node_id], [key1], run),
        adapter.attach_node_source_refs([node_id], [key2], run),
    )

    snap = (await adapter.get_node_delete_data([node_id]))[node_id]
    assert sorted(snap.source_ref_keys) == sorted([key1, key2])
    assert sorted(snap.source_run_refs) == sorted(
        [make_source_run_ref(UUID(run), key1), make_source_run_ref(UUID(run), key2)]
    )
