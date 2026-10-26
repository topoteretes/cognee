"""Remote namespacing must preserve every supported writer reference."""

from copy import deepcopy

import pytest

from cognee.tasks.code_graph.extract_code_graph import (
    build_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
    namespace_snapshot_repos,
)


def test_namespace_keeps_id_only_caller_target_and_matched_route_edges():
    facts = [
        {"kind": "symbol", "name": "fetch", "repo": "ui", "id": "a" * 32},
        {"kind": "route", "name": "GET /api", "repo": "server", "id": "b" * 32},
        {
            "kind": "route",
            "name": "GET /api",
            "repo": "ui",
            "id": "c" * 32,
            "props": {
                "role": "client",
                "caller_id": "a" * 32,
                "matched_routes": [{"id": "b" * 32}],
            },
        },
        {
            "kind": "symbol",
            "name": "main",
            "repo": "ui",
            "id": "d" * 32,
            "relations": [{"kind": "calls", "target": "ambiguous", "target_id": "a" * 32}],
        },
    ]
    original = deepcopy(facts)
    namespaces = ["github.com/acme/api", "github.com/another/api"]
    id_sets = []
    for namespace in namespaces:
        rewritten = namespace_snapshot_repos(facts, namespace, "/clones/hashed")
        edges, skipped = build_code_graph_edges(rewritten, repo_name=namespace)
        assert skipped == 0
        triples = {(source, target, relation) for source, target, relation, _ in edges}
        ui = namespace + "::ui"
        server = namespace + "::server"
        assert (
            fact_node_id(ui, "symbol", "fetch"),
            fact_node_id(ui, "route", "GET /api"),
            "makes_request",
        ) in triples
        assert (
            fact_node_id(ui, "symbol", "main"),
            fact_node_id(ui, "symbol", "fetch"),
            "calls",
        ) in triples
        assert any(
            source == fact_node_id(ui, "route", "GET /api")
            and target == fact_node_id(server, "route", "GET /api")
            for source, target, _relation in triples
        )
        points = map_facts_to_data_points(rewritten, repo_name=namespace)
        node_ids = {point.id for point in points}
        assert all(source in node_ids and target in node_ids for source, target, _ in triples)
        id_sets.append(node_ids)
    assert id_sets[0].isdisjoint(id_sets[1])
    assert facts == original


def test_primary_remote_repository_is_stable_with_empty_or_reordered_subrepositories():
    namespace = "github.com/acme/api"
    facts = [
        {"kind": "symbol", "name": "run", "repo": "service"},
        {"kind": "symbol", "name": "render", "repo": "ui"},
    ]
    for snapshot in (facts, facts[::-1], []):
        points = map_facts_to_data_points(
            namespace_snapshot_repos(snapshot, namespace),
            repo_name=namespace,
        )
        assert points[0].id == fact_node_id(namespace, "repository", namespace)
        assert points[0].name == namespace


@pytest.mark.asyncio
async def test_remote_sweep_removes_disappeared_subrepositories_only(tmp_path, monkeypatch):
    import importlib
    from unittest.mock import AsyncMock

    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
    from cognee.tasks.code_graph.extract_code_graph import (
        _sweep_stale_code_graph,
        add_code_graph_data_points,
    )

    engine = LadybugAdapter(db_path=str(tmp_path / "graph"))
    module = importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=engine))
    namespace = "github.com/acme/api"
    other = "github.com/other/api"
    facts = [{"kind": "symbol", "name": "old.run", "repo": "service"}]
    try:
        for remote in (namespace, other):
            await add_code_graph_data_points(
                map_facts_to_data_points(namespace_snapshot_repos(facts, remote), repo_name=remote),
                graph_only=True,
            )
        before, edges = await engine.get_graph_data()
        other_ids = {
            str(node_id) for node_id, props in before if props.get("remote_repository") == other
        }
        assert len(other_ids) == 3
        removed, _, _ = await _sweep_stale_code_graph(
            engine,
            [],
            [],
            None,
            before,
            {(str(source), str(target), relation) for source, target, relation, _ in edges},
            repo_name=namespace,
        )
        assert removed == 2  # vanished subrepository and its symbol, root survives
        after, _ = await engine.get_graph_data()
        assert {str(node_id) for node_id, _ in after} == other_ids | {
            str(fact_node_id(namespace, "repository", namespace))
        }
    finally:
        await engine.close()


def test_namespacing_gives_the_same_enola_id_a_different_identity_per_remote():
    facts = [{"kind": "symbol", "name": "fetch", "repo": "ui", "id": "a" * 32}]

    first = namespace_snapshot_repos(facts, "github.com/acme/api")
    second = namespace_snapshot_repos(facts, "github.com/another/api")

    assert len({facts[0]["id"], first[0]["id"], second[0]["id"]}) == 3
