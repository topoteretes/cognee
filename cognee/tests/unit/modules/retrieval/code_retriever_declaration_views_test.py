"""CODE name lookups on a name a framework extractor gave a symbol and a view of it.

An Eloquent model is a class ``symbol`` plus a ``storage`` fact named exactly
like it, and a relation method is a method ``symbol`` plus an ``association``
named exactly like it (enola >= 0.4.29; JPA entities and Rails models get a
storage twin too). The graph here is built from such facts by the code-graph
mapping and edge builder, so each lookup sees what ingestion stores.
"""

from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.code_retriever import (
    CodeRetriever,
    CodeSeedNotFoundError,
    invalidate_code_graph_snapshot_cache,
)
from cognee.tasks.code_graph.extract_code_graph import (
    build_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
)

REPO = "lv"
MODEL = "App\\Models\\User"
MODEL_FILE = "app/Models/User.php"
RELATION = "App\\Models\\User::posts"
SHOW = "App\\Http\\Controllers\\UserController::show"
CONTROLLER_FILE = "app/Http/Controllers/UserController.php"
ROUTE = "GET /users/{user}"


@pytest.fixture(autouse=True)
def clear_code_graph_snapshot_cache():
    invalidate_code_graph_snapshot_cache(all_entries=True)
    yield
    invalidate_code_graph_snapshot_cache(all_entries=True)


def _model(repo=REPO, *, storage_file=MODEL_FILE):
    """The class and its storage twin, the relation method and its association twin."""
    return [
        {
            "kind": "symbol",
            "name": MODEL,
            "file": MODEL_FILE,
            "line": 8,
            "repo": repo,
            "props": {"symbol_kind": "class", "eloquent_table": "users"},
        },
        {
            "kind": "storage",
            "name": MODEL,
            "file": storage_file,
            "line": 8,
            "repo": repo,
            "props": {"framework": "eloquent", "storage_kind": "model", "table": "users"},
        },
        {
            "kind": "symbol",
            "name": RELATION,
            "file": MODEL_FILE,
            "line": 14,
            "repo": repo,
            "props": {"symbol_kind": "method", "eloquent_relation": "hasMany"},
        },
        {
            "kind": "association",
            "name": RELATION,
            "file": MODEL_FILE,
            "line": 14,
            "repo": repo,
            "props": {"framework": "eloquent", "model": MODEL, "target": "App\\Models\\Post"},
        },
    ]


def _app():
    """A route, its controller method (calling into the model) and the route's flow."""
    entry = {"name": SHOW, "file": CONTROLLER_FILE, "line": 12}
    return [
        {
            "kind": "symbol",
            "name": SHOW,
            "file": CONTROLLER_FILE,
            "line": 12,
            "repo": REPO,
            "props": {"symbol_kind": "method"},
            # No target_id: enola leaves it off a name two facts share.
            "relations": [
                {"kind": "calls", "target": MODEL},
                {"kind": "instantiates", "target": MODEL},
                {"kind": "calls", "target": RELATION},
            ],
        },
        {
            "kind": "route",
            "name": ROUTE,
            "file": "routes/web.php",
            "line": 5,
            "repo": REPO,
            "props": {"method": "GET"},
            "relations": [{"kind": "handled_by", "target": SHOW}],
        },
        {
            "kind": "flow",
            "name": ROUTE,
            "file": CONTROLLER_FILE,
            "line": 12,
            "repo": REPO,
            "props": {"mechanism": "route", "rank": 1, "entry": entry, "metrics": {"reach": 3}},
            "relations": [{"kind": "flow_entry", "target": SHOW}],
        },
    ]


def _graph(facts):
    """(nodes, edges) the way the graph adapter hands them back after ingestion."""
    edges, _skipped = build_code_graph_edges(facts)
    nodes = [
        (str(data_point.id), data_point.model_dump(mode="json", exclude={"part_of"}))
        for data_point in map_facts_to_data_points(facts)
    ]
    return nodes, [(str(source), str(target), name, props) for source, target, name, props in edges]


async def _run(config, query="", facts=None):
    nodes, edges = _graph([*_model(), *_app()] if facts is None else facts)
    invalidate_code_graph_snapshot_cache(all_entries=True)
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    with patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    ):
        return await CodeRetriever(config=config).get_retrieved_objects(query)


def _id(kind, name, repo=REPO):
    return str(fact_node_id(repo, kind, name))


USER = _id("symbol", MODEL)
USER_STORAGE = _id("storage", MODEL)
POSTS = _id("symbol", RELATION)
POSTS_ASSOCIATION = _id("association", RELATION)


@pytest.mark.asyncio
async def test_explore_and_neighbors_anchor_a_model_name_on_its_class():
    explored = await _run({"operation": "explore", "name": MODEL})
    assert explored["focus"]["id"] == USER
    assert explored["focus"]["type"] == "CodeSymbol"
    assert {node["name"] for node in explored["nodes"]} == {MODEL, SHOW}

    neighbors = await _run({"operation": "neighbors"}, MODEL)
    assert neighbors["anchor"]["id"] == USER
    assert [
        (entry["relation"], entry["direction"], entry["node"]["name"])
        for entry in neighbors["neighbors"]
    ] == [("calls", "in", SHOW), ("instantiates", "in", SHOW)]

    method = await _run({"operation": "neighbors", "name": RELATION})
    assert method["anchor"]["id"] == POSTS
    assert [entry["node"]["name"] for entry in method["neighbors"]] == [SHOW]


@pytest.mark.asyncio
async def test_impact_traverse_and_paths_seed_the_class():
    impact = await _run({"operation": "impact_analysis", "target": MODEL})
    assert [target["id"] for target in impact["targets"]] == [USER]
    assert {
        depth: [node["name"] for node in nodes] for depth, nodes in impact["by_depth"].items()
    } == {
        "1": [SHOW],
        "2": [ROUTE],
    }
    assert impact["total_dependents"] == 2

    callers = await _run({"operation": "traverse", "start": MODEL, "direction": "reverse"})
    assert [node["name"] for node in callers["nodes"]] == [MODEL, SHOW, ROUTE]

    path = await _run({"operation": "find_path", "source": ROUTE, "target": MODEL})
    assert path["found"] is True
    assert [step["name"] for step in path["path"]] == [ROUTE, SHOW, MODEL]
    assert path["to"]["id"] == USER


@pytest.mark.asyncio
async def test_subgraph_and_flows_through_take_the_class_and_the_method():
    subgraph = await _run({"operation": "subgraph", "seeds": [MODEL, RELATION]})
    assert [seed["id"] for seed in subgraph["seeds"]] == [USER, POSTS]
    assert subgraph["seeds_missing"] == []

    through = await _run({"operation": "flows", "symbol": MODEL})
    assert through["symbol"]["id"] == USER
    assert [(flow["name"], flow["via"], flow["distance"]) for flow in through["flows"]] == [
        (ROUTE, "reach", 1)
    ]


@pytest.mark.asyncio
async def test_query_facts_lists_every_fact_of_the_name_and_the_edges_into_the_class():
    named = await _run({"operation": "query_facts", "names": [MODEL]})
    # A filter, not a lookup: both facts of the name, the class first.
    assert [(fact["type"], fact["id"]) for fact in named["facts"]] == [
        ("CodeSymbol", USER),
        ("StorageResource", USER_STORAGE),
    ]

    caller = await _run({"operation": "query_facts", "names": [SHOW]})
    assert [
        (relation["type"], relation["target_id"]) for relation in caller["facts"][0]["relations"]
    ] == [
        ("calls", USER),
        ("calls", POSTS),
        ("instantiates", USER),
    ]


@pytest.mark.asyncio
async def test_search_and_ask_give_the_name_bonus_to_the_declaration():
    hits = (await _run({"operation": "search", "text": MODEL}))["hits"]
    assert hits[0]["id"] == USER
    view_score = next(hit["score"] for hit in hits if hit["id"] == USER_STORAGE)
    # With the class filtered out the name means the storage, which keeps the bonus.
    storage_only = await _run({"operation": "search", "text": MODEL, "kinds": ["storage"]})
    assert [hit["id"] for hit in storage_only["hits"]] == [USER_STORAGE]
    assert storage_only["hits"][0]["score"] - view_score == pytest.approx(5.0, abs=1e-3)

    # The method and its association tie on every text field; the method wins.
    every_kind = await _run({"operation": "search", "text": RELATION, "kinds": "all"})
    assert [hit["id"] for hit in every_kind["hits"][:2]] == [POSTS, POSTS_ASSOCIATION]
    asked = await _run({"operation": "ask", "question": RELATION, "kinds": "all"})
    assert asked["seeds"][0]["id"] == POSTS


@pytest.mark.asyncio
async def test_two_same_named_symbols_stay_ambiguous():
    facts = [*_model(), *_model("admin")]

    with pytest.raises(CodeSeedNotFoundError) as error:
        await _run({"operation": "explore", "name": MODEL}, facts=facts)
    assert (error.value.error_code, error.value.status_code) == ("ambiguous", 422)

    with pytest.raises(CodeSeedNotFoundError) as error:
        await _run({"operation": "neighbors", "name": MODEL}, facts=facts)
    assert error.value.error_code == "ambiguous"

    subgraph = await _run(
        {"operation": "subgraph", "seeds": [MODEL, RELATION], "repo": "admin"}, facts=facts
    )
    assert [seed["id"] for seed in subgraph["seeds"]] == [
        _id("symbol", MODEL, "admin"),
        _id("symbol", RELATION, "admin"),
    ]

    scoped = await _run({"operation": "explore", "name": MODEL, "repo": "admin"}, facts=facts)
    assert scoped["focus"]["id"] == _id("symbol", MODEL, "admin")


@pytest.mark.asyncio
async def test_a_same_named_storage_from_another_file_stays_ambiguous():
    facts = [*_model(storage_file="database/migrations/2024_01_01_create_users_table.php")]

    with pytest.raises(CodeSeedNotFoundError) as error:
        await _run({"operation": "explore", "name": MODEL}, facts=facts)
    assert error.value.error_code == "ambiguous"

    subgraph = await _run({"operation": "subgraph", "seeds": [MODEL, RELATION]}, facts=facts)
    assert [seed["id"] for seed in subgraph["seeds"]] == [POSTS]
    (missing,) = subgraph["seeds_missing"]
    assert missing["reason"] == "ambiguous"
    assert [candidate["id"] for candidate in missing["candidates"]] == [USER, USER_STORAGE]
