"""The CODE navigation operations: capabilities, search, neighbors, subgraph, flows, hubs, ask.

The fixture is a small shop repository the way the code-graph pipeline stores
it: documented symbols (a top-level ``doc`` plus the ``doc`` prop, or only the
prop for snapshots loaded before docs were read), a route, containment edges,
and CodeFlow rows from Enola's flows.json linked to their entry, stages and
routes by flow_entry / flow_stage / flow_route edges.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.code_retriever import (
    CodeGraphUnavailableError,
    CodeRetriever,
    CodeSearchValidationError,
    CodeSeedNotFoundError,
    _CodeGraphSnapshot,
    invalidate_code_graph_snapshot_cache,
)

REPO = "shop"
E_CREATE = "1" * 32
E_VALIDATE = "2" * 32
E_PLACE = "3" * 32
E_CHARGE = "4" * 32
E_CHARGE_ALIAS = "5" * 32  # a second enola id that collapsed onto the charge_card node
E_ROUTE = "6" * 32
E_MAIN = "7" * 32
E_VENDOR = "8" * 32  # a stage Enola names that the graph does not hold
F_ORDERS = "a" * 32
F_MAIN = "b" * 32
F_SEED = "c" * 32

VALIDATE_DOC = "Validates a bearer token and returns its claims."
CREATE_DOC = "Creates an order after checking the caller's token."


@pytest.fixture(autouse=True)
def clear_code_graph_snapshot_cache():
    invalidate_code_graph_snapshot_cache(all_entries=True)
    yield
    invalidate_code_graph_snapshot_cache(all_entries=True)


def _symbol(name, file_path, line, *, doc=None, prop_doc_only=False, enola_id=None, **extra):
    properties = {"symbol_kind": extra.pop("symbol_kind", "function"), "language": "python"}
    node = {
        "name": name,
        "type": "CodeSymbol",
        "kind": "symbol",
        "file_path": file_path,
        "line": line,
        "repo": REPO,
        "symbol_kind": properties["symbol_kind"],
        "fact_properties": properties,
        **extra,
    }
    if doc is not None:
        properties["doc"] = doc
        node["description"] = doc
        if not prop_doc_only:
            node["doc"] = doc
    if enola_id is not None:
        node["enola_id"] = enola_id
    return node


def _ref(name, enola_id, file_path, line, **extra):
    return {"name": name, "id": enola_id, "file": file_path, "line": line, **extra}


def _flow(name, enola_id, *, rank, mechanism, entry, stages=(), routes=(), side=False, **metrics):
    summary = f"The program starts at {name}, which hands off to its stages."
    properties = {
        "mechanism": mechanism,
        "rank": rank,
        "summary": summary,
        "metrics": {"reach": 5, "modules": 3, "max_depth": 2, "unresolved_calls": 0, **metrics},
        "modules": [{"name": "orders", "symbols": 3}, {"name": "auth", "symbols": 2}],
        "entry": entry,
    }
    if stages:
        properties["stages"] = list(stages)
    if routes:
        properties["routes"] = list(routes)
    if side:
        properties["side"] = True
    return {
        "name": name,
        "type": "CodeFlow",
        "kind": "flow",
        "file_path": entry["file"],
        "line": entry["line"],
        "repo": REPO,
        "enola_id": enola_id,
        "description": summary,
        # Some adapters hand nested properties back as JSON text.
        "fact_properties": json.dumps(properties),
    }


def _shop_graph(*, with_flows=True, with_docs=True):
    def doc(text):
        return text if with_docs else None

    nodes = [
        ("repo", {"name": REPO, "type": "CodeRepository", "path": "/src/shop"}),
        ("m-auth", {"name": "auth", "type": "CodeModule", "file_path": "auth", "repo": REPO}),
        ("m-orders", {"name": "orders", "type": "CodeModule", "file_path": "orders", "repo": REPO}),
        (
            "validate",
            _symbol(
                "auth/tokens.validate_token",
                "auth/tokens.py",
                12,
                doc=doc(VALIDATE_DOC),
                enola_id=E_VALIDATE,
                end_line=30,
            ),
        ),
        ("parse", _symbol("auth/tokens.parse_header", "auth/tokens.py", 40)),
        (
            "session",
            _symbol(
                "auth/session.SessionStore",
                "auth/session.py",
                5,
                doc=doc("Keeps user sessions in Redis."),
                prop_doc_only=True,
                symbol_kind="class",
            ),
        ),
        (
            "create",
            _symbol(
                "orders/api.create_order",
                "orders/api.py",
                8,
                doc=doc(CREATE_DOC),
                enola_id=E_CREATE,
            ),
        ),
        (
            "place",
            _symbol(
                "orders/service.place_order",
                "orders/service.py",
                20,
                doc=doc("Places an order and charges the card."),
                enola_id=E_PLACE,
            ),
        ),
        ("save", _symbol("orders/repository.save", "orders/repository.py", 3)),
        (
            "charge",
            _symbol(
                "billing.charge_card",
                "billing/stripe.py",
                7,
                doc=doc("Charges a card through the payment provider."),
                enola_id=E_CHARGE,
            ),
        ),
        (
            "report",
            _symbol(
                "orders/report.render_report",
                "orders/report.py",
                2,
                doc=doc("Renders the monthly sales report."),
            ),
        ),
        ("camel", _symbol("web/auth.validateToken", "web/auth.ts", 4)),
        (
            "test",
            _symbol("tests/test_tokens.test_validate_token", "tests/test_tokens.py", 3),
        ),
        ("cli", _symbol("cli.main", "cli.py", 1, enola_id=E_MAIN)),
        (
            "route",
            {
                "name": "POST /orders",
                "type": "ApiEndpoint",
                "file_path": "orders/api.py",
                "line": 7,
                "repo": REPO,
                "enola_id": E_ROUTE,
                "fact_properties": {"method": "POST"},
            },
        ),
        (
            "table",
            {
                "name": "orders_table",
                "type": "StorageResource",
                "file_path": "orders/repository.py",
                "repo": REPO,
            },
        ),
        ("noise", {"name": "ordinary text", "type": "DocumentChunk", "text": "token"}),
    ]
    edges = [
        ("route", "create", "handled_by", {}),
        ("create", "validate", "calls", {}),
        ("create", "place", "calls", {}),
        ("place", "save", "calls", {}),
        ("place", "charge", "calls", {}),
        ("validate", "parse", "calls", {}),
        ("test", "validate", "calls", {}),
        ("cli", "report", "calls", {}),
        ("save", "table", "depends_on", {}),
        ("validate", "m-auth", "declares", {}),
        ("parse", "m-auth", "declares", {}),
        ("create", "m-orders", "declares", {}),
        ("place", "m-orders", "declares", {}),
        ("create", "repo", "part_of", {}),
    ]
    if with_flows:
        orders_entry = _ref("orders/api.create_order", E_CREATE, "orders/api.py", 8, doc=CREATE_DOC)
        nodes += [
            (
                "flow-orders",
                _flow(
                    "POST /orders",
                    F_ORDERS,
                    rank=1,
                    mechanism="route",
                    entry=orders_entry,
                    stages=[
                        {
                            **_ref("orders/service.place_order", E_PLACE, "orders/service.py", 20),
                            "order": 1,
                            "depth": 1,
                            "reach": 3,
                        },
                        {
                            # Enola's id for this stage is one that collapsed onto
                            # the node; only the flow_stage edge names it.
                            **_ref("billing.charge_card", E_CHARGE_ALIAS, "billing/stripe.py", 7),
                            "order": 2,
                            "depth": 2,
                            "parent": E_PLACE,
                            "reach": 1,
                        },
                        {
                            **_ref("auth/tokens.validate_token", E_VALIDATE, "auth/tokens.py", 12),
                            "order": 3,
                            "depth": 1,
                            "reach": 2,
                        },
                        {
                            **_ref("vendor.retry", E_VENDOR, "vendor/retry.py", 9),
                            "order": 4,
                            "depth": 2,
                            "parent": E_PLACE,
                            "reach": 1,
                        },
                    ],
                    routes=[_ref("POST /orders", E_ROUTE, "orders/api.py", 7, method="POST")],
                    unresolved_calls=4,
                ),
            ),
            (
                "flow-main",
                _flow(
                    "cli.main",
                    F_MAIN,
                    rank=2,
                    mechanism="main",
                    entry=_ref("cli.main", E_MAIN, "cli.py", 1),
                ),
            ),
            (
                "flow-seed",
                _flow(
                    "scripts/seed.main",
                    F_SEED,
                    rank=3,
                    mechanism="main",
                    side=True,
                    entry=_ref("scripts/seed.main", "9" * 32, "scripts/seed.py", 1),
                ),
            ),
        ]
        edges += [
            ("flow-orders", "create", "flow_entry", {}),
            ("flow-orders", "place", "flow_stage", {}),
            ("flow-orders", "charge", "flow_stage", {}),
            ("flow-orders", "validate", "flow_stage", {}),
            ("flow-orders", "route", "flow_route", {}),
            ("flow-main", "cli", "flow_entry", {}),
        ]
    return nodes, edges


def _patch_graph(nodes, edges):
    # Every patch serves its own graph; a snapshot cached from an earlier one
    # in the same test would otherwise answer instead.
    invalidate_code_graph_snapshot_cache(all_entries=True)
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=(nodes, edges))
    return patch(
        "cognee.modules.retrieval.code_retriever.get_graph_engine",
        AsyncMock(return_value=engine),
    )


async def _run(config, query="", **graph_options):
    with _patch_graph(*_shop_graph(**graph_options)):
        return await CodeRetriever(config=config).get_retrieved_objects(query)


# --- capabilities --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_reports_operations_features_counts_and_limits():
    nodes, edges = _shop_graph()
    nodes[0][1]["last_receipt"] = json.dumps({"enola_version": "0.5.1", "format_version": 1})
    with _patch_graph(nodes, edges):
        result = await CodeRetriever(config={"operation": "capabilities"}).get_retrieved_objects("")

    assert result["operation"] == "capabilities"
    assert result["version"] == 1
    assert result["operations"] == sorted(result["operations"])
    assert {"ask", "search", "neighbors", "subgraph", "flows", "hubs", "capabilities"} <= set(
        result["operations"]
    )
    assert {"query_facts", "explore", "impact_analysis", "delta"} <= set(result["operations"])
    assert result["features"] == {"docs": True, "flows": True}
    assert result["counts"] == {
        "nodes": 15,
        "edges": 13,
        "symbols": 11,
        "symbols_with_docs": 6,
        "flows": 3,
        "repositories": 1,
    }
    assert result["enola_versions"] == ["0.5.1"]
    assert result["limits"]["subgraph"] == {
        "seeds": 20,
        "depth": {"default": 1, "max": 3},
        "max_nodes": {"default": 100, "max": 500},
        "max_edges": {"default": 300, "max": 1000},
    }
    assert result["limits"]["ask"]["max_citations"] == {"default": 25, "max": 60}
    assert result["limits"]["search"] == {"limit": {"default": 10, "max": 50}}


@pytest.mark.asyncio
async def test_capabilities_never_raises_for_an_empty_graph():
    with _patch_graph([], []):
        result = await CodeRetriever(config={"operation": "capabilities"}).get_retrieved_objects("")

    assert result["features"] == {"docs": False, "flows": False}
    assert set(result["counts"].values()) == {0}
    assert result["enola_versions"] == []


# --- search --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_ranks_the_documented_token_validator_first():
    result = await _run({"operation": "search", "text": "validate token"})

    assert result["operation"] == "search"
    assert result["query"] == "validate token"
    assert result["terms"] == ["valid", "token"]
    top = result["hits"][0]
    assert top["name"] == "auth/tokens.validate_token"
    assert top == {
        "id": "validate",
        "kind": "symbol",
        "type": "CodeSymbol",
        "name": "auth/tokens.validate_token",
        "file": "auth/tokens.py",
        "line": 12,
        "end_line": 30,
        "repo": REPO,
        "symbol_kind": "function",
        "doc": VALIDATE_DOC,
        "score": top["score"],
        "matched": ["name", "doc", "file"],
    }
    scores = {hit["name"]: hit["score"] for hit in result["hits"]}
    assert all(round(score, 4) == score for score in scores.values())
    # Unrelated symbols either do not match or rank below the validator.
    assert "orders/report.render_report" not in scores
    assert scores["auth/tokens.validate_token"] > scores["auth/tokens.parse_header"]
    # Test code is left out unless asked for.
    assert "tests/test_tokens.test_validate_token" not in scores
    assert result["total"] == len(result["hits"]) and result["has_more"] is False


@pytest.mark.asyncio
async def test_search_splits_camel_and_snake_case_queries_into_the_same_terms():
    camel = await _run({"operation": "search"}, "validateToken")
    snake = await _run({"operation": "search", "q": "validate_token"})
    spaced = await _run({"operation": "search", "query": "validate token"})

    assert camel["terms"] == snake["terms"] == spaced["terms"] == ["valid", "token"]
    names = [hit["name"] for hit in camel["hits"]]
    assert {"auth/tokens.validate_token", "web/auth.validateToken"} <= set(names)
    assert snake["hits"][0]["name"] == "auth/tokens.validate_token"

    def score(result, name):
        return next(hit["score"] for hit in result["hits"] if hit["name"] == name)

    # Equal terms score equally; a short-name match on the whole query adds 2.5.
    camel_bonus = score(camel, "web/auth.validateToken") - score(spaced, "web/auth.validateToken")
    assert camel_bonus == pytest.approx(2.5, abs=1e-3)
    snake_bonus = score(snake, "auth/tokens.validate_token") - score(
        spaced, "auth/tokens.validate_token"
    )
    assert snake_bonus == pytest.approx(2.5, abs=1e-3)
    assert score(camel, "auth/tokens.validate_token") == score(spaced, "auth/tokens.validate_token")


@pytest.mark.asyncio
async def test_search_filters_kinds_tests_repo_and_paginates():
    flows = await _run({"operation": "search", "text": "orders", "kinds": ["flow"]})
    assert [hit["kind"] for hit in flows["hits"]] == ["flow"]
    (flow,) = flows["hits"]
    assert flow["id"] == "flow-orders"
    assert flow["type"] == "CodeFlow"
    assert flow["mechanism"] == "route"
    assert flow["rank"] == 1
    assert flow["enola_id"] == F_ORDERS
    assert flow["summary"].startswith("The program starts at POST /orders")

    with_tests = await _run(
        {"operation": "search", "text": "validate token", "include_tests": True}
    )
    assert "tests/test_tokens.test_validate_token" in [hit["name"] for hit in with_tests["hits"]]

    everything = await _run({"operation": "search", "text": "orders", "kinds": "all"})
    assert {"module", "storage", "flow", "route"} <= {hit["kind"] for hit in everything["hits"]}

    page = await _run({"operation": "search", "text": "orders", "limit": 2, "offset": 1})
    assert len(page["hits"]) == 2
    assert page["offset"] == 1 and page["limit"] == 2 and page["has_more"] is True

    elsewhere = await _run({"operation": "search", "text": "token", "repo": "other"})
    assert elsewhere["hits"] == [] and elsewhere["total"] == 0


@pytest.mark.asyncio
async def test_search_without_hits_is_empty_and_without_text_is_an_error():
    nothing = await _run({"operation": "search", "text": "kubernetes operator"})
    assert nothing["hits"] == [] and nothing["total"] == 0

    with pytest.raises(CodeSearchValidationError) as error:
        await _run({"operation": "search"}, "   ")
    assert error.value.error_code == "invalid_argument"
    assert error.value.error_param == "text"

    for bad, param in (({"limit": -1}, "limit"), ({"kinds": ["widget"]}, "kinds")):
        with pytest.raises(CodeSearchValidationError) as error:
            await _run({"operation": "search", "text": "token", **bad})
        assert (error.value.error_code, error.value.error_param) == ("invalid_argument", param)
    # The kinds a text search takes beyond the fact kinds are named too.
    assert "file_ref, flow, insight" in error.value.message
    assert error.value.message.endswith("test_ref, or all.")


@pytest.mark.asyncio
async def test_search_index_is_built_once_on_first_use_and_cached():
    nodes, edges = _shop_graph()
    with _patch_graph(nodes, edges):
        await CodeRetriever(
            config={"operation": "neighbors", "id": "create"}
        ).get_retrieved_objects("")
        snapshot = await CodeRetriever(config={"operation": "search"})._snapshot()
        assert snapshot._text_index is None
        await CodeRetriever(config={"operation": "search", "text": "token"}).get_retrieved_objects(
            ""
        )
        index = snapshot._text_index
        await CodeRetriever(config={"operation": "search", "text": "order"}).get_retrieved_objects(
            ""
        )

    assert index is not None
    assert snapshot._text_index is index


# --- neighbors -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neighbors_lists_every_edge_with_relation_and_direction():
    result = await _run({"operation": "neighbors", "name": "orders/api.create_order"})

    assert result["anchor"]["id"] == "create"
    assert "relations" not in result["anchor"]
    assert [
        (entry["relation"], entry["direction"], entry["node"]["name"])
        for entry in result["neighbors"]
    ] == [
        ("calls", "out", "auth/tokens.validate_token"),
        ("calls", "out", "orders/service.place_order"),
        ("declares", "out", "orders"),
        ("handled_by", "in", "POST /orders"),
    ]
    assert all("relations" not in entry["node"] for entry in result["neighbors"])
    assert result["flows"] == [{"id": "flow-orders", "name": "POST /orders", "as": "entry"}]
    assert result["total"] == 4
    assert result["truncated"] is False and result["has_more"] is False


@pytest.mark.asyncio
async def test_neighbors_filters_and_truncates():
    callers = await _run(
        {"operation": "neighbors", "id": E_VALIDATE, "direction": "reverse", "relation": "calls"}
    )
    assert [(entry["direction"], entry["node"]["name"]) for entry in callers["neighbors"]] == [
        ("in", "orders/api.create_order"),
        ("in", "tests/test_tokens.test_validate_token"),
    ]
    assert callers["flows"] == [{"id": "flow-orders", "name": "POST /orders", "as": "stage"}]

    page = await _run({"operation": "neighbors", "node_id": "create", "limit": 1})
    assert len(page["neighbors"]) == 1
    assert page["truncated"] is True and page["has_more"] is True and page["total"] == 4

    plain = await _run({"operation": "neighbors"}, "orders/report.render_report")
    assert "flows" not in plain


@pytest.mark.asyncio
async def test_neighbors_reports_missing_and_ambiguous_anchors_with_codes():
    with pytest.raises(CodeSeedNotFoundError) as missing:
        await _run({"operation": "neighbors", "name": "nowhere.at_all"})
    assert (missing.value.error_code, missing.value.error_param) == ("not_found", "name")
    assert "could not resolve neighbors anchor name 'nowhere.at_all'" in missing.value.message

    with pytest.raises(CodeSeedNotFoundError) as unknown_id:
        await _run({"operation": "neighbors", "id": "f" * 32})
    assert (unknown_id.value.error_code, unknown_id.value.error_param) == ("not_found", "id")

    with pytest.raises(CodeSeedNotFoundError) as ambiguous:
        await _run({"operation": "neighbors", "name": "tokens"})
    assert (ambiguous.value.error_code, ambiguous.value.error_param) == ("ambiguous", "name")


# --- subgraph ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subgraph_returns_nodes_edges_and_no_truncation_within_limits():
    result = await _run({"operation": "subgraph", "seeds": ["orders/api.create_order"]})

    assert [seed["id"] for seed in result["seeds"]] == ["create"]
    assert result["seeds_missing"] == []
    depths = {node["name"]: node["depth"] for node in result["nodes"]}
    assert depths == {
        "orders/api.create_order": 0,
        "POST /orders": 1,
        "auth/tokens.validate_token": 1,
        "orders": 1,
        "orders/service.place_order": 1,
    }
    assert {(edge["source"], edge["target"], edge["type"]) for edge in result["edges"]} == {
        ("POST /orders", "orders/api.create_order", "handled_by"),
        ("orders/api.create_order", "auth/tokens.validate_token", "calls"),
        ("orders/api.create_order", "orders/service.place_order", "calls"),
        ("orders/api.create_order", "orders", "declares"),
    }
    assert result["truncated"] is False
    assert result["truncation_reason"] is None and result["truncation_reasons"] == []
    assert result["limits"] == {"seeds": 20, "depth": 1, "max_nodes": 100, "max_edges": 300}
    assert result["stats"]["depth_reached"] == 1


@pytest.mark.asyncio
async def test_subgraph_reports_every_truncation_reason_in_order():
    seeds = [f"missing_{index}" for index in range(21)]
    seeds[0] = "orders/api.create_order"
    result = await _run(
        {
            "operation": "subgraph",
            "seeds": seeds,
            "depth": 7,
            "max_nodes": 3,
            "max_edges": 1,
            "relation_types": ["calls", "handled_by"],
        }
    )

    assert result["truncated"] is True
    assert result["truncation_reason"] == "max_seeds"
    assert result["truncation_reasons"] == ["max_seeds", "max_depth", "max_nodes", "max_edges"]
    assert result["limits"]["depth"] == 3
    assert len(result["nodes"]) == 3
    assert len(result["edges"]) == 1
    # 20 seeds were kept in order; the 19 that do not resolve are listed.
    assert len(result["seeds_missing"]) == 19
    assert result["seeds_missing"][0]["seed"] == "missing_1"
    assert result["seeds_missing"][0]["reason"] == "not_found"


@pytest.mark.asyncio
async def test_subgraph_reports_each_limit_on_its_own():
    deep = await _run({"operation": "subgraph", "seed_ids": ["create"], "max_depth": 4})
    assert deep["truncation_reasons"] == ["max_depth"]
    assert deep["stats"]["depth_reached"] == 3

    nodes = await _run(
        {"operation": "subgraph", "names": ["orders/api.create_order"], "max_nodes": 2}
    )
    assert nodes["truncation_reason"] == "max_nodes" and len(nodes["nodes"]) == 2

    edges = await _run({"operation": "subgraph", "ids": [E_CREATE], "max_edges": 2})
    assert edges["truncation_reasons"] == ["max_edges"] and len(edges["edges"]) == 2


@pytest.mark.asyncio
async def test_subgraph_lists_unresolved_seeds_and_fails_only_when_none_resolves():
    result = await _run(
        {"operation": "subgraph", "seeds": ["orders/api.create_order", "tokens", "validat_tokens"]}
    )
    assert [seed["id"] for seed in result["seeds"]] == ["create"]
    missing = {item["seed"]: item for item in result["seeds_missing"]}
    assert missing["tokens"]["reason"] == "ambiguous"
    assert [candidate["name"] for candidate in missing["tokens"]["candidates"]] == [
        "auth/tokens.parse_header",
        "auth/tokens.validate_token",
        "tests/test_tokens.test_validate_token",
    ]
    assert missing["validat_tokens"]["reason"] == "not_found"
    assert missing["validat_tokens"]["candidates"][0] == {
        "id": "validate",
        "name": "auth/tokens.validate_token",
        "kind": "symbol",
        "file": "auth/tokens.py",
    }

    with pytest.raises(CodeSeedNotFoundError) as error:
        await _run({"operation": "subgraph", "seeds": ["nowhere", "nothing.here"]})
    assert (error.value.error_code, error.value.error_param) == ("not_found", "seeds")


# --- flows ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flows_list_ranks_and_filters():
    result = await _run({"operation": "flows"})

    assert result["mode"] == "list"
    assert [flow["name"] for flow in result["flows"]] == ["POST /orders", "cli.main"]
    orders = result["flows"][0]
    assert orders["id"] == "flow-orders"
    assert orders["enola_id"] == F_ORDERS
    assert orders["rank"] == 1 and orders["mechanism"] == "route" and orders["repo"] == REPO
    assert orders["entry"] == {
        "name": "orders/api.create_order",
        "enola_id": E_CREATE,
        "node_id": "create",
        "file": "orders/api.py",
        "line": 8,
        "doc": CREATE_DOC,
    }
    assert orders["routes"] == [
        {
            "name": "POST /orders",
            "enola_id": E_ROUTE,
            "node_id": "route",
            "file": "orders/api.py",
            "line": 7,
            "method": "POST",
        }
    ]
    assert orders["metrics"]["unresolved_calls"] == 4
    assert "stages" not in orders and "side" not in orders

    with_side = await _run({"operation": "flows", "include_side": True, "mechanism": "main"})
    assert [flow["name"] for flow in with_side["flows"]] == ["cli.main", "scripts/seed.main"]
    assert with_side["flows"][1]["side"] is True
    assert with_side["flows"][1]["entry"]["node_id"] is None

    page = await _run({"operation": "flows", "limit": 1})
    assert page["total"] == 2 and page["has_more"] is True


@pytest.mark.asyncio
async def test_flows_show_resolves_stage_refs_by_id_edge_or_not_at_all():
    by_name = await _run({"operation": "flows", "name": "/orders"})
    by_enola_id = await _run({"operation": "flows", "id": F_ORDERS})
    by_node_id = await _run({"operation": "flows", "mode": "show"}, "POST /orders")

    assert by_name == by_enola_id == by_node_id
    flow = by_name["flow"]
    assert by_name["mode"] == "show"
    assert [(stage["order"], stage["name"], stage["node_id"]) for stage in flow["stages"]] == [
        (1, "orders/service.place_order", "place"),
        (2, "billing.charge_card", "charge"),
        (3, "auth/tokens.validate_token", "validate"),
        (4, "vendor.retry", None),
    ]
    assert flow["stages"][1]["parent"] == E_PLACE
    assert flow["stages"][1]["doc"] == "Charges a card through the payment provider."
    assert flow["stages"][3] == {
        "name": "vendor.retry",
        "enola_id": E_VENDOR,
        "node_id": None,
        "file": "vendor/retry.py",
        "line": 9,
        "order": 4,
        "depth": 2,
        "parent": E_PLACE,
        "reach": 1,
    }
    assert flow["modules"][0] == {"name": "orders", "symbols": 3}

    with pytest.raises(CodeSeedNotFoundError) as ambiguous:
        await _run({"operation": "flows", "name": "main"})
    assert (ambiguous.value.error_code, ambiguous.value.error_param) == ("ambiguous", "name")
    with pytest.raises(CodeSeedNotFoundError) as missing:
        await _run({"operation": "flows", "flow": "DELETE /orders"})
    assert (missing.value.error_code, missing.value.error_param) == ("not_found", "name")


@pytest.mark.asyncio
async def test_flows_through_walks_callers_up_to_the_entries():
    through = await _run({"operation": "flows", "symbol": "auth/tokens.parse_header"})

    assert through["mode"] == "through"
    assert through["symbol"]["id"] == "parse"
    assert [(flow["name"], flow["via"], flow["distance"]) for flow in through["flows"]] == [
        ("POST /orders", "reach", 2)
    ]
    assert through["total"] == 1 and through["truncated"] is False

    stage = await _run({"operation": "flows", "through": "billing.charge_card"})
    assert [(flow["via"], flow["distance"]) for flow in stage["flows"]] == [("stage", 2)]

    entry = await _run({"operation": "flows", "mode": "through", "id": "create"})
    assert [(flow["via"], flow["distance"]) for flow in entry["flows"]] == [("entry", 0)]

    route = await _run({"operation": "flows", "mode": "through"}, "POST /orders")
    assert [(flow["via"], flow["distance"]) for flow in route["flows"]] == [("route", 0)]

    outside = await _run({"operation": "flows", "symbol": "orders/report.render_report"})
    assert [(flow["name"], flow["via"]) for flow in outside["flows"]] == [("cli.main", "reach")]


@pytest.mark.asyncio
async def test_flows_need_flows_json_and_a_valid_mode():
    with pytest.raises(CodeGraphUnavailableError) as unavailable:
        await _run({"operation": "flows"}, with_flows=False)
    assert unavailable.value.error_code == "feature_unavailable"
    assert unavailable.value.error_param == "operation"
    assert "flows.json" in unavailable.value.message
    assert "re-index" in unavailable.value.message
    # A per-dataset miss, like an unresolved seed: recall drops it, search softens it.
    assert isinstance(unavailable.value, CodeSeedNotFoundError)

    with _patch_graph([], []), pytest.raises(CodeGraphUnavailableError) as empty:
        await CodeRetriever(config={"operation": "flows"}).get_retrieved_objects("")
    assert empty.value.error_code == "no_code_graph"

    with pytest.raises(CodeSearchValidationError) as bad_mode:
        await _run({"operation": "flows", "mode": "draw"})
    assert bad_mode.value.error_param == "mode"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "param"),
    [
        ({"operation": "flows", "limit": -1}, "limit"),
        ({"operation": "flows", "offset": "x"}, "offset"),
        ({"operation": "flows", "mechanism": 5}, "mechanism"),
        ({"operation": "flows", "include_side": "maybe"}, "include_side"),
        ({"operation": "flows", "repo": 7}, "repo"),
        ({"operation": "flows", "mode": "show", "id": ["F1"]}, "id"),
        ({"operation": "flows", "mode": "show", "name": 3}, "name"),
        ({"operation": "flows", "mode": "through", "symbol": ["create"]}, "symbol"),
    ],
)
async def test_flows_reject_bad_arguments_before_reporting_missing_flows(config, param):
    """feature_unavailable is softened into a per-dataset miss, so checking for flows
    first would hide a caller's bad argument behind a silent empty result."""
    with pytest.raises(CodeSearchValidationError) as error:
        await _run(config, with_flows=False)
    assert (error.value.error_code, error.value.error_param) == ("invalid_argument", param)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "param"),
    [
        ({"operation": "neighbors", "id": ["create"]}, "id"),
        ({"operation": "neighbors", "node_id": {"id": "create"}}, "node_id"),
        ({"operation": "neighbors", "name": 5}, "name"),
        ({"operation": "neighbors", "name": "orders.create_order", "repo": 1}, "repo"),
        ({"operation": "flows", "mode": "show", "id": True}, "id"),
        ({"operation": "flows", "mode": "show", "flow": ["POST /orders"]}, "flow"),
        ({"operation": "flows", "mode": "through", "through": 1.5}, "through"),
        ({"operation": "flows", "mode": "through", "symbol": "create", "id": [1]}, "id"),
    ],
)
async def test_ids_and_names_of_the_wrong_type_are_invalid_not_missing(config, param):
    """A list or number is the caller's mistake (invalid_argument), never an identifier
    that fails to resolve (not_found, which recall and multi-dataset search swallow)."""
    with pytest.raises(CodeSearchValidationError) as error:
        await _run(config)
    assert not isinstance(error.value, CodeSeedNotFoundError)
    assert (error.value.error_code, error.value.error_param) == ("invalid_argument", param)


@pytest.mark.asyncio
async def test_a_uuid_id_is_an_identifier_not_a_type_error():
    from uuid import uuid4

    with pytest.raises(CodeSeedNotFoundError) as missing:
        await _run({"operation": "neighbors", "id": uuid4()})
    assert missing.value.error_code == "not_found"


# --- hubs ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hubs_rank_by_degree_without_containment_edges():
    result = await _run({"operation": "hubs"})

    # Equal degrees fall back to the stable node order; declares and part_of
    # do not count (create_order: calls x2 out, handled_by in), and neither
    # does the test's call into validate_token.
    assert [
        (hub["name"], hub["degree"], hub["in_degree"], hub["out_degree"])
        for hub in result["hubs"][:3]
    ] == [
        ("orders/api.create_order", 3, 1, 2),
        ("orders/service.place_order", 3, 1, 2),
        ("auth/tokens.validate_token", 2, 1, 1),
    ]
    assert "relations" not in result["hubs"][0]
    assert all(hub["kind"] == "symbol" for hub in result["hubs"])
    assert "tests/test_tokens.test_validate_token" not in [hub["name"] for hub in result["hubs"]]
    assert result["limit"] == 20 and result["total"] == len(result["hubs"])

    with_tests = await _run({"operation": "hubs", "include_tests": True})
    assert [
        (hub["name"], hub["degree"], hub["in_degree"], hub["out_degree"])
        for hub in with_tests["hubs"][:1]
    ] == [("auth/tokens.validate_token", 3, 2, 1)]
    assert "tests/test_tokens.test_validate_token" in [hub["name"] for hub in with_tests["hubs"]]

    callers = await _run(
        {"operation": "hubs", "direction": "reverse", "include_tests": True, "limit": 1}
    )
    assert [(hub["name"], hub["degree"]) for hub in callers["hubs"]] == [
        ("auth/tokens.validate_token", 2)
    ]

    modules = await _run({"operation": "hubs", "kinds": ["module"], "relation_types": ["declares"]})
    assert [(hub["name"], hub["degree"]) for hub in modules["hubs"]] == [("auth", 2), ("orders", 2)]


# --- ask -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_seeds_flows_neighbourhood_citations_and_coverage():
    result = await _run(
        {"operation": "ask", "question": "Where do we validate auth tokens for new orders?"}
    )

    assert result["operation"] == "ask"
    assert result["terms"] == ["valid", "auth", "token", "new", "order"]
    seed_names = [seed["name"] for seed in result["seeds"]]
    assert seed_names[0] == "auth/tokens.validate_token"
    assert "orders/api.create_order" in seed_names
    assert all(seed["kind"] != "flow" for seed in result["seeds"])
    assert result["seeds"][0]["doc"] == VALIDATE_DOC

    (orders,) = [flow for flow in result["flows"] if flow["name"] == "POST /orders"]
    assert [stage["name"] for stage in orders["stages"]][:2] == [
        "orders/service.place_order",
        "billing.charge_card",
    ]
    assert "auth/tokens.validate_token" in orders["seeds_reached"]
    assert len(result["flows"]) <= 3

    roles = {node["name"]: node["role"] for node in result["neighborhood"]["nodes"]}
    assert roles["auth/tokens.validate_token"] == "seed"
    assert roles["orders/service.place_order"] == "callee"
    assert roles["POST /orders"] == "neighbor"
    # The test calling validate_token is test code, left out by default.
    assert "tests/test_tokens.test_validate_token" not in roles
    assert all(
        node["depth"] == (0 if node["role"] == "seed" else 1)
        for node in result["neighborhood"]["nodes"]
    )
    assert "declares" not in {edge["type"] for edge in result["neighborhood"]["edges"]}

    citations = {citation["cite"]: citation for citation in result["citations"]}
    assert citations["auth/tokens.py:12"]["why"] == "seed"
    assert citations["auth/tokens.py:12"]["doc"] == VALIDATE_DOC
    assert citations["auth/tokens.py:12"]["end_line"] == 30
    assert citations["orders/service.py:20"]["why"] == "stage 1 of flow POST /orders"
    assert citations["vendor/retry.py:9"]["node_id"] is None
    assert citations["vendor/retry.py:9"]["why"] == "stage 4 of flow POST /orders"
    assert "tests/test_tokens.py:3" not in citations
    assert citations["orders/api.py:7"]["why"] == (
        "linked to orders/api.create_order by handled_by"
    )
    whys = [citation["why"] for citation in result["citations"]]
    # Seeds first, then flow entries and stages, then callers/callees, then the rest.
    assert whys[: len(result["seeds"])] == ["seed"] * len(result["seeds"])
    assert whys[-1].startswith("linked to")
    assert len(
        {citation["node_id"] or citation["cite"] for citation in result["citations"]}
    ) == len(result["citations"])
    assert all(citation["file"] for citation in result["citations"])

    coverage = result["coverage"]
    assert coverage["code_graph"] is True
    assert coverage["docs_present"] is True and coverage["flows_present"] is True
    # Docs are counted over the seeds that can carry one: the auth module
    # seed has no docstring to miss.
    documentable = [seed for seed in result["seeds"] if seed["kind"] == "symbol" or "doc" in seed]
    assert [seed["name"] for seed in result["seeds"] if seed not in documentable] == ["auth"]
    assert coverage["seeds"] == len(documentable)
    assert coverage["seeds_with_docs"] == sum(1 for seed in documentable if "doc" in seed) == 2
    assert coverage["unresolved_calls"] == 4
    assert coverage["truncated"] is False and result["truncated"] is False
    assert coverage["notes"] == [
        (
            "4 calls in these flows go to code outside the graph (libraries or callees the "
            "extractor could not resolve)."
        )
    ]

    next_steps = [step["code_query"] for step in result["next"]]
    assert 2 <= len(next_steps) <= 4
    assert next_steps[0] == {"operation": "flows", "mode": "show", "id": result["flows"][0]["id"]}
    assert {"operation": "neighbors", "id": result["seeds"][0]["id"]} in next_steps
    assert any(step["operation"] == "subgraph" and step["depth"] == 2 for step in next_steps)
    assert all(step["why"] for step in result["next"])
    # recall reads a result's text from these keys; ask keeps them nested.
    assert not {"text", "completion", "summary", "name", "content", "answer"} & set(result)


@pytest.mark.asyncio
async def test_ask_coverage_names_missing_docs_flows_and_matches():
    bare = await _run(
        {"operation": "ask"}, "how does order placement work?", with_flows=False, with_docs=False
    )
    assert bare["coverage"]["docs_present"] is False
    assert bare["coverage"]["flows_present"] is False
    assert bare["coverage"]["unresolved_calls"] is None
    assert bare["flows"] == []
    assert bare["coverage"]["notes"] == [
        "This snapshot has no docstrings, so matching used names and paths only.",
        "This snapshot has no flows, so entry points were not traced.",
    ]

    nothing = await _run({"operation": "ask", "text": "kubernetes operator reconcile"})
    assert nothing["seeds"] == [] and nothing["citations"] == []
    assert nothing["coverage"]["notes"][0] == (
        "No symbol, route or flow matched these terms: kubernet, oper, reconcil. "
        "Try an exact identifier with neighbors, or grep."
    )
    assert [step["code_query"]["operation"] for step in nothing["next"]] == ["flows", "hubs"]


@pytest.mark.asyncio
async def test_ask_reports_neighbourhood_and_citation_cuts():
    result = await _run(
        {
            "operation": "ask",
            "question": "validate token order",
            "max_seeds": 2,
            "max_nodes": 3,
            "max_citations": 2,
        }
    )

    assert len(result["seeds"]) == 2
    assert len(result["neighborhood"]["nodes"]) == 3
    assert len(result["citations"]) == 2
    assert result["truncated"] is True and result["coverage"]["truncated"] is True
    assert (
        "The neighbourhood was cut at 3 nodes; use subgraph or neighbors to see more."
        in result["coverage"]["notes"]
    )
    assert "Citations were cut at 2." in result["coverage"]["notes"]


@pytest.mark.asyncio
async def test_ask_and_hubs_leave_test_code_out_unless_asked():
    nodes, edges = _shop_graph()
    # Thirty tests call the token validator; one production caller does.
    for index in range(30):
        path = f"tests/test_tokens_{index:02d}.py"
        nodes.append(
            (
                f"test-ref-{index}",
                {"name": path, "type": "CodeTestReference", "file_path": path, "repo": REPO},
            )
        )
        edges.append((f"test-ref-{index}", "validate", "calls", {}))
    ask = {"operation": "ask", "question": "validate token", "max_seeds": 1, "max_nodes": 10}
    with _patch_graph(nodes, edges):
        asked = await CodeRetriever(config=ask).get_retrieved_objects("")
        asked_with_tests = await CodeRetriever(
            config={**ask, "include_tests": True}
        ).get_retrieved_objects("")
        hubs = await CodeRetriever(config={"operation": "hubs"}).get_retrieved_objects("")
        hubs_with_tests = await CodeRetriever(
            config={"operation": "hubs", "include_tests": True}
        ).get_retrieved_objects("")

    assert {node["name"]: node["role"] for node in asked["neighborhood"]["nodes"]} == {
        "auth/tokens.validate_token": "seed",
        "orders/api.create_order": "caller",
        "auth/tokens.parse_header": "callee",
    }
    assert not [c for c in asked["citations"] if c["file"].startswith("tests/")]
    assert asked["truncated"] is False

    # Asked for, the tests come first among the callers and fill the budget.
    names = [node["name"] for node in asked_with_tests["neighborhood"]["nodes"]]
    assert len(names) == 10 and "auth/tokens.parse_header" not in names
    assert sum(name.startswith("tests/") for name in names) == 8
    citations = {citation["cite"]: citation["why"] for citation in asked_with_tests["citations"]}
    assert citations["tests/test_tokens.py:3"] == "calls auth/tokens.validate_token"
    assert asked_with_tests["truncated"] is True

    def degrees(result, name):
        hub = next(hub for hub in result["hubs"] if hub["name"] == name)
        return hub["degree"], hub["in_degree"], hub["out_degree"]

    assert degrees(hubs, "auth/tokens.validate_token") == (2, 1, 1)
    assert hubs["hubs"][0]["name"] == "orders/api.create_order"
    assert degrees(hubs_with_tests, "auth/tokens.validate_token") == (33, 32, 1)
    assert hubs_with_tests["hubs"][0]["name"] == "auth/tokens.validate_token"


@pytest.mark.asyncio
async def test_ask_counts_docs_over_the_seeds_that_can_carry_one():
    route = {
        "name": "/checkout",
        "type": "ApiEndpoint",
        "file_path": "shop/api.py",
        "line": 14,
        "repo": REPO,
    }
    handler = _symbol(
        "shop/api.process_checkout",
        "shop/api.py",
        15,
        doc="Charge the cart, then create the order.",
    )
    edges = [("route", "handler", "handled_by", {})]
    ask = {"operation": "ask", "question": "How does checkout work?"}
    with _patch_graph([("route", route), ("handler", handler)], edges):
        result = await CodeRetriever(config=ask).get_retrieved_objects("")
    # Enola documents symbols: a route seed is no docstring missing.
    assert sorted(seed["kind"] for seed in result["seeds"]) == ["route", "symbol"]
    assert (result["coverage"]["seeds_with_docs"], result["coverage"]["seeds"]) == (1, 1)

    documented_route = {**route, "doc": "Checks the cart out."}
    with _patch_graph([("route", documented_route), ("handler", handler)], edges):
        result = await CodeRetriever(config=ask).get_retrieved_objects("")
    assert (result["coverage"]["seeds_with_docs"], result["coverage"]["seeds"]) == (2, 2)


@pytest.mark.asyncio
async def test_max_nodes_caps_subgraph_and_ask_seeds_included():
    seeds = [
        "orders/api.create_order",
        "auth/tokens.validate_token",
        "orders/service.place_order",
        "orders/repository.save",
        "billing.charge_card",
    ]
    subgraph = await _run({"operation": "subgraph", "seeds": seeds, "max_nodes": 2})
    # The first two seeds, in the order given; the rest are cut.
    assert [seed["name"] for seed in subgraph["seeds"]] == seeds[:2]
    assert sorted(node["name"] for node in subgraph["nodes"]) == sorted(seeds[:2])
    assert {(edge["source"], edge["target"]) for edge in subgraph["edges"]} == {
        ("orders/api.create_order", "auth/tokens.validate_token")
    }
    assert subgraph["truncation_reasons"] == ["max_nodes"]
    assert subgraph["limits"]["max_nodes"] == 2

    asked = await _run(
        {"operation": "ask", "question": "orders token card report", "max_seeds": 5, "max_nodes": 2}
    )
    assert len(asked["seeds"]) == 5
    assert [node["role"] for node in asked["neighborhood"]["nodes"]] == ["seed", "seed"]
    assert asked["truncated"] is True
    assert (
        "The neighbourhood was cut at 2 nodes; use subgraph or neighbors to see more."
        in asked["coverage"]["notes"]
    )
    # Seeds stay evidence even when the neighbourhood has no room for them.
    assert [citation["why"] for citation in asked["citations"]][:5] == ["seed"] * 5


def test_a_limited_text_match_is_the_head_of_the_full_ranking():
    nodes, edges = _shop_graph()
    # Ten symbols shaped alike tie on every score; the stable order breaks the tie.
    nodes += [
        (f"tie-{index}", _symbol(f"pkg{index}/tokens.check_token", f"pkg{index}/tokens.py", 1))
        for index in reversed(range(10))
    ]
    graph = _CodeGraphSnapshot(nodes, edges)

    def matches(text, limit=None):
        return CodeRetriever._text_matches(
            graph,
            text,
            kinds=None,
            repo=None,
            file_prefix="",
            include_tests=True,
            limit=limit,
        )

    for text in ("check token", "orders", "validate token", "kubernetes"):
        terms, ranked = matches(text)
        for limit in (1, 3, 5, 50):
            assert matches(text, limit) == (terms, ranked[:limit])
    _terms, head = matches("check token", 3)
    assert [key for key, _score, _fields in head] == ["tie-0", "tie-1", "tie-2"]


@pytest.mark.asyncio
async def test_ask_needs_a_question_and_a_code_graph():
    with pytest.raises(CodeSearchValidationError) as no_question:
        await _run({"operation": "ask"}, "")
    assert (no_question.value.error_code, no_question.value.error_param) == (
        "invalid_argument",
        "question",
    )

    with _patch_graph([], []), pytest.raises(CodeGraphUnavailableError) as no_graph:
        await CodeRetriever(config={"operation": "ask"}).get_retrieved_objects("checkout")
    assert no_graph.value.error_code == "no_code_graph"


# --- shared contract -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_operations_ignore_flow_nodes_and_gain_only_doc():
    with_flows = await _run({"operation": "query_facts", "limit": 100})
    without_flows = await _run({"operation": "query_facts", "limit": 100}, with_flows=False)

    assert with_flows == without_flows
    assert all(fact["kind"] != "flow" for fact in with_flows["facts"])
    by_name = {fact["name"]: fact for fact in with_flows["facts"]}
    assert by_name["auth/tokens.validate_token"]["doc"] == VALIDATE_DOC
    # Snapshots loaded before docs were read keep the doc only as a prop.
    assert by_name["auth/session.SessionStore"]["doc"] == "Keeps user sessions in Redis."
    assert "doc" not in by_name["auth/tokens.parse_header"]

    explored = await _run({"operation": "explore", "name": "orders/api.create_order"})
    assert {edge["type"] for edge in explored["edges"]} == {"calls", "declares", "handled_by"}
    assert all(not node["id"].startswith("flow-") for node in explored["nodes"])

    with pytest.raises(CodeSearchValidationError, match="Unsupported CODE node kind/type 'flow'"):
        await _run({"operation": "query_facts", "kind": "flow"})


def test_snapshot_keeps_flows_aside_and_indexes_membership():
    graph = _CodeGraphSnapshot(*_shop_graph())

    assert set(graph.flows) == {"flow-orders", "flow-main", "flow-seed"}
    assert not set(graph.flows) & set(graph.nodes)
    assert all(edge["type"] not in {"flow_entry", "flow_stage"} for edge in graph.edges)
    assert graph.flow_members["flow-orders"] == {
        "entry": "create",
        "stages": ["place", "charge", "validate", None],
        "routes": ["route"],
    }
    assert graph.flows_by_node["charge"] == [("flow-orders", "stage")]
    assert graph.flow_by_enola_id[F_MAIN] == "flow-main"
    assert graph.documented == {"validate", "session", "create", "place", "charge", "report"}


def test_docs_are_read_by_the_rule_ingestion_stores_them_with():
    nodes = [
        # Ingestion never reads an insight's doc prop: its prose is the description.
        (
            "insight",
            {
                "name": "Call-graph hotspot: billing.charge_card",
                "type": "CodeInsight",
                "description": "A pinch point.",
                "doc": None,
                "fact_properties": {"source": "hotspots", "doc": "Not a doc comment."},
            },
        ),
        ("blank", _symbol("auth/tokens.strip", "auth/tokens.py", 50, doc="   ")),
        # Written since docs were read: the null field means no doc.
        ("none", {**_symbol("auth/tokens.none", "auth/tokens.py", 60), "doc": None}),
    ]
    graph = _CodeGraphSnapshot(nodes, [])

    assert graph.documented == set()
    assert all("doc" not in graph.fact(node_id) for node_id, _node in nodes)


@pytest.mark.asyncio
async def test_a_null_doc_field_is_no_doc_to_the_property_filters():
    def with_null_docs(nodes, edges):
        # Ladybug keeps the null doc of every node written since docs were
        # read; Neo4j and nodes written before keep no doc key at all.
        return [(node_id, {"doc": None, **node}) for node_id, node in nodes], edges

    query = {"operation": "query_facts", "property": "doc", "limit": 100}
    with _patch_graph(*with_null_docs(*_shop_graph())):
        documented = await CodeRetriever(config=query).get_retrieved_objects("")
    with _patch_graph(*with_null_docs(*_shop_graph(with_docs=False))):
        undocumented = await CodeRetriever(config=query).get_retrieved_objects("")

    assert sorted(fact["name"] for fact in documented["facts"]) == [
        "auth/session.SessionStore",
        "auth/tokens.validate_token",
        "billing.charge_card",
        "orders/api.create_order",
        "orders/report.render_report",
        "orders/service.place_order",
    ]
    assert documented == await _run(query)
    assert undocumented["total"] == 0


def test_unknown_operation_and_unsupported_diagrams_carry_codes():
    with pytest.raises(CodeSearchValidationError, match="Unsupported CODE operation") as error:
        CodeRetriever(config={"operation": "guess"})
    assert (error.value.error_code, error.value.error_param) == ("unknown_operation", "operation")
    assert error.value.message.startswith("Unsupported CODE operation 'guess'; expected one of ")
    assert "ask" in error.value.message

    for operation in ("search", "ask", "flows", "hubs", "capabilities"):
        with pytest.raises(CodeSearchValidationError) as diagram:
            CodeRetriever(config={"operation": operation, "diagram": "mermaid"})
        assert (diagram.value.error_code, diagram.value.error_param) == (
            "invalid_argument",
            "diagram",
        )
        CodeRetriever(config={"operation": operation, "diagram": False})

    with pytest.raises(CodeSearchValidationError) as bad_format:
        CodeRetriever(config={"operation": "explore", "diagram": "png"})
    assert bad_format.value.error_param == "diagram"


@pytest.mark.asyncio
async def test_argument_errors_name_the_offending_parameter():
    cases = [
        ({"operation": "neighbors", "id": "create", "direction": "sideways"}, "direction"),
        ({"operation": "neighbors", "id": "create", "limit": "ten"}, "limit"),
        ({"operation": "neighbors", "id": "create", "offset": -1}, "offset"),
        # A bad argument is a usage error even when the anchor does not resolve.
        ({"operation": "neighbors", "name": "nowhere.at_all", "limit": -1}, "limit"),
        ({"operation": "neighbors", "name": "nowhere.at_all", "kinds": ["widget"]}, "kinds"),
        ({"operation": "subgraph", "seed_ids": ["create"], "max_nodes": -5}, "max_nodes"),
        ({"operation": "subgraph", "seed_ids": ["create"], "kinds": ["widget"]}, "kinds"),
        ({"operation": "hubs", "include_tests": "maybe"}, "include_tests"),
        ({"operation": "explore", "id": "create", "node_types": ["widget"]}, "node_types"),
        ({"operation": "query_facts", "limit": 1.5}, "limit"),
        ({"operation": "insights", "min_confidence": 2}, "min_confidence"),
    ]
    for config, param in cases:
        with pytest.raises(CodeSearchValidationError) as error:
            await _run(config)
        assert (error.value.error_code, error.value.error_param) == ("invalid_argument", param)

    with _patch_graph(*_shop_graph()), pytest.raises(CodeSearchValidationError) as batch:
        await CodeRetriever(config={"operation": "search"}).get_retrieved_objects("x", ["y"])
    assert batch.value.error_param == "query_batch"


@pytest.mark.asyncio
async def test_neighbors_and_subgraph_render_diagrams():
    neighbors = await _run({"operation": "neighbors", "id": "create", "diagram": "mermaid"})
    source = neighbors["diagram"]["source"]
    assert source.startswith('---\ntitle: "neighbors: orders/api.create_order"\n---\nflowchart LR')
    assert neighbors["diagram"]["edges"] == 4
    assert '-- "handled_by" -->' in source

    subgraph = await _run({"operation": "subgraph", "seed_ids": ["create"], "diagram": "dot"})
    assert subgraph["diagram"]["format"] == "dot"
    assert subgraph["diagram"]["nodes"] == len(subgraph["nodes"])
    assert subgraph["diagram"]["edges"] == len(subgraph["edges"])
    assert "penwidth=3" in subgraph["diagram"]["source"]


@pytest.mark.asyncio
async def test_navigation_is_deterministic_and_never_calls_llm_or_vectors():
    llm_call = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    vector_call = MagicMock(side_effect=AssertionError("vector engine must not be called"))
    configs = [
        {"operation": "capabilities"},
        {"operation": "search", "text": "validate token"},
        {"operation": "neighbors", "id": "create"},
        {"operation": "subgraph", "seed_ids": ["create", "place"], "depth": 2},
        {"operation": "flows", "id": "flow-orders"},
        {"operation": "hubs"},
        {"operation": "ask", "question": "how are orders charged?"},
    ]
    nodes, edges = _shop_graph()
    with (
        patch("cognee.modules.retrieval.utils.completion.generate_completion", llm_call),
        patch("cognee.infrastructure.databases.vector.get_vector_engine", vector_call),
    ):
        forward = []
        with _patch_graph(nodes, edges):
            for config in configs:
                forward.append(await CodeRetriever(config=config).get_retrieved_objects(""))
        invalidate_code_graph_snapshot_cache(all_entries=True)
        backward = []
        with _patch_graph(list(reversed(nodes)), list(reversed(edges))):
            for config in configs:
                backward.append(await CodeRetriever(config=config).get_retrieved_objects(""))

    assert forward == backward
    llm_call.assert_not_awaited()
    vector_call.assert_not_called()
