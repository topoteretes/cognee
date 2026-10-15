"""Symbol docs and entry-point flows from an enola snapshot (SDK-1119).

enola writes the comment an author put on a symbol as the descriptive ``doc``
prop, and ranks the ways into the program in flows.json. A doc lands on its
node as ``doc`` and as the description; each flow becomes a CodeFlow node
linked to its entry, stages and routes. A snapshot with neither ingests
exactly as before: the golden values below were computed with the code that
predates both (origin/dev 2fb7e0329).
"""

import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.tasks.code_graph.enola import parse_enola_snapshot
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    build_code_graph_edges,
    extract_code_graph,
    fact_node_id,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.models import CodeFlow, CodeGraphEntity, CodeRepository

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# A small Python shop: doc comments on most symbols, and a flows.json with a
# served route's flow, the program's main and a side script.
DOCS_AND_FLOWS_DIR = FIXTURES_DIR / "docs_and_flows"
REPO = "shop"
ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32

# --- golden values, computed with the code before docs and flows were read ----

# Every fact of fixtures/, which has no doc props and no flows.json.
GOLDEN_FIXTURE_HASHES = {
    "app": "sha256:f2dce3d555c747c7bd1ca9aa7a279021e83f3ce6465a3cc2a6f0845ced2d215d",
    "utils": "sha256:7fa17c384a91bf8dc874247ba467779361276d4f189956235226cd91a38a6b3b",
    "main": "sha256:036e48f152a374d3b8b2be320bea8bba89732d57a5ecbba9964776ad7c857973",
    "helper": "sha256:8499b79c7abc5ad9e4c88f0da87fd967479c811ccde0d6509a67413e2da4b61d",
    "Config": "sha256:3d2e11e16a295f85a000fe0bcb0d056071ab1d164bdaa823c00258cb7e2afe1c",
    "Load": "sha256:4739c575ab3ef79c54a89807769b538b6194235f3b742478b2cbcd8503beba49",
    "Storer": "sha256:74a92ddf832ae406b9c52306c4e139dfa0cafa45aeea5ef4271eeb588ec715d3",
    "PostgresStore": "sha256:138c42e769d5b9b71c44263abf01c9faad71e4ca1c18c735162c8c87e6c123c3",
    "MAX_RETRIES": "sha256:4cc1c614b4b5a684c92210950c7a8d1a07094571f136525b781c8315969acc95",
    "log_level": "sha256:aa3b2d677f70a60dd6676f79bd55687ef608304e6f6eccbb9aa669c7ca87d7c8",
    "Color": "sha256:87ce0bcdeb70cc65a280e54f116f6297e4b2484385d236d02cf6f2c53be4ee0b",
    "OrderID": "sha256:c9957e8e964b4397486a566235c4a06aa8f4c4fdc39ee0ef56e9531c1d6f4394",
    "GET /orders": "sha256:b7458067a4881f61bc89a84b6c87ef20c201ef1a5aebfc5ccd75f3644e9b3805",
    "orders_table": "sha256:46e6ea1b702dc013d9e29bbf8b5118480c9018770129615487e40b74cd7fcd1e",
    "github.com/lib/pq": "sha256:b150c4f0473778b380a078ad9047fbe7a49500b79d63ad8235dd3d4a948ed173",
    "shop-api": "sha256:c2508b513891d886dd921be89c2dd0b946549ef7835010a8095e8ac6bac66d66",
}
GOLDEN_FIXTURE_EDGES = [
    ("Config", "has_method", "Load"),
    ("GET /orders", "handled_by", "helper"),
    ("PostgresStore", "implements", "Storer"),
    ("app", "declares", "main"),
    ("app", "imports", "utils"),
    ("main", "calls", "helper"),
    ("main", "instantiates", "Config"),
    ("shop-api", "depends_on", "github.com/lib/pq"),
    ("shop-api", "injects", "PostgresStore"),
]

# The undocumented facts of fixtures/docs_and_flows, and every edge between its
# facts: the docs and the flows next to them must leave both untouched.
GOLDEN_UNDOCUMENTED_HASHES = {
    ".": "sha256:4e7b17a7f021621785446ae4a972c0af46a864d550ff0c1792e22da07aa52f36",
    "auth": "sha256:7d1c8a7a0a7c388a871b61493ee97d5c239ae4ff7c353cb0d73f85160dd33e6e",
    "orders": "sha256:d7e44a4cf9fedad3b08a953d042b757aec71df4d08e34cd786c7c6d8270f4375",
    "scripts": "sha256:a9e327a2ffafe5968664794819e30ff5e6bb6d5de2e715323f93b7ca2edff830",
    "/checkout": "sha256:34c8f4b2f644ca197ba94ebea3edca13a8ddb575de344009b57a593be54dc8f1",
    "app.serve": "sha256:55b611e717d9817215620517935c23f3034fb97400c6379122ef4b44711b927f",
    "auth/tokens.decode": "sha256:59c983d4a68e064f2f6237b26b57c0dd0562f776558fb114c5c229b366a88ca3",
    "orders/repository.OrderRepository.save": (
        "sha256:a8f808fa46be3315fe9af60f43b9510f0fb3bfb7c4503c4d8767c47d6fc4cfde"
    ),
}
GOLDEN_CODE_EDGES = [
    ("/checkout", "declares", "orders"),
    ("/checkout", "handled_by", "orders/checkout.place_order"),
    ("app.main", "calls", "app.serve"),
    ("app.main", "declares", "."),
    ("app.serve", "declares", "."),
    ("auth/tokens.decode", "declares", "auth"),
    ("auth/tokens.validate_token", "calls", "auth/tokens.decode"),
    ("auth/tokens.validate_token", "declares", "auth"),
    ("orders/checkout.place_order", "calls", "auth/tokens.validate_token"),
    ("orders/checkout.place_order", "calls", "orders/repository.OrderRepository.save"),
    ("orders/checkout.place_order", "declares", "orders"),
    ("orders/repository.OrderRepository", "declares", "orders"),
    ("orders/repository.OrderRepository", "has_method", "orders/repository.OrderRepository.save"),
    ("orders/repository.OrderRepository.save", "declares", "orders"),
    ("scripts/seed.main", "calls", "orders/repository.OrderRepository.save"),
    ("scripts/seed.main", "declares", "scripts"),
]

# Facts whose doc prop is no doc to cognee: empty or blank, not text, or on an insight.
NOT_DOCUMENTED_FACTS = [
    {
        "kind": "symbol",
        "name": "app/db.connect",
        "file": "app/db.py",
        "line": 3,
        "repo": REPO,
        "props": {"doc": "", "symbol_kind": "function"},
    },
    {
        "kind": "symbol",
        "name": "app/db.ping",
        "file": "app/db.py",
        "line": 15,
        "repo": REPO,
        "props": {"doc": "   ", "symbol_kind": "function"},
    },
    {
        "kind": "symbol",
        "name": "app/db.close",
        "file": "app/db.py",
        "line": 9,
        "repo": REPO,
        "props": {"doc": ["not", "prose"], "symbol_kind": "function"},
    },
    {
        "kind": "insight",
        "name": "Call-graph hotspot: app/db.connect",
        "props": {
            "source": "hotspots",
            "description": "A pinch point.",
            "doc": "Not a doc comment.",
        },
    },
    {"kind": "insight", "name": "Empty finding", "props": {"source": "domain", "description": ""}},
    {"kind": "insight", "name": "Bare finding", "props": {"source": "layers", "confidence": 0.5}},
]
GOLDEN_NOT_DOCUMENTED_HASHES = {
    "app/db.connect": "sha256:b765e80dee56352e5bbbbe08d5f6f9ee4ddb8249ce09b5aeb38b9f67edc59a3a",
    "app/db.ping": "sha256:3e778ddbe0405c83c53e176d64296b6c52955136cdfa236bf73743656a0d54ab",
    "app/db.close": "sha256:54e056fb77d11a5fedca01854c9c32ae08978843b1ac9a58927ea6357c770675",
    "Call-graph hotspot: app/db.connect": (
        "sha256:3caef8cd94ea8ae8e81dd26d585805650a1e2b2466df55e801f92338da523f62"
    ),
    "Empty finding": "sha256:ed33ed0a2e1fd0e57cab45b707d68ce4d5547f384c3daa9dc3aa654e750fe1c0",
    "Bare finding": "sha256:008084dd5b32f3f5c3534a8d116829631d62be0718f1092deee083f0a4512525",
}

# Facts from a writer that labels no repository; they fall back to the
# directory name of repo_path.
UNLABELLED_FACTS = [
    {
        "kind": "symbol",
        "name": "main",
        "file": "main.go",
        "line": 3,
        "props": {"symbol_kind": "function"},
        "relations": [{"kind": "calls", "target": "helper"}],
    },
    {
        "kind": "symbol",
        "name": "helper",
        "file": "main.go",
        "line": 9,
        "props": {"symbol_kind": "function"},
    },
]
GOLDEN_UNLABELLED_HASHES = {
    "main": "sha256:9d055a17d6673f97970de0fa698c6ca8bbb6e836b4759bddb2327a1274e5b74a",
    "helper": "sha256:220cac306e84fbfcae9d39dbecf68fc1e9562d402203769c191d2fce1a824e24",
}


def _parse_and_map(snapshot_dir=DOCS_AND_FLOWS_DIR):
    facts, _receipt = parse_enola_snapshot(snapshot_dir)
    return facts, map_facts_to_data_points(facts, repo_path="/tmp/shop")


def _fixture_flows():
    return json.loads((DOCS_AND_FLOWS_DIR / "flows.json").read_text())["flows"]


def _by_type_and_name(data_points):
    return {(type(point).__name__, point.name): point for point in data_points}


def _fact_hashes(data_points):
    return {
        point.name: point.fact_hash
        for point in data_points
        if not isinstance(point, CodeRepository)
    }


def _edge_triples(edges, data_points):
    names = {point.id: point.name for point in data_points}
    return sorted(
        (names[source], relationship, names[target])
        for source, target, relationship, _properties in edges
    )


def _write_snapshot(tmp_path, facts, flows=None):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    if flows is not None:
        (tmp_path / "flows.json").write_text(json.dumps(flows))
    return tmp_path


def _mock_engine(monkeypatch, nodes=(), edges=()):
    engine = AsyncMock()
    engine.get_graph_data.return_value = (list(nodes), list(edges))
    engine.get_node.return_value = None
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    return engine


# --- a snapshot without docs or flows ingests exactly as before ---------------


def test_snapshot_without_docs_or_flows_ingests_exactly_as_before():
    facts, data_points = _parse_and_map(FIXTURES_DIR)
    edges, skipped = build_code_graph_edges(facts, repo_path="/tmp/shop")

    assert not any(fact["kind"] == "flow" for fact in facts)
    assert not any(isinstance(point, CodeFlow) for point in data_points)
    assert all(getattr(point, "doc", None) is None for point in data_points)
    assert _fact_hashes(data_points) == GOLDEN_FIXTURE_HASHES
    assert _edge_triples(edges, data_points) == GOLDEN_FIXTURE_EDGES
    assert skipped == 2


def test_docs_and_flows_leave_undocumented_facts_and_code_edges_untouched():
    facts, data_points = _parse_and_map()
    edges, skipped = build_code_graph_edges(facts, repo_path="/tmp/shop")

    undocumented = {
        name: point.fact_hash
        for (node_type, name), point in _by_type_and_name(data_points).items()
        if node_type not in ("CodeRepository", "CodeFlow") and point.doc is None
    }
    assert undocumented == GOLDEN_UNDOCUMENTED_HASHES
    code_edges = [edge for edge in edges if not edge[2].startswith("flow_")]
    assert _edge_triples(code_edges, data_points) == GOLDEN_CODE_EDGES
    # The call into a library (jwt.decode) is the only skip: every flow ref resolved.
    assert skipped == 1


def test_empty_or_non_text_docs_and_docs_on_insights_change_nothing():
    data_points = map_facts_to_data_points(NOT_DOCUMENTED_FACTS, repo_path="/tmp/shop")
    by_name = {point.name: point for point in data_points}

    assert all(getattr(point, "doc", None) is None for point in data_points)
    # An insight keeps describing itself with its explainer's prose, even empty.
    assert by_name["Call-graph hotspot: app/db.connect"].description == "A pinch point."
    assert by_name["Empty finding"].description == ""
    assert _fact_hashes(data_points) == GOLDEN_NOT_DOCUMENTED_HASHES


# --- docs ---------------------------------------------------------------------


def test_documented_symbol_carries_its_doc_as_doc_and_description():
    _facts, data_points = _parse_and_map()
    by_key = _by_type_and_name(data_points)

    validate = by_key[("CodeSymbol", "auth/tokens.validate_token")]
    assert validate.doc == (
        "Check a bearer token's signature and expiry, then return the user it was issued to."
    )
    assert validate.description == validate.doc
    # Like every enola prop, the doc stays in fact_properties too.
    assert validate.fact_properties["doc"] == validate.doc
    repository = by_key[("CodeSymbol", "orders/repository.OrderRepository")]
    assert repository.doc == repository.description == "Persist orders in the orders table."


def test_undocumented_symbol_has_no_doc_and_keeps_the_generic_description():
    _facts, data_points = _parse_and_map()
    serve = _by_type_and_name(data_points)[("CodeSymbol", "app.serve")]

    assert serve.doc is None
    assert serve.description == "symbol: language=python, symbol_kind=function"


def test_a_doc_on_any_kind_but_insight_is_read():
    facts = [{"kind": "route", "name": "/health", "repo": REPO, "props": {"doc": "Liveness."}}]

    route = map_facts_to_data_points(facts)[1]

    assert route.doc == route.description == "Liveness."


def test_docs_and_flows_add_no_embedded_field():
    assert CodeGraphEntity.model_fields["metadata"].default == {"index_fields": ["name"]}
    assert CodeFlow.model_fields["metadata"].default == {"index_fields": ["name"]}


# --- flows --------------------------------------------------------------------


def test_each_flow_becomes_a_fact_linked_to_its_entry_stages_and_routes():
    facts, _receipt = parse_enola_snapshot(DOCS_AND_FLOWS_DIR)
    flow = _fixture_flows()[0]

    # Appended after facts.jsonl's own facts, one per flow, in flows.json order.
    assert [fact["kind"] for fact in facts].count("flow") == 3
    assert [fact["name"] for fact in facts[-3:]] == [
        "POST /checkout",
        "app.main",
        "scripts/seed.main",
    ]

    def relation(kind, ref):
        return {"kind": kind, "target": ref["name"], "target_id": ref["id"]}

    relations = [relation("flow_entry", flow["entry"])]
    relations += [relation("flow_stage", stage) for stage in flow["stages"]]
    relations += [relation("flow_route", route) for route in flow["routes"]]
    checkout = facts[-3]
    assert {key: value for key, value in checkout.items() if key != "props"} == {
        "kind": "flow",
        "name": "POST /checkout",
        "repo": REPO,
        "file": "orders/checkout.py",
        "line": 14,
        "id": flow["id"],
        "relations": relations,
    }


def test_flows_json_becomes_one_code_flow_per_flow():
    _facts, data_points = _parse_and_map()
    flows = {flow["name"]: flow for flow in _fixture_flows()}
    flow_points = {point.name: point for point in data_points if isinstance(point, CodeFlow)}

    assert sorted(flow_points) == sorted(flows)
    checkout = flow_points["POST /checkout"]
    assert checkout.id == fact_node_id(REPO, "flow", "POST /checkout")
    assert (checkout.kind, checkout.repo, checkout.enola_id) == (
        "flow",
        REPO,
        flows["POST /checkout"]["id"],
    )
    # The entry's position stands for the flow's; the summary is its prose.
    assert (checkout.file_path, checkout.line) == ("orders/checkout.py", 14)
    assert checkout.description == flows["POST /checkout"]["summary"]
    assert checkout.doc is None
    # Everything but the fields the fact itself carries.
    assert checkout.fact_properties == {
        key: value
        for key, value in flows["POST /checkout"].items()
        if key not in ("id", "name", "repo")
    }
    assert isinstance(checkout.part_of, CodeRepository)
    assert checkout.part_of.name == REPO
    # side is kept only when true.
    assert flow_points["scripts/seed.main"].fact_properties["side"] is True
    assert "side" not in flow_points["app.main"].fact_properties


def test_flow_edges_join_each_flow_to_its_entry_stages_and_routes():
    facts, _data_points = _parse_and_map()
    edges, _skipped = build_code_graph_edges(facts, repo_path="/tmp/shop")

    def node(kind, name):
        return fact_node_id(REPO, kind, name)

    checkout = node("flow", "POST /checkout")
    main = node("flow", "app.main")
    seed = node("flow", "scripts/seed.main")
    save = node("symbol", "orders/repository.OrderRepository.save")
    assert {(edge[0], edge[2], edge[1]) for edge in edges if edge[2].startswith("flow_")} == {
        (checkout, "flow_entry", node("symbol", "orders/checkout.place_order")),
        (checkout, "flow_stage", node("symbol", "auth/tokens.validate_token")),
        (checkout, "flow_stage", node("symbol", "auth/tokens.decode")),
        (checkout, "flow_stage", save),
        (checkout, "flow_route", node("route", "/checkout")),
        # A main flow is named after its entry symbol; they stay two nodes.
        (main, "flow_entry", node("symbol", "app.main")),
        (main, "flow_stage", node("symbol", "app.serve")),
        (seed, "flow_entry", node("symbol", "scripts/seed.main")),
        (seed, "flow_stage", save),
    }


def test_flow_refs_resolve_by_fact_id_before_name(tmp_path):
    facts = [
        {"kind": "symbol", "name": "handlers.checkout", "file": "a.py", "repo": REPO, "id": ID_A},
        {"kind": "symbol", "name": "handlers.checkout", "file": "b.py", "repo": "web", "id": ID_B},
        {"kind": "symbol", "name": "orders.save", "file": "c.py", "repo": REPO},
    ]
    flows = {
        "flows": [
            {
                "name": "handlers.checkout",
                "repo": REPO,
                # By name, this repository's own handlers.checkout would win.
                "entry": {"name": "handlers.checkout", "id": ID_B},
                # A ref without an id resolves by name.
                "stages": [{"order": 1, "depth": 1, "name": "orders.save", "reach": 1}],
            }
        ]
    }
    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, flows))

    edges, skipped = build_code_graph_edges(parsed)

    flow = fact_node_id(REPO, "flow", "handlers.checkout")
    assert skipped == 0
    assert {(edge[0], edge[2], edge[1]) for edge in edges} == {
        (flow, "flow_entry", fact_node_id("web", "symbol", "handlers.checkout")),
        (flow, "flow_stage", fact_node_id(REPO, "symbol", "orders.save")),
    }


def test_flows_sharing_a_route_name_stay_separate_nodes_with_their_own_edges(tmp_path):
    """Two services serving GET /health: two flows, never one node with both edge sets."""
    facts = [
        {"kind": "symbol", "name": "svc_a/app.health", "file": "svc_a/app.py", "repo": REPO},
        {"kind": "symbol", "name": "svc_b/app.health", "file": "svc_b/app.py", "repo": REPO},
        {"kind": "symbol", "name": "svc_a/app.work", "file": "svc_a/app.py", "repo": REPO},
        {"kind": "symbol", "name": "svc_b/app.work", "file": "svc_b/app.py", "repo": REPO},
        {"kind": "symbol", "name": "orders.save", "file": "orders.py", "repo": REPO},
    ]
    flows = {
        "flows": [
            {
                "name": "GET /health",
                "repo": REPO,
                "id": ID_A,
                "summary": "svc_a health",
                "entry": {"name": "svc_a/app.health"},
                "stages": [{"order": 1, "depth": 1, "name": "svc_a/app.work"}],
            },
            {
                "name": "GET /health",
                "repo": REPO,
                "id": ID_B,
                "summary": "svc_b health",
                "entry": {"name": "svc_b/app.health"},
                "stages": [{"order": 1, "depth": 1, "name": "svc_b/app.work"}],
            },
            {"name": "POST /orders", "repo": REPO, "entry": {"name": "orders.save"}},
        ]
    }
    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, flows))
    data_points = map_facts_to_data_points(parsed, repo_path=f"/tmp/{REPO}")
    edges, skipped = build_code_graph_edges(parsed, repo_path=f"/tmp/{REPO}")

    flow_nodes = {point.name: point for point in data_points if isinstance(point, CodeFlow)}
    assert set(flow_nodes) == {
        "GET /health (svc_a/app.health)",
        "GET /health (svc_b/app.health)",
        # A flow whose name is not shared keeps it, and so its node id.
        "POST /orders",
    }
    assert flow_nodes["POST /orders"].id == fact_node_id(REPO, "flow", "POST /orders")
    assert skipped == 0
    flow_edges = {triple for triple in _edge_triples(edges, data_points) if triple[0] in flow_nodes}
    assert flow_edges == {
        ("GET /health (svc_a/app.health)", "flow_entry", "svc_a/app.health"),
        ("GET /health (svc_a/app.health)", "flow_stage", "svc_a/app.work"),
        ("GET /health (svc_b/app.health)", "flow_entry", "svc_b/app.health"),
        ("GET /health (svc_b/app.health)", "flow_stage", "svc_b/app.work"),
        ("POST /orders", "flow_entry", "orders.save"),
    }


def test_flows_sharing_a_name_and_an_entry_fall_back_to_their_ids(tmp_path):
    facts = [{"kind": "symbol", "name": "app.health", "file": "app.py", "repo": REPO}]
    flows = {
        "flows": [
            {"name": "GET /health", "repo": REPO, "id": flow_id, "entry": {"name": "app.health"}}
            for flow_id in (ID_A, ID_B)
        ]
        + [{"name": "GET /ping", "repo": REPO, "entry": {"name": "app.health"}} for _ in range(2)]
    }
    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, flows))

    names = sorted(fact["name"] for fact in parsed if fact["kind"] == "flow")

    assert names == [
        f"GET /health (app.health) ({ID_A})",
        f"GET /health (app.health) ({ID_B})",
        # Nothing tells these apart: numbered in flows.json order.
        "GET /ping (app.health) #1",
        "GET /ping (app.health) #2",
    ]


def test_a_flow_named_like_its_entry_leaves_name_resolution_unambiguous():
    facts = [
        {"kind": "symbol", "name": "jobs.send_receipt", "repo": REPO},
        {
            "kind": "symbol",
            "name": "orders.complete",
            "repo": REPO,
            # No target_ids: both resolve by name, exactly and by suffix.
            "relations": [
                {"kind": "calls", "target": "jobs.send_receipt"},
                {"kind": "calls", "target": "send_receipt"},
            ],
        },
        {
            "kind": "flow",
            "name": "jobs.send_receipt",
            "repo": REPO,
            "relations": [{"kind": "flow_entry", "target": "jobs.send_receipt"}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    entry = fact_node_id(REPO, "symbol", "jobs.send_receipt")
    assert skipped == 0
    assert sorted((edge[0], edge[2], edge[1]) for edge in edges) == sorted(
        [
            (fact_node_id(REPO, "symbol", "orders.complete"), "calls", entry),
            (fact_node_id(REPO, "flow", "jobs.send_receipt"), "flow_entry", entry),
        ]
    )


def test_flow_props_keep_the_documented_fields_that_are_set(tmp_path):
    flow = {
        "id": ID_C,
        "name": "POST /orders",
        "rank": 2,
        "repo": REPO,
        "mechanism": "route",
        "side": False,
        "routes": [{"name": "/orders", "id": ID_A, "method": "POST"}],
        "entry": {"name": "orders.create", "id": ID_B, "file": "orders.py", "line": 4},
        "summary": "POST /orders is served by orders.create.",
        "metrics": {"reach": 1, "modules": 1, "max_depth": 0, "unresolved_calls": 0},
        "modules": [{"name": "orders", "symbols": 1}],
        "same_reach_as": ["PUT /orders"],
        "nested_in": None,
        "renderer_hint": "not part of the contract",
    }
    facts = [{"kind": "symbol", "name": "orders.create", "repo": REPO, "id": ID_B}]
    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, {"flows": [flow]}))

    # The fact's own fields, a false side, a null and an undocumented field stay out.
    assert parsed[-1]["props"] == {
        key: value
        for key, value in flow.items()
        if key not in ("id", "name", "repo", "side", "nested_in", "renderer_hint")
    }


def test_flows_of_other_repositories_are_skipped(tmp_path):
    facts = [{"kind": "symbol", "name": "app.main", "file": "shop/app.py", "repo": REPO}]
    flows = {
        "flows": [
            {"name": "app.main", "repo": REPO, "entry": {"name": "app.main"}},
            # A multi-repo run writes every repository's flows into every output dir.
            {"name": "invoices.main", "repo": "billing", "entry": {"name": "invoices.main"}},
            {"name": "unlabelled.main", "entry": {"name": "unlabelled.main"}},
        ]
    }

    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, flows))

    assert [(fact["name"], fact["repo"]) for fact in parsed if fact["kind"] == "flow"] == [
        ("app.main", REPO)
    ]


def test_without_repo_labels_every_flow_is_kept_in_the_facts_repository(tmp_path):
    flows = {
        "flows": [
            {
                "name": "main",
                "repo": "elsewhere",
                "mechanism": "main",
                "entry": {"name": "main", "file": "main.go", "line": 3},
            }
        ]
    }
    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, UNLABELLED_FACTS, flows))

    data_points = map_facts_to_data_points(parsed, repo_path="/repos/myrepo")

    # The flow's own label did not become the fallback repository: the facts
    # keep their ids and hashes, and the flow joins them.
    assert "repo" not in parsed[-1]
    by_key = _by_type_and_name(data_points)
    assert {name: by_key[("CodeSymbol", name)].fact_hash for name in ("main", "helper")} == (
        GOLDEN_UNLABELLED_HASHES
    )
    assert by_key[("CodeSymbol", "main")].id == fact_node_id("myrepo", "symbol", "main")
    assert by_key[("CodeFlow", "main")].repo == "myrepo"
    assert [type(point).__name__ for point in data_points].count("CodeRepository") == 1


@pytest.mark.parametrize(
    "content, unparseable",
    [
        (b"{ not json", True),
        (b"\xff\xfe not utf-8", True),
        (b"[]", False),
        (b'"flows"', False),
        (b'{"flows": {"name": "app.main"}}', False),
        (b"{}", False),
    ],
)
def test_unreadable_or_odd_flows_json_adds_no_flows(tmp_path, monkeypatch, content, unparseable):
    _write_snapshot(tmp_path, [{"kind": "symbol", "name": "app.main", "repo": REPO}])
    (tmp_path / "flows.json").write_bytes(content)
    logger = MagicMock()
    monkeypatch.setattr(enola_module, "logger", logger)

    facts, _receipt = parse_enola_snapshot(tmp_path)

    assert [fact["kind"] for fact in facts] == ["symbol"]
    # Only a file that cannot be read is worth a warning.
    assert logger.warning.called is unparseable


def test_flows_without_a_usable_name_are_skipped_and_bad_refs_link_nothing(tmp_path):
    flows = {
        "flows": [
            "not an object",
            {"rank": 1},
            {"name": ""},
            {"name": 7},
            {
                "name": "worker.run",
                "repo": REPO,
                "id": "not-an-enola-id",
                "entry": "not an object",
                "stages": "not a list",
                "routes": [1, {"id": ID_A}, {"name": ""}],
            },
        ]
    }
    facts = [{"kind": "symbol", "name": "worker.run", "repo": REPO}]

    parsed, _receipt = parse_enola_snapshot(_write_snapshot(tmp_path, facts, flows))

    flow_facts = [fact for fact in parsed if fact["kind"] == "flow"]
    assert [fact["name"] for fact in flow_facts] == ["worker.run"]
    assert flow_facts[0]["relations"] == []
    assert "id" not in flow_facts[0]
    assert "file" not in flow_facts[0]


def test_receipt_counts_facts_jsonl_alone_and_the_flow_count_is_logged(monkeypatch):
    counted = []
    validate_receipt = enola_module.validate_receipt

    def _spy(receipt, snapshot_dir, fact_count=None):
        counted.append(fact_count)
        validate_receipt(receipt, snapshot_dir, fact_count=fact_count)

    logger = MagicMock()
    monkeypatch.setattr(enola_module, "validate_receipt", _spy)
    monkeypatch.setattr(enola_module, "logger", logger)

    facts, receipt = parse_enola_snapshot(DOCS_AND_FLOWS_DIR)

    assert counted == [receipt["fact_count"]] == [13]
    assert len(facts) == 16
    logger.warning.assert_not_called()
    message, *arguments = logger.info.call_args.args
    assert "flows.json" in message
    assert arguments[:3] == [16, 0, 3]


# Stands for the hash of the flows.json the test writes.
THESE_BYTES = "sha256:these-bytes"


@pytest.mark.parametrize(
    "receipt_fields, read",
    [
        # A release without the flows renderer (the pinned one) lists what it wrote.
        pytest.param(
            {
                "renderers": ["llm_context"],
                "output_hashes": {"facts.jsonl": "sha256:0", "insights.json": "sha256:1"},
            },
            False,
            id="not-written-by-this-run",
        ),
        pytest.param(
            {"renderers": ["llm_context", "flows"], "output_hashes": {"flows.json": THESE_BYTES}},
            True,
            id="written-by-this-run",
        ),
        pytest.param(
            {"renderers": ["flows"], "output_hashes": {"flows.json": "sha256:" + "0" * 64}},
            False,
            id="changed-since-this-run",
        ),
        # A receipt without output hashes vouches through its renderers.
        pytest.param({"renderers": ["flows"]}, True, id="flows-renderer-ran"),
        pytest.param({"renderers": ["llm_context"]}, False, id="flows-renderer-did-not-run"),
        pytest.param({}, False, id="pre-flows-writer"),
    ],
)
def test_flows_json_is_read_only_when_the_receipt_vouches_for_it(
    tmp_path, monkeypatch, receipt_fields, read
):
    facts = [{"kind": "symbol", "name": "app.main", "file": "app.py", "line": 1, "repo": REPO}]
    flows = {"flows": [{"name": "app.main", "repo": REPO, "entry": {"name": "app.main"}}]}
    _write_snapshot(tmp_path, facts, flows)
    flows_hash = "sha256:" + hashlib.sha256((tmp_path / "flows.json").read_bytes()).hexdigest()
    receipt = {"snapshot_id": "sha256:5e1f", "format_version": 1, "fact_count": 1}
    for key, value in receipt_fields.items():
        if key == "output_hashes":
            value = {
                name: flows_hash if digest == THESE_BYTES else digest
                for name, digest in value.items()
            }
        receipt[key] = value
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    logger = MagicMock()
    monkeypatch.setattr(enola_module, "logger", logger)

    parsed, _receipt = parse_enola_snapshot(tmp_path)

    flow_names = [fact["name"] for fact in parsed if fact["kind"] == "flow"]
    assert flow_names == (["app.main"] if read else [])
    # An earlier run's file is expected, not a fault: no warning.
    logger.warning.assert_not_called()
    skipped = [call for call in logger.info.call_args_list if "Ignoring flows.json" in call.args[0]]
    assert bool(skipped) is not read


# --- flows are code-graph nodes: written, swept and detagged -------------------


@pytest.mark.asyncio
async def test_stale_flows_are_swept_and_retagged_like_other_code_nodes(monkeypatch):
    checkout = str(fact_node_id(REPO, "flow", "POST /checkout"))
    removed_flow = str(fact_node_id(REPO, "flow", "GET /cart"))
    other_repo_flow = str(fact_node_id("billing", "flow", "GET /cart"))
    serve = str(fact_node_id(REPO, "symbol", "app.serve"))
    existing_nodes = [
        (checkout, {"type": "CodeFlow", "repo": REPO, "belongs_to_set": ["old-team"]}),
        # The entry point is gone from the code -> its flow must be swept.
        (removed_flow, {"type": "CodeFlow", "repo": REPO}),
        # Another repository's flow is out of this snapshot's scope.
        (other_repo_flow, {"type": "CodeFlow", "repo": "billing"}),
        (serve, {"type": "CodeSymbol", "repo": REPO}),
    ]
    # A stage the checkout flow no longer passes through.
    existing_edges = [(checkout, serve, "flow_stage", {})]
    engine = _mock_engine(monkeypatch, existing_nodes, existing_edges)

    await add_code_graph_edges(
        ["sentinel"],
        repo_path=f"/repos/{REPO}",
        snapshot_dir=DOCS_AND_FLOWS_DIR,
        node_set=["new-team"],
    )

    deleted_nodes = [
        node_id for call in engine.delete_nodes.await_args_list for node_id in call.args[0]
    ]
    assert deleted_nodes == [removed_flow]
    deleted_edges = {
        (edge.source_id, edge.target_id, edge.relationship_name)
        for call in engine.delete_edge_triples.await_args_list
        for edge in call.args[0]
    }
    assert (checkout, serve, "flow_stage") in deleted_edges
    tags_call = engine.remove_belongs_to_set_tags.await_args
    assert tags_call.args[0] == ["old-team"]
    assert checkout in tags_call.kwargs["node_ids"]
    assert other_repo_flow not in tags_call.kwargs["node_ids"]
    (added_edges,) = engine.add_edges.await_args.args
    entry = str(fact_node_id(REPO, "symbol", "orders/checkout.place_order"))
    assert (checkout, entry, "flow_entry") in {
        (str(edge[0]), str(edge[1]), edge[2]) for edge in added_edges
    }


@pytest.mark.asyncio
async def test_real_ladybug_stores_docs_and_flows_with_their_edges(tmp_path, monkeypatch):
    """Against a real Ladybug engine: the stored nodes carry the doc and the
    flow fields the CODE retriever reads, and the flow edges are written."""
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    engine = LadybugAdapter(db_path=str(tmp_path / "graph.lbug"))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )
    repo_path = f"/repos/{REPO}"

    data_points = await extract_code_graph(repo_path=repo_path, snapshot_dir=DOCS_AND_FLOWS_DIR)
    state = await add_code_graph_data_points(data_points)
    await add_code_graph_edges(state, repo_path=repo_path, snapshot_dir=DOCS_AND_FLOWS_DIR)

    # 13 facts and 3 flows, all new.
    assert state.node_delta["nodes_added"] == 16
    nodes, edges = await engine.get_graph_data()
    stored = {(properties["type"], properties["name"]): properties for _id, properties in nodes}
    flow = _fixture_flows()[0]
    checkout = stored[("CodeFlow", "POST /checkout")]
    assert (checkout["kind"], checkout["repo"], checkout["enola_id"]) == ("flow", REPO, flow["id"])
    assert (checkout["file_path"], checkout["line"]) == ("orders/checkout.py", 14)
    assert checkout["description"] == checkout["fact_properties"]["summary"] == flow["summary"]
    assert checkout["fact_properties"]["entry"] == flow["entry"]
    assert checkout["fact_properties"]["stages"] == flow["stages"]
    validate = stored[("CodeSymbol", "auth/tokens.validate_token")]
    assert validate["doc"] == validate["description"] == validate["fact_properties"]["doc"]
    assert stored[("CodeSymbol", "app.serve")]["doc"] is None

    names = {str(node_id): properties["name"] for node_id, properties in nodes}
    assert sorted(
        (names[str(source)], relationship, names[str(target)])
        for source, target, relationship, _properties in edges
        if relationship.startswith("flow_")
    ) == [
        ("POST /checkout", "flow_entry", "orders/checkout.place_order"),
        ("POST /checkout", "flow_route", "/checkout"),
        ("POST /checkout", "flow_stage", "auth/tokens.decode"),
        ("POST /checkout", "flow_stage", "auth/tokens.validate_token"),
        ("POST /checkout", "flow_stage", "orders/repository.OrderRepository.save"),
        ("app.main", "flow_entry", "app.main"),
        ("app.main", "flow_stage", "app.serve"),
        ("scripts/seed.main", "flow_entry", "scripts/seed.main"),
        ("scripts/seed.main", "flow_stage", "orders/repository.OrderRepository.save"),
    ]
