"""The CODE navigation operations over a graph the code-graph pipeline ingested.

An Enola-style snapshot -- facts.jsonl with documented and undocumented
symbols, a served route and its handler, calls between symbols, receipt.json,
and a flows.json with a route flow and a main flow, every id computed the way
Enola computes it -- goes through the pipeline tasks into a real Ladybug
graph. The retriever then loads that graph as it does in production (one
get_filtered_graph_data call), so every assertion below reads the node and
edge shapes ingestion actually stores: the ``doc`` field, the CodeFlow rows
and their fact_properties, the flow_entry / flow_stage / flow_route edges and
the Enola ids that join a flow to its symbols.
"""

import hashlib
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.retrieval.code_retriever import (
    CODE_NODE_TYPES,
    CodeRetriever,
    CodeSeedNotFoundError,
    invalidate_code_graph_snapshot_cache,
)
from cognee.tasks.code_graph.enola import parse_enola_snapshot
from cognee.tasks.code_graph.extract_code_graph import (
    add_code_graph_data_points,
    add_code_graph_edges,
    build_code_graph_edges,
    extract_code_graph,
    fact_node_id,
)

graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")
search_pkg = importlib.import_module("cognee.api.v1.search")

REPO = "storefront"
REPO_PATH = f"/repos/{REPO}"
CREATE_DOC = "Create an order from the request body once the caller's token checks out."
VALIDATE_DOC = "Validate a bearer token's signature and expiry and return its claims."
SAVE = "orders/repository.OrderRepository.save"

# --- the snapshot ---------------------------------------------------------------------

# (name, file, line, end_line, symbol_kind, doc, calls, module). The call into
# a library (jwt.decode) names no fact, so it carries no target_id, as in Enola.
SYMBOLS = [
    (
        "api/orders.create_order",
        "api/orders.py",
        12,
        30,
        "function",
        CREATE_DOC,
        ["auth/tokens.validate_token", "orders/service.place_order"],
        "api",
    ),
    (
        "auth/tokens.validate_token",
        "auth/tokens.py",
        5,
        20,
        "function",
        VALIDATE_DOC,
        ["auth/tokens.decode_header"],
        "auth",
    ),
    (
        "auth/tokens.decode_header",
        "auth/tokens.py",
        25,
        None,
        "function",
        None,
        ["jwt.decode"],
        "auth",
    ),
    ("web/session.validateToken", "web/session.ts", 3, None, "function", None, [], "web"),
    (
        "orders/service.place_order",
        "orders/service.py",
        8,
        None,
        "function",
        "Place an order and reserve its stock.",
        [SAVE],
        "orders",
    ),
    (
        "orders/repository.OrderRepository",
        "orders/repository.py",
        4,
        None,
        "class",
        "Store orders in Postgres.",
        [],
        "orders",
    ),
    (SAVE, "orders/repository.py", 9, None, "method", None, [], "orders"),
    (
        "orders/archive.archive_orders",
        "orders/archive.py",
        6,
        None,
        "function",
        None,
        [SAVE],
        "orders",
    ),
    (
        "cli.main",
        "cli.py",
        1,
        None,
        "function",
        "Run the nightly order archive from the command line.",
        ["orders/archive.archive_orders"],
        ".",
    ),
    (
        "reports/monthly.render_report",
        "reports/monthly.py",
        3,
        None,
        "function",
        "Render the monthly sales report as HTML.",
        [],
        "reports",
    ),
    (
        "tests/test_tokens.test_validate_token",
        "tests/test_tokens.py",
        3,
        None,
        "function",
        None,
        ["auth/tokens.validate_token"],
        "tests",
    ),
]
MODULES = [".", "api", "auth", "orders", "reports", "tests", "web"]
DOCS = {name: doc for name, _file, _line, _end_line, _kind, doc, _calls, _module in SYMBOLS if doc}

# fact_hash of every fact of the snapshot written without doc props and
# without flows.json, computed with the code from before docs and flows were
# read (origin/dev 2fb7e0329): such a snapshot must ingest exactly as then.
GOLDEN_HASHES_WITHOUT_DOCS = {
    ".": "sha256:3114e3d70992da46042a83d43bf9f5b1b96f2a8277e5760ffbfaf270f3bf4a4e",
    "api": "sha256:f86c9d6f9021247eb767b5150c9d8d7962b641d0149dd31bb2531f718646ea0d",
    "auth": "sha256:4105931668dce70e7f7a8e3abfc08d513dc6be5ce5a3d37a4e7872e88b6d0629",
    "orders": "sha256:521c42f73840d49f3b91bf670458b5e569bd9d7e88e3d0baf16fcff3a20e3536",
    "reports": "sha256:d6d2f227d2ebbcab08f8b7f56af8a7266b113d98262c900c884792d2c85b50fa",
    "tests": "sha256:7ba2a7f4ad5cbb5dea954cb6f8cbf2cca5907d99e82fc924c64749dd8e9372d0",
    "web": "sha256:2e02f1b4ebe133772151e118f59cca218cd3d9d30909e3bc5f5eb664010dd54d",
    "/orders": "sha256:d536af8b2e89f7b243c8ae983e9d9685616d2fa0a9253b01d046c73bbb18600d",
    "api/orders.create_order": (
        "sha256:71f6ae5ae1be4e3942ee17107960b751e1ad2a6d642a9bf0ecec2dfbf9262b74"
    ),
    "auth/tokens.validate_token": (
        "sha256:fba28b45481898b3efbd13ff98e9c774eadc7eabe0996179cd64be7e260f2d08"
    ),
    "auth/tokens.decode_header": (
        "sha256:793d7e14f7464e59bd069571eaaafe25ea1031973885e02f02bd458be5583c93"
    ),
    "web/session.validateToken": (
        "sha256:91f355862a7700011104352656db80fb5bbfabc90ffdd43e8572fa67bcacc929"
    ),
    "orders/service.place_order": (
        "sha256:1653dd4bca329bd105fdf878564d8c0f25d5cc2b312c96331fa548125aa9c3c6"
    ),
    "orders/repository.OrderRepository": (
        "sha256:41954b054560b160e84946d4b4e705f2e0d0d5df39e2f8f4cc82b22139b1939f"
    ),
    "orders/repository.OrderRepository.save": (
        "sha256:2c84d5585d38fb18b7bc5262525d67f645be5475f80e7417522352f67e794ed4"
    ),
    "orders/archive.archive_orders": (
        "sha256:ee19746cbd78ba4b998fb7e5b0c90bba75911bc25d0a302f76059454c2905fc7"
    ),
    "cli.main": "sha256:f00e2b112c836eca5dd4dc6d93c3f81089314b48866e34ddbaae3efce8d09679",
    "reports/monthly.render_report": (
        "sha256:9ea1b3ee15b43191ecdd79aaf2fb358bda6b8f41b40b811adb715af4ed23d781"
    ),
    "tests/test_tokens.test_validate_token": (
        "sha256:280d79b3555029e80e10848efe7fd55107774e491b465f043d053383e20273e5"
    ),
}


def _enola_id(*parts: str) -> str:
    """An id the way Enola writes one: sha256 of the NUL-joined parts, 32 hex chars.

    A fact's parts are (repo, kind, name, file); a flow's are (repo, "flow",
    entry name, entry file).
    """
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


def _snapshot_facts(*, docs: bool = True) -> list[dict]:
    files = {name: file for name, file, *_rest in SYMBOLS}

    def declares(module):
        return {
            "kind": "declares",
            "target": module,
            "target_id": _enola_id(REPO, "module", module, module),
        }

    facts = [
        {
            "kind": "module",
            "name": module,
            "file": module,
            "repo": REPO,
            "props": {"language": "python"},
            "id": _enola_id(REPO, "module", module, module),
        }
        for module in MODULES
    ]
    facts.append(
        {
            "kind": "route",
            "name": "/orders",
            "file": "api/routes.py",
            "line": 10,
            "repo": REPO,
            "props": {"framework": "fastapi", "language": "python", "method": "POST"},
            "relations": [
                declares("api"),
                {
                    "kind": "handled_by",
                    "target": "api/orders.create_order",
                    "target_id": _enola_id(
                        REPO, "symbol", "api/orders.create_order", "api/orders.py"
                    ),
                },
            ],
            "id": _enola_id(REPO, "route", "/orders", "api/routes.py"),
        }
    )
    for name, file, line, end_line, symbol_kind, doc, calls, module in SYMBOLS:
        props = {"language": "typescript" if file.endswith(".ts") else "python"}
        if doc and docs:
            props["doc"] = doc
        props["symbol_kind"] = symbol_kind
        relations = []
        for target in calls:
            relation = {"kind": "calls", "target": target}
            if target in files:
                relation["target_id"] = _enola_id(REPO, "symbol", target, files[target])
            relations.append(relation)
        relations.append(declares(module))
        fact = {"kind": "symbol", "name": name, "file": file, "line": line, "repo": REPO}
        if end_line is not None:
            fact["end_line"] = end_line
        fact.update(props=props, relations=relations, id=_enola_id(REPO, "symbol", name, file))
        facts.append(fact)
    return facts


def _flows_document(facts: list[dict]) -> dict:
    by_name = {fact["name"]: fact for fact in facts}

    def ref(name):
        fact = by_name[name]
        result = {"name": name, "id": fact["id"], "file": fact["file"], "line": fact["line"]}
        if fact["props"].get("doc"):
            result["doc"] = fact["props"]["doc"]
        return result

    def stage(order, depth, name, reach, parent=None):
        head = {"order": order, "depth": depth}
        if parent is not None:
            head["parent"] = by_name[parent]["id"]
        return {**head, **ref(name), "reach": reach}

    return {
        "roots": {"main": 1, "route": 1},
        "leaves": 0,
        "total": 2,
        "flows": [
            {
                "id": _enola_id(REPO, "flow", "api/orders.create_order", "api/orders.py"),
                "name": "POST /orders",
                "rank": 1,
                "repo": REPO,
                "mechanism": "route",
                "routes": [{**ref("/orders"), "method": "POST"}],
                "entry": ref("api/orders.create_order"),
                "summary": (
                    "POST /orders is served by api/orders.create_order. It hands off to "
                    "tokens.validate_token, which calls tokens.decode_header, and to "
                    "service.place_order. Its 5 symbols sit in auth (2 symbols), orders (2) "
                    "and api (1)."
                ),
                "metrics": {"reach": 5, "modules": 3, "max_depth": 2, "unresolved_calls": 1},
                "modules": [
                    {"name": "auth", "symbols": 2},
                    {"name": "orders", "symbols": 2},
                    {"name": "api", "symbols": 1},
                ],
                "stages": [
                    stage(1, 1, "auth/tokens.validate_token", 2),
                    stage(
                        2, 2, "auth/tokens.decode_header", 1, parent="auth/tokens.validate_token"
                    ),
                    stage(3, 1, "orders/service.place_order", 2),
                    stage(4, 2, SAVE, 1, parent="orders/service.place_order"),
                ],
            },
            {
                "id": _enola_id(REPO, "flow", "cli.main", "cli.py"),
                "name": "cli.main",
                "rank": 2,
                "repo": REPO,
                "mechanism": "main",
                "entry": ref("cli.main"),
                "summary": (
                    "The program starts at cli.main. It hands off to archive.archive_orders, "
                    "which calls repository.OrderRepository.save. Its 3 symbols sit in orders "
                    "(2 symbols) and . (1)."
                ),
                "metrics": {"reach": 3, "modules": 2, "max_depth": 2, "unresolved_calls": 0},
                "modules": [{"name": "orders", "symbols": 2}, {"name": ".", "symbols": 1}],
                "stages": [
                    stage(1, 1, "orders/archive.archive_orders", 2),
                    stage(2, 2, SAVE, 1, parent="orders/archive.archive_orders"),
                ],
            },
        ],
    }


def _write_snapshot(directory, *, docs: bool = True, flows: bool = True, stale_flows: bool = False):
    """facts.jsonl and receipt.json, plus flows.json when ``flows`` is set.

    The receipt lists every file the run wrote with its hash, as Enola's
    does. ``stale_flows`` leaves an earlier run's flows.json beside a receipt
    whose run wrote none.
    """
    facts = _snapshot_facts(docs=docs)
    facts_bytes = "".join(json.dumps(fact) + "\n" for fact in facts).encode("utf-8")
    flows_bytes = json.dumps(_flows_document(facts)).encode("utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "facts.jsonl").write_bytes(facts_bytes)
    output_hashes = {"facts.jsonl": "sha256:" + hashlib.sha256(facts_bytes).hexdigest()}
    if flows:
        output_hashes["flows.json"] = "sha256:" + hashlib.sha256(flows_bytes).hexdigest()
    receipt = {
        "snapshot_id": output_hashes["facts.jsonl"],
        "format_version": 1,
        "enola_version": "dev",
        "generated_at": "2026-10-14T09:30:00Z",
        "repo_path": REPO_PATH,
        "extractors": ["python", "typescript"],
        "renderers": ["llm_context", "flows"] if flows else ["llm_context"],
        "output_hashes": output_hashes,
        "fact_count": len(facts),
        "insight_count": 0,
        "quality": {"files_seen": 11, "files_parsed": 11, "parse_errors": 0},
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))
    if flows or stale_flows:
        (directory / "flows.json").write_bytes(flows_bytes)
    return directory


# --- ingestion into a real graph, read back by the retriever ---------------------------


@pytest.fixture(autouse=True)
def clear_code_graph_snapshot_cache():
    invalidate_code_graph_snapshot_cache(all_entries=True)
    yield
    invalidate_code_graph_snapshot_cache(all_entries=True)


async def _ingest(
    directory, monkeypatch, *, docs: bool = True, flows: bool = True, stale_flows: bool = False
):
    """Run the code-graph tasks over the snapshot into a fresh Ladybug graph.

    The retriever reads the same engine, so its load is the production one.
    """
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    snapshot_dir = _write_snapshot(
        directory / "snapshot", docs=docs, flows=flows, stale_flows=stale_flows
    )
    engine = LadybugAdapter(db_path=str(directory / "graph.lbug"))
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(code_retriever_module, "get_graph_engine", AsyncMock(return_value=engine))
    data_points = await extract_code_graph(repo_path=REPO_PATH, snapshot_dir=snapshot_dir)
    state = await add_code_graph_data_points(data_points)
    await add_code_graph_edges(state, repo_path=REPO_PATH, snapshot_dir=snapshot_dir)
    invalidate_code_graph_snapshot_cache(all_entries=True)
    return engine


async def _stored_rows(engine):
    """The rows the retriever's own graph load returns."""
    return await engine.get_filtered_graph_data([{"type": [*CODE_NODE_TYPES, "CodeRepository"]}])


async def _run(config, query=""):
    return await CodeRetriever(config=config).get_retrieved_objects(query)


def _node_id(kind, name):
    return str(fact_node_id(REPO, kind, name))


def _flow(name):
    return next(
        flow for flow in _flows_document(_snapshot_facts())["flows"] if flow["name"] == name
    )


# --- acceptance 1: docs and flows are stored; without them nothing changes ------------


@pytest.mark.asyncio
async def test_ingestion_stores_docs_and_one_flow_per_flow_linked_to_its_refs(
    tmp_path, monkeypatch
):
    engine = await _ingest(tmp_path, monkeypatch)
    facts, _receipt = parse_enola_snapshot(tmp_path / "snapshot")
    _edges, skipped = build_code_graph_edges(facts, repo_path=REPO_PATH)

    # Every flow ref resolved by its Enola id; the one skip is the library call.
    assert [fact["name"] for fact in facts if fact["kind"] == "flow"] == [
        "POST /orders",
        "cli.main",
    ]
    assert skipped == 1

    nodes, edges = await _stored_rows(engine)
    stored = {(row["type"], row["name"]): row for _id, row in nodes}
    for name, *_rest in SYMBOLS:
        symbol = stored[("CodeSymbol", name)]
        assert symbol["doc"] == DOCS.get(name)
        if name in DOCS:
            assert symbol["description"] == symbol["fact_properties"]["doc"] == DOCS[name]

    stored_flows = {
        name: row for (node_type, name), row in stored.items() if node_type == "CodeFlow"
    }
    assert sorted(stored_flows) == ["POST /orders", "cli.main"]
    for name, row in stored_flows.items():
        flow = _flow(name)
        assert (row["kind"], row["repo"], row["enola_id"]) == ("flow", REPO, flow["id"])
        assert (row["file_path"], row["line"]) == (flow["entry"]["file"], flow["entry"]["line"])
        assert row["description"] == flow["summary"]
        for key in ("entry", "stages", "metrics", "modules"):
            assert row["fact_properties"][key] == flow[key]
    assert (
        stored_flows["POST /orders"]["fact_properties"]["routes"] == _flow("POST /orders")["routes"]
    )

    names = {str(node_id): row["name"] for node_id, row in nodes}
    assert sorted(
        (names[str(source)], relationship, names[str(target)])
        for source, target, relationship, _properties in edges
        if relationship.startswith("flow_")
    ) == [
        ("POST /orders", "flow_entry", "api/orders.create_order"),
        ("POST /orders", "flow_route", "/orders"),
        ("POST /orders", "flow_stage", "auth/tokens.decode_header"),
        ("POST /orders", "flow_stage", "auth/tokens.validate_token"),
        ("POST /orders", "flow_stage", SAVE),
        ("POST /orders", "flow_stage", "orders/service.place_order"),
        ("cli.main", "flow_entry", "cli.main"),
        ("cli.main", "flow_stage", "orders/archive.archive_orders"),
        ("cli.main", "flow_stage", SAVE),
    ]


@pytest.mark.asyncio
async def test_a_snapshot_without_docs_or_flows_ingests_as_before_and_says_so(
    tmp_path, monkeypatch
):
    documented_nodes, _edges = await _stored_rows(await _ingest(tmp_path / "with", monkeypatch))
    facts_with = await _run({"operation": "query_facts"})
    explored_with = await _run({"operation": "explore", "name": "api/orders.create_order"})
    # Ladybug stores a null doc on every undocumented node: no doc to a filter.
    assert any(row.get("doc", "") is None for _id, row in documented_nodes)
    with_doc = await _run({"operation": "query_facts", "property": "doc"})
    assert sorted(fact["name"] for fact in with_doc["facts"]) == sorted(DOCS)

    engine = await _ingest(tmp_path / "without", monkeypatch, docs=False, flows=False)
    nodes, _edges = await _stored_rows(engine)
    assert (await _run({"operation": "query_facts", "property": "doc"}))["total"] == 0

    def fact_hashes(rows):
        return {
            row["name"]: row["fact_hash"]
            for _id, row in rows
            if row["type"] in CODE_NODE_TYPES and row["type"] != "CodeFlow"
        }

    assert fact_hashes(nodes) == GOLDEN_HASHES_WITHOUT_DOCS
    assert not any(row["type"] == "CodeFlow" or row.get("doc") for _id, row in nodes)
    # Docs and flows beside them leave every undocumented fact untouched; a
    # documented one changes, which is what rewrites it with its doc.
    documented_hashes = fact_hashes(documented_nodes)
    for name, golden in GOLDEN_HASHES_WITHOUT_DOCS.items():
        assert (documented_hashes[name] == golden) is (name not in DOCS)

    # The existing operations answer as before, but for the docs themselves.
    def without_docs(fact):
        fact = {key: value for key, value in fact.items() if key not in ("doc", "description")}
        fact["properties"] = {
            key: value for key, value in fact.get("properties", {}).items() if key != "doc"
        }
        return fact

    facts_without = await _run({"operation": "query_facts"})
    assert [without_docs(fact) for fact in facts_with["facts"]] == [
        without_docs(fact) for fact in facts_without["facts"]
    ]
    explored = await _run({"operation": "explore", "name": "api/orders.create_order"})
    assert [without_docs(node) for node in explored_with["nodes"]] == [
        without_docs(node) for node in explored["nodes"]
    ]
    assert explored_with["edges"] == explored["edges"]

    capabilities = await _run({"operation": "capabilities"})
    assert capabilities["features"] == {"docs": False, "flows": False}
    with pytest.raises(CodeSeedNotFoundError) as unavailable:
        await _run({"operation": "flows"})
    assert (unavailable.value.error_code, unavailable.value.error_param) == (
        "feature_unavailable",
        "operation",
    )

    asked = await _run({"operation": "ask", "question": "Where do we validate auth tokens?"})
    assert asked["flows"] == []
    coverage = asked["coverage"]
    assert (coverage["docs_present"], coverage["flows_present"], coverage["unresolved_calls"]) == (
        False,
        False,
        None,
    )
    assert coverage["notes"] == [
        "This snapshot has no docstrings, so matching used names and paths only.",
        "This snapshot has no flows, so entry points were not traced.",
    ]


@pytest.mark.asyncio
async def test_a_flows_json_an_earlier_run_left_behind_is_not_ingested(tmp_path, monkeypatch):
    """A release without the flows renderer (the pinned one) writes no flows.json
    and leaves an earlier run's in place; its receipt does not list it."""
    engine = await _ingest(tmp_path, monkeypatch, docs=False, flows=False, stale_flows=True)
    nodes, _edges = await _stored_rows(engine)

    assert (tmp_path / "snapshot" / "flows.json").is_file()
    assert not any(row["type"] == "CodeFlow" for _id, row in nodes)
    capabilities = await _run({"operation": "capabilities"})
    assert capabilities["features"] == {"docs": False, "flows": False}
    assert capabilities["counts"]["flows"] == 0


# --- the operations over the ingested graph -------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_describe_the_ingested_graph(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    result = await _run({"operation": "capabilities"})

    assert {"capabilities", "search", "neighbors", "subgraph", "flows", "hubs", "ask"} <= set(
        result["operations"]
    )
    assert result["features"] == {"docs": True, "flows": True}
    # 7 modules, the route and 11 symbols; their 21 edges (7 calls, 12
    # declares, handled_by, has_method). Flow edges and part_of stay out.
    assert result["counts"] == {
        "nodes": 19,
        "edges": 21,
        "symbols": 11,
        "symbols_with_docs": len(DOCS),
        "flows": 2,
        "repositories": 1,
    }
    assert result["enola_versions"] == ["dev"]


@pytest.mark.asyncio
async def test_search_ranks_the_documented_token_validator_first(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    result = await _run({"operation": "search", "text": "validate token"})

    assert result["terms"] == ["valid", "token"]
    first, *rest = result["hits"]
    assert (first["id"], first["kind"], first["name"]) == (
        _node_id("symbol", "auth/tokens.validate_token"),
        "symbol",
        "auth/tokens.validate_token",
    )
    assert (first["file"], first["line"], first["doc"]) == ("auth/tokens.py", 5, VALIDATE_DOC)
    assert first["matched"] == ["name", "doc", "file"]
    assert all(first["score"] > hit["score"] for hit in rest)
    for hit in result["hits"]:
        assert {"id", "kind", "name", "file", "line", "score"} <= set(hit)
        assert hit.get("doc") == DOCS.get(hit["name"])
    names = [hit["name"] for hit in result["hits"]]
    # validateToken splits into the same terms; nothing unrelated matches,
    # and test code stays out unless asked for.
    assert names[1] == "web/session.validateToken"
    assert "reports/monthly.render_report" not in names
    assert "tests/test_tokens.test_validate_token" not in names
    flow_hit = next(hit for hit in result["hits"] if hit["kind"] == "flow")
    assert (flow_hit["id"], flow_hit["type"], flow_hit["enola_id"]) == (
        _node_id("flow", "POST /orders"),
        "CodeFlow",
        _flow("POST /orders")["id"],
    )

    for text in ("validateToken", "validate_token"):
        spelled = await _run({"operation": "search", "text": text})
        assert spelled["terms"] == ["valid", "token"]
        assert spelled["hits"][0]["name"] == "auth/tokens.validate_token"
    with_tests = await _run(
        {"operation": "search", "text": "validate token", "include_tests": True}
    )
    assert "tests/test_tokens.test_validate_token" in [hit["name"] for hit in with_tests["hits"]]


@pytest.mark.asyncio
async def test_flows_resolve_every_ref_to_the_ingested_nodes(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    listed = await _run({"operation": "flows"})
    assert [flow["name"] for flow in listed["flows"]] == ["POST /orders", "cli.main"]
    for flow in listed["flows"]:
        source = _flow(flow["name"])
        assert (flow["id"], flow["enola_id"]) == (_node_id("flow", flow["name"]), source["id"])
        assert flow["entry"]["node_id"] == _node_id("symbol", source["entry"]["name"])
        assert flow["summary"] == source["summary"]
        assert flow["metrics"] == source["metrics"]
    assert [route["node_id"] for route in listed["flows"][0]["routes"]] == [
        _node_id("route", "/orders")
    ]

    # Shown by Enola's own flow id: every stage joins its symbol.
    shown = await _run({"operation": "flows", "id": _flow("POST /orders")["id"]})
    stages = shown["flow"]["stages"]
    assert [(stage["order"], stage["depth"], stage["name"]) for stage in stages] == [
        (1, 1, "auth/tokens.validate_token"),
        (2, 2, "auth/tokens.decode_header"),
        (3, 1, "orders/service.place_order"),
        (4, 2, SAVE),
    ]
    assert [stage["node_id"] for stage in stages] == [
        _node_id("symbol", stage["name"]) for stage in stages
    ]
    assert stages[1]["parent"] == stages[0]["enola_id"]
    assert stages[0]["doc"] == VALIDATE_DOC
    by_name = await _run({"operation": "flows", "name": "cli.main"})
    assert by_name["flow"]["id"] == _node_id("flow", "cli.main")

    through = await _run({"operation": "flows", "symbol": SAVE})
    assert [(flow["name"], flow["via"], flow["distance"]) for flow in through["flows"]] == [
        ("POST /orders", "stage", 2),
        ("cli.main", "stage", 2),
    ]
    assert through["symbol"]["id"] == _node_id("symbol", SAVE)


@pytest.mark.asyncio
async def test_neighbors_gives_every_neighbour_its_relation_and_direction(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    result = await _run({"operation": "neighbors", "name": "auth/tokens.validate_token"})

    assert result["anchor"]["doc"] == VALIDATE_DOC
    assert [
        (entry["relation"], entry["direction"], entry["node"]["name"])
        for entry in result["neighbors"]
    ] == [
        ("calls", "in", "api/orders.create_order"),
        ("calls", "in", "tests/test_tokens.test_validate_token"),
        ("calls", "out", "auth/tokens.decode_header"),
        ("declares", "out", "auth"),
    ]
    assert result["flows"] == [
        {"id": _node_id("flow", "POST /orders"), "name": "POST /orders", "as": "stage"}
    ]
    assert (result["total"], result["truncated"]) == (4, False)


@pytest.mark.asyncio
async def test_subgraph_reports_each_cut_with_its_reason(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)
    # Seeded by the ids Enola wrote: 19 facts and two ids the graph lacks.
    enola_ids = [fact["id"] for fact in _snapshot_facts()] + ["e" * 32, "f" * 32]

    seeds = await _run({"operation": "subgraph", "seed_ids": enola_ids})
    assert (seeds["truncated"], seeds["truncation_reason"]) == (True, "max_seeds")
    assert seeds["truncation_reasons"] == ["max_seeds"]
    assert len(seeds["seeds"]) == 19
    assert seeds["seeds_missing"] == [{"seed": "e" * 32, "reason": "not_found", "candidates": []}]

    deep = await _run({"operation": "subgraph", "seeds": ["cli.main"], "depth": 5})
    assert (deep["truncated"], deep["truncation_reason"]) == (True, "max_depth")
    assert deep["limits"]["depth"] == 3

    crowded = await _run(
        {"operation": "subgraph", "seeds": ["api/orders.create_order"], "max_nodes": 3}
    )
    assert (crowded["truncated"], crowded["truncation_reason"]) == (True, "max_nodes")
    assert len(crowded["nodes"]) == 3

    within = await _run({"operation": "subgraph", "seeds": ["api/orders.create_order"]})
    assert (within["truncated"], within["truncation_reason"], within["truncation_reasons"]) == (
        False,
        None,
        [],
    )
    assert sorted((node["name"], node["depth"]) for node in within["nodes"]) == [
        ("/orders", 1),
        ("api", 1),
        ("api/orders.create_order", 0),
        ("auth/tokens.validate_token", 1),
        ("orders/service.place_order", 1),
    ]


@pytest.mark.asyncio
async def test_hubs_rank_the_ingested_call_graph(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    result = await _run({"operation": "hubs"})

    # Containment (declares, part_of) is not connectivity; test code is
    # neither listed nor counted (validate_token's test calls it).
    assert [
        (hub["name"], hub["degree"], hub["in_degree"], hub["out_degree"]) for hub in result["hubs"]
    ] == [
        ("api/orders.create_order", 3, 1, 2),
        (SAVE, 3, 3, 0),
        ("auth/tokens.validate_token", 2, 1, 1),
        ("orders/archive.archive_orders", 2, 1, 1),
        ("orders/service.place_order", 2, 1, 1),
        ("auth/tokens.decode_header", 1, 1, 0),
        ("cli.main", 1, 0, 1),
        ("orders/repository.OrderRepository", 1, 0, 1),
    ]
    with_tests = await _run({"operation": "hubs", "include_tests": True, "limit": 2})
    assert [(hub["name"], hub["in_degree"]) for hub in with_tests["hubs"]] == [
        ("api/orders.create_order", 1),
        ("auth/tokens.validate_token", 2),
    ]


@pytest.mark.asyncio
async def test_a_neo4j_shaped_load_answers_exactly_like_ladybug(tmp_path, monkeypatch):
    """Neo4j hands a nested dict back as JSON text and keeps no null property,
    so an undocumented fact has no ``doc`` key at all: same answers."""
    nodes, edges = await _stored_rows(await _ingest(tmp_path, monkeypatch))
    configs = [
        {"operation": "capabilities"},
        {"operation": "search", "text": "validate token"},
        {"operation": "neighbors", "name": "auth/tokens.validate_token"},
        {"operation": "subgraph", "seeds": ["api/orders.create_order"], "depth": 2},
        {"operation": "flows"},
        {"operation": "flows", "id": _flow("POST /orders")["id"]},
        {"operation": "flows", "symbol": SAVE},
        {"operation": "hubs"},
        {"operation": "ask", "question": "Where do we validate auth tokens?"},
        {"operation": "query_facts"},
        {"operation": "query_facts", "property": "doc"},
        {"operation": "explore", "name": "api/orders.create_order"},
    ]
    from_ladybug = [await _run(config) for config in configs]

    def neo4j_shaped(properties):
        return {
            key: json.dumps(value) if isinstance(value, dict) else value
            for key, value in properties.items()
            if value is not None
        }

    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(
        return_value=(
            [(node_id, neo4j_shaped(row)) for node_id, row in nodes],
            [(source, target, kind, neo4j_shaped(row)) for source, target, kind, row in edges],
        )
    )
    monkeypatch.setattr(code_retriever_module, "get_graph_engine", AsyncMock(return_value=engine))
    invalidate_code_graph_snapshot_cache(all_entries=True)

    assert [await _run(config) for config in configs] == from_ladybug


# --- acceptance 4: ask ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_seeds_flows_neighbourhood_citations_and_coverage(tmp_path, monkeypatch):
    await _ingest(tmp_path, monkeypatch)

    result = await _run({"operation": "ask", "question": "Where do we validate auth tokens?"})

    assert result["terms"] == ["valid", "auth", "token"]
    seeds = [seed["name"] for seed in result["seeds"]]
    assert seeds[0] == "auth/tokens.validate_token"
    assert result["seeds"][0]["doc"] == VALIDATE_DOC

    (flow,) = result["flows"]
    assert flow["name"] == "POST /orders"
    assert [stage["node_id"] for stage in flow["stages"]] == [
        _node_id("symbol", stage["name"]) for stage in flow["stages"]
    ]

    roles = {
        node["name"]: (node["role"], node["depth"]) for node in result["neighborhood"]["nodes"]
    }
    assert {name: roles[name] for name in seeds} == {name: ("seed", 0) for name in seeds}
    # The test calling validate_token is test code: left out unless asked for.
    assert "tests/test_tokens.test_validate_token" not in roles
    assert roles["orders/service.place_order"] == ("callee", 1)
    assert roles["/orders"] == ("neighbor", 1)
    assert ("/orders", "handled_by", "api/orders.create_order") in {
        (edge["source"], edge["type"], edge["target"]) for edge in result["neighborhood"]["edges"]
    }

    citations = {citation["name"]: citation for citation in result["citations"]}
    assert citations["auth/tokens.validate_token"] == {
        "node_id": _node_id("symbol", "auth/tokens.validate_token"),
        "kind": "symbol",
        "name": "auth/tokens.validate_token",
        "file": "auth/tokens.py",
        "line": 5,
        "end_line": 20,
        "cite": "auth/tokens.py:5",
        "doc": VALIDATE_DOC,
        "why": "seed",
    }
    assert (citations["orders/service.place_order"]["cite"], citations[SAVE]["why"]) == (
        "orders/service.py:8",
        "stage 4 of flow POST /orders",
    )
    assert citations["/orders"]["why"] == "linked to api/orders.create_order by handled_by"
    assert all(citation["file"] and citation["cite"] for citation in result["citations"])

    # The auth module matched too; it has no docstring to miss.
    symbols = [seed["name"] for seed in result["seeds"] if seed["kind"] == "symbol"]
    assert [name for name in seeds if name not in symbols] == ["auth"]
    assert result["coverage"] == {
        "code_graph": True,
        "docs_present": True,
        "seeds_with_docs": sum(1 for name in symbols if name in DOCS),
        "seeds": len(symbols),
        "flows_present": True,
        "unresolved_calls": 1,
        "truncated": False,
        "notes": [
            (
                "1 calls in these flows go to code outside the graph (libraries or callees the "
                "extractor could not resolve)."
            )
        ],
    }
    assert [step["code_query"]["operation"] for step in result["next"]] == [
        "flows",
        "neighbors",
        "subgraph",
    ]
    assert result["next"][0]["code_query"]["id"] == _node_id("flow", "POST /orders")
    assert result["truncated"] is False

    cut = await _run(
        {"operation": "ask", "question": "Where do we validate auth tokens?", "max_citations": 2}
    )
    assert len(cut["citations"]) == 2
    assert cut["truncated"] is cut["coverage"]["truncated"] is True
    assert cut["coverage"]["notes"][-1] == "Citations were cut at 2."


# --- acceptance 5: REST rejections carry a stable code ---------------------------------


@pytest.mark.asyncio
async def test_rest_rejections_carry_their_code_and_param(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognee.api.client import exception_handler
    from cognee.api.v1.search.routers.get_search_router import get_search_router
    from cognee.exceptions import CogneeApiError
    from cognee.modules.users.methods import get_authenticated_user

    rows = await _stored_rows(await _ingest(tmp_path, monkeypatch))
    # The app answers on its own event loop: serve it the rows the ingested
    # graph returned rather than the engine bound to this test's loop.
    engine = AsyncMock()
    engine.get_filtered_graph_data = AsyncMock(return_value=rows)

    async def search_code(*, query_text, code_query, **_kwargs):
        return [await CodeRetriever(config=code_query).get_retrieved_objects(query_text)]

    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.setattr(search_pkg, "search", search_code)
    app = FastAPI()
    app.include_router(get_search_router(), prefix="/api/v1/search")
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), email="test@example.com", is_active=True, tenant_id=None
    )

    def post(code_query):
        return client.post(
            "/api/v1/search",
            json={"query": "", "search_type": "CODE", "code_query": code_query},
        )

    def rejection(code_query):
        response = post(code_query)
        return response.status_code, response.json()["code"], response.json().get("param")

    with (
        patch.object(code_retriever_module, "get_graph_engine", AsyncMock(return_value=engine)),
        TestClient(app) as client,
    ):
        assert rejection({"operation": "ask_the_oracle"}) == (
            422,
            "unknown_operation",
            "operation",
        )
        assert rejection({"operation": "neighbors", "name": "payments.refund"}) == (
            422,
            "not_found",
            "name",
        )
        assert rejection({"operation": "neighbors", "id": "f" * 32}) == (422, "not_found", "id")
        # validate_token, validateToken and the test all contain "validate".
        assert rejection({"operation": "neighbors", "name": "validate"}) == (
            422,
            "ambiguous",
            "name",
        )
        assert rejection({"operation": "search", "text": "token", "limit": -1}) == (
            422,
            "invalid_argument",
            "limit",
        )
        answered = post({"operation": "neighbors", "name": "cli.main"})
        assert answered.status_code == 200
        assert answered.json()[0]["anchor"]["id"] == _node_id("symbol", "cli.main")
