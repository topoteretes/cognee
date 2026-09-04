"""The enola 0.4.x snapshot contract: receipt validation, fact/target ids, insights, new kinds.

Covers what changed between the previously pinned 0.3.13 and 0.4.12: a
versioned receipt (format_version), writer fact ids and resolved relation
target_ids, insight evidence fact_ids plus metrics, the manifest/intent/
extraction/association/lint fact kinds, and the offline subprocess contract.
"""

import importlib
import json
from unittest.mock import AsyncMock

import pytest

from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES, _KIND_BY_TYPE
from cognee.tasks.code_graph.enola import (
    SUPPORTED_FORMAT_VERSIONS,
    EnolaSnapshotError,
    is_enola_id,
    parse_enola_snapshot,
    relation_target_id,
    validate_receipt,
)
from cognee.tasks.code_graph.extract_code_graph import (
    KIND_TO_MODEL,
    add_code_graph_edges,
    build_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
    receipt_projection,
)
from cognee.tasks.code_graph.models import (
    CodeAssociation,
    CodeExtractionAccount,
    CodeIntent,
    CodeLintFinding,
    ExternalDependency,
)

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

REPO = "acme/shop"
ID_A = "a" * 32
ID_B = "b" * 32
ID_C = "c" * 32
ID_UNKNOWN = "f" * 32

RECEIPT = {
    "snapshot_id": "sha256:1ceea542",
    "format_version": 1,
    "enola_version": "0.4.12",
    "extractor_version": "v259",
    "generated_at": "2026-09-03T05:54:25Z",
    "duration": "160.7ms",
    "repo_path": "/home/someone/src/shop",
    "git": {"ref": "main", "commit": "17f4e44c", "dirty": False},
    "extractors": ["manifests", "python"],
    "output_hashes": {"facts.jsonl": "sha256:3468"},
    "fact_count": 2,
    "insight_count": 0,
    "quality": {"files_seen": 6, "files_parsed": 5, "parse_errors": 0},
}


def _write_snapshot(tmp_path, facts, receipt=RECEIPT, insights=None):
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(fact) for fact in facts) + "\n")
    if receipt is not None:
        (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    if insights is not None:
        (tmp_path / "insights.json").write_text(json.dumps(insights))
    return tmp_path


TWO_FACTS = [
    {"kind": "symbol", "name": "app/db.Database", "file": "app/db.py", "repo": REPO, "id": ID_A},
    {"kind": "symbol", "name": "app/api.handler", "file": "app/api.py", "repo": REPO, "id": ID_B},
]


# --- ids ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (ID_A, True),
        ("b40cc8199deadc4199623e0a6a8c64b1", True),
        ("B40CC8199DEADC4199623E0A6A8C64B1", False),  # lowercase only
        ("a" * 31, False),
        ("a" * 33, False),
        ("g" * 32, False),
        ("", False),
        (None, False),
        (12345678901234567890123456789012, False),
    ],
)
def test_is_enola_id(value, expected):
    assert is_enola_id(value) is expected


@pytest.mark.parametrize(
    "relation, expected",
    [
        ({"kind": "calls", "target": "x", "target_id": ID_A}, ID_A),
        ({"kind": "calls", "target": "x"}, None),
        ({"kind": "calls", "target": "x", "target_id": "nope"}, None),
        ("not a dict", None),
    ],
)
def test_relation_target_id(relation, expected):
    assert relation_target_id(relation) == expected


# --- receipt -----------------------------------------------------------------


def test_supported_format_versions_is_exactly_one():
    assert SUPPORTED_FORMAT_VERSIONS == frozenset({1})


def test_format_version_1_and_historical_receipts_are_accepted(tmp_path):
    _write_snapshot(tmp_path, TWO_FACTS)
    facts, receipt = parse_enola_snapshot(tmp_path)
    assert len(facts) == 2
    assert receipt["format_version"] == 1

    # A receipt from a pre-0.4.10 writer has no format_version at all.
    historical = {key: value for key, value in RECEIPT.items() if key != "format_version"}
    _write_snapshot(tmp_path, TWO_FACTS, receipt=historical)
    facts, receipt = parse_enola_snapshot(tmp_path)
    assert len(facts) == 2
    assert "format_version" not in receipt


@pytest.mark.parametrize("format_version", [0, 2, "1", True, 1.0])
def test_unsupported_format_version_is_rejected(tmp_path, format_version):
    _write_snapshot(tmp_path, TWO_FACTS, receipt={**RECEIPT, "format_version": format_version})

    with pytest.raises(EnolaSnapshotError) as exc_info:
        parse_enola_snapshot(tmp_path)

    assert "format_version" in str(exc_info.value)
    assert "0.4.12" in str(exc_info.value)


def test_validate_receipt_tolerates_missing_or_odd_receipts(tmp_path):
    validate_receipt(None, tmp_path)
    validate_receipt({}, tmp_path, fact_count=3)
    # A count mismatch and quality signals are surfaced as logs, never raised.
    validate_receipt({**RECEIPT, "fact_count": 99}, tmp_path, fact_count=2)
    validate_receipt(
        {
            **RECEIPT,
            "quality": {
                "files_seen": 10,
                "files_parsed": 4,
                "parse_errors": 2,
                "parse_error_sample": [{"extractor": "python", "msg": "boom"}],
                "census": {"top_skip_causes": [{"cause": "claimed by python", "count": 1}]},
            },
        },
        tmp_path,
        fact_count=2,
    )
    validate_receipt({**RECEIPT, "quality": "not a dict", "fact_count": "2"}, tmp_path, 2)


def test_receipt_projection_keeps_provenance_and_drops_machine_specifics():
    projection = receipt_projection(RECEIPT)

    assert projection["format_version"] == 1
    assert projection["enola_version"] == "0.4.12"
    assert projection["extractor_version"] == "v259"
    assert projection["git"] == RECEIPT["git"]
    assert projection["quality"] == RECEIPT["quality"]
    assert projection["fact_count"] == 2
    for machine_specific in ("repo_path", "duration", "output_hashes", "snapshot_id"):
        assert machine_specific not in projection

    assert receipt_projection(None) is None
    assert receipt_projection({}) is None
    assert receipt_projection({"unrelated": 1}) is None


@pytest.mark.asyncio
async def test_add_code_graph_edges_stamps_receipt_projection(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, TWO_FACTS)
    engine = AsyncMock()
    engine.get_graph_data.return_value = ([], [])
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", lambda **kwargs: None
    )

    await add_code_graph_edges(["sentinel"], repo_path="/repos/shop", snapshot_dir=tmp_path)

    stamped = engine.add_nodes.await_args.args[0]
    assert [node.name for node in stamped] == [REPO]
    assert stamped[0].last_snapshot_id == RECEIPT["snapshot_id"]
    assert stamped[0].last_receipt["format_version"] == 1
    assert stamped[0].last_receipt["enola_version"] == "0.4.12"
    assert stamped[0].last_receipt["quality"]["files_parsed"] == 5
    assert "repo_path" not in stamped[0].last_receipt


# --- facts: ids, positions, new kinds ----------------------------------------


def test_enola_id_and_end_line_are_mapped_onto_nodes():
    facts = [
        {"kind": "symbol", "name": "x", "repo": REPO, "id": ID_A, "line": 3, "end_line": 9},
        {"kind": "symbol", "name": "y", "repo": REPO, "id": "not-a-hex-id", "end_line": "9"},
        {"kind": "symbol", "name": "z", "repo": REPO},
    ]

    by_name = {point.name: point for point in map_facts_to_data_points(facts, repo_path="/shop")}

    assert by_name["x"].enola_id == ID_A
    assert by_name["x"].line == 3
    assert by_name["x"].end_line == 9
    assert by_name["y"].enola_id is None
    assert by_name["y"].end_line is None
    assert by_name["z"].enola_id is None
    # Cognee's node identity is unchanged by the writer id.
    assert by_name["x"].id == fact_node_id(REPO, "symbol", "x")


def test_new_fact_kinds_map_to_typed_models():
    facts = [
        {
            "kind": "intent",
            "name": "delivery",
            "repo": REPO,
            "props": {"intent_kind": "layer", "layer_name": "delivery", "order": 0},
        },
        {
            "kind": "extraction",
            "name": "python:calls",
            "file": "shop",
            "repo": REPO,
            "props": {"extractor": "python", "language": "python"},
        },
        {
            "kind": "association",
            "name": "Order#items",
            "repo": REPO,
            "props": {"model": "Order", "macro": "has_many", "target": "items"},
        },
        {
            "kind": "lint",
            "name": "RUBOCOP-001",
            "file": "app/models/order.rb",
            "repo": REPO,
            "props": {"lint_engine": "rubocop", "lint_severity": "warning"},
        },
        {
            "kind": "dependency",
            "name": "pkg:pypi/requests",
            "file": "pyproject.toml",
            "repo": REPO,
            "props": {"type": "package", "ecosystem": "pypi", "pinned": False},
        },
    ]

    by_name = {point.name: point for point in map_facts_to_data_points(facts, repo_path="/shop")}

    assert isinstance(by_name["delivery"], CodeIntent)
    assert isinstance(by_name["python:calls"], CodeExtractionAccount)
    assert isinstance(by_name["Order#items"], CodeAssociation)
    assert isinstance(by_name["RUBOCOP-001"], CodeLintFinding)
    assert isinstance(by_name["pkg:pypi/requests"], ExternalDependency)
    assert by_name["pkg:pypi/requests"].fact_properties["ecosystem"] == "pypi"
    assert len(by_name) == 6  # five facts + the repository node


def test_retriever_vocabulary_matches_the_pipeline_models():
    """Every model the pipeline can write must be a type the CODE retriever reads."""
    assert {model.__name__ for model in KIND_TO_MODEL.values()} == set(CODE_NODE_TYPES)
    assert {kind: model.__name__ for kind, model in KIND_TO_MODEL.items()} == {
        kind: node_type for node_type, kind in _KIND_BY_TYPE.items()
    }


def test_unknown_kinds_are_dropped_without_failing():
    facts = [
        {"kind": "topic", "name": "orders.created", "repo": REPO},
        {
            "kind": "symbol",
            "name": "main",
            "repo": REPO,
            "relations": [{"kind": "calls", "target": "orders.created"}],
        },
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/shop")
    edges, skipped = build_code_graph_edges(facts, repo_path="/shop")

    assert [point.name for point in data_points] == [REPO, "main"]
    assert edges == []
    assert skipped == 1


# --- relations: target_id -----------------------------------------------------


def test_target_id_resolves_what_a_bare_name_cannot():
    facts = [
        {"kind": "symbol", "name": "helper", "file": "a.py", "repo": "repo-a", "id": ID_A},
        {"kind": "symbol", "name": "helper", "file": "b.py", "repo": "repo-b", "id": ID_B},
        {
            "kind": "symbol",
            "name": "main",
            "repo": "repo-c",
            "id": ID_C,
            "relations": [{"kind": "calls", "target": "helper", "target_id": ID_B}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    # By name alone, "helper" is ambiguous from repo-c (two other repos have
    # one) and would be skipped; the writer's target_id settles it.
    assert skipped == 0
    assert [(edge[0], edge[1], edge[2]) for edge in edges] == [
        (
            fact_node_id("repo-c", "symbol", "main"),
            fact_node_id("repo-b", "symbol", "helper"),
            "calls",
        )
    ]

    without_target_id = json.loads(json.dumps(facts))
    del without_target_id[2]["relations"][0]["target_id"]
    edges, skipped = build_code_graph_edges(without_target_id)
    assert edges == []
    assert skipped == 1


def test_unknown_target_id_falls_back_to_name_resolution():
    facts = [
        {"kind": "symbol", "name": "helper", "repo": REPO, "id": ID_A},
        {
            "kind": "symbol",
            "name": "main",
            "repo": REPO,
            "id": ID_B,
            "relations": [{"kind": "calls", "target": "helper", "target_id": ID_UNKNOWN}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    assert skipped == 0
    assert edges[0][1] == fact_node_id(REPO, "symbol", "helper")


def test_target_id_pointing_at_an_unmapped_kind_is_skipped_not_bound():
    facts = [
        {"kind": "topic", "name": "orders", "repo": REPO, "id": ID_A},
        {"kind": "symbol", "name": "orders", "repo": REPO, "id": ID_B},
        {
            "kind": "symbol",
            "name": "main",
            "repo": REPO,
            "id": ID_C,
            "relations": [{"kind": "calls", "target": "orders", "target_id": ID_A}],
        },
    ]

    edges, skipped = build_code_graph_edges(facts)

    # The writer bound the relation to the topic fact, which cognee has no
    # model for. The same-named symbol is a different fact — by name the
    # target is still unique among mapped facts, so name resolution binds
    # it, exactly as a pre-target_id snapshot would have.
    assert skipped == 0
    assert edges[0][1] == fact_node_id(REPO, "symbol", "orders")


def test_records_sharing_an_enola_id_collapse_onto_one_node():
    facts = [
        {"kind": "symbol", "name": "init", "file": "a.go", "repo": REPO, "id": ID_A},
        {"kind": "symbol", "name": "init", "file": "a.go", "repo": REPO, "id": ID_A},
        {
            "kind": "symbol",
            "name": "main",
            "repo": REPO,
            "id": ID_B,
            "relations": [{"kind": "calls", "target": "init", "target_id": ID_A}],
        },
    ]

    data_points = map_facts_to_data_points(facts, repo_path="/shop")
    edges, skipped = build_code_graph_edges(facts, repo_path="/shop")

    assert [point.name for point in data_points] == [REPO, "init", "main"]
    assert skipped == 0
    assert len(edges) == 1
    assert edges[0][1] == fact_node_id(REPO, "symbol", "init")


# --- insights -----------------------------------------------------------------


def test_insight_evidence_fact_id_disambiguates_and_metrics_ride_along(tmp_path):
    facts = [
        {
            "kind": "symbol",
            "name": "app/db.Database",
            "file": "app/db.py",
            "repo": REPO,
            "id": ID_A,
        },
        {
            "kind": "symbol",
            "name": "lib/db.Database",
            "file": "lib/db.py",
            "repo": REPO,
            "id": ID_B,
        },
    ]
    insights = [
        {
            "title": "Call-graph hotspot: Database",
            "source": "hotspots",
            "description": "A pinch point.",
            "confidence": 0.7,
            "metrics": {"fan_in": 9, "fan_out": 2},
            "suggested_actions": ["Split the class."],
            "evidence": [
                {"symbol": "Database", "fact_id": ID_B, "detail": "fan-in 9"},
                {"fact": "Database", "detail": "no id here: ambiguous by name"},
            ],
        },
        {
            "title": "Domain findings do not apply to this repository",
            "source": "domain",
            "description": "Nothing to grade.",
            "confidence": 1,
            "informational": True,
            "evidence": [{"detail": "examined: 0 associations"}],
        },
    ]
    _write_snapshot(tmp_path, facts, insights=insights)

    parsed, _receipt = parse_enola_snapshot(tmp_path)
    insight_facts = [fact for fact in parsed if fact["kind"] == "insight"]
    assert len(insight_facts) == 2
    hotspot, informational = insight_facts
    assert hotspot["props"]["metrics"] == {"fan_in": 9, "fan_out": 2}
    assert hotspot["props"]["suggested_actions"] == ["Split the class."]
    assert hotspot["relations"] == [
        {"kind": "evidences", "target": "Database", "target_id": ID_B},
        {"kind": "evidences", "target": "Database"},
    ]
    assert informational["props"]["informational"] is True
    assert informational["relations"] == []

    edges, skipped = build_code_graph_edges(parsed, repo_path="/shop")
    evidences = [edge for edge in edges if edge[2] == "evidences"]
    # The fact_id-carrying citation binds to lib/db.Database; the bare-name
    # one is ambiguous between the two *.Database symbols and is skipped.
    assert [edge[1] for edge in evidences] == [fact_node_id(REPO, "symbol", "lib/db.Database")]
    assert skipped == 1

    by_name = {point.name: point for point in map_facts_to_data_points(parsed, repo_path="/shop")}
    assert by_name["Call-graph hotspot: Database"].description == "A pinch point."
    assert by_name["Call-graph hotspot: Database"].fact_properties["metrics"]["fan_in"] == 9


# --- subprocess contract ------------------------------------------------------


@pytest.mark.asyncio
async def test_run_enola_generate_passes_the_repo_explicitly_and_stays_offline(
    monkeypatch, tmp_path
):
    fake_binary = tmp_path / "enola"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    monkeypatch.setenv("SOME_USER_VARIABLE", "kept")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        snapshot_dir = repo_path / ".enola"
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / "facts.jsonl").write_text('{"kind": "module", "name": "app"}\n')

        class _Process:
            returncode = 0

            async def communicate(self):
                return b"", b"enola: no mcp-arch.yaml in /repo, using built-in defaults\n"

        return _Process()

    monkeypatch.setattr(enola_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    snapshot_dir = await enola_module.run_enola_generate(repo_path)

    assert snapshot_dir == repo_path / ".enola"
    assert captured["args"] == (str(fake_binary), "--generate", str(repo_path))
    assert captured["kwargs"]["cwd"] == str(repo_path)
    env = captured["kwargs"]["env"]
    assert env["ENOLA_NO_UPDATE_CHECK"] == "1"
    assert env["ENOLA_NO_PROMPTS"] == "1"
    assert env["SOME_USER_VARIABLE"] == "kept"
