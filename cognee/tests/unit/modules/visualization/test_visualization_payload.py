"""build_visualization_payload / build_semantic_payload: the JSON API surface.

The one thing worth pinning here that test_preprocessor.py does not: the
whole payload has to survive a real ``json.dumps``, including
``provenance_index`` — the field CLO-401 flagged as unverified because it is
the one dataclass field that does not pass through ``_safe_json_embed`` on
the HTML path today. If a UUID, datetime or other non-JSON-safe value ever
reached it, this is where that would show up first.
"""

import json

from cognee.modules.visualization.cognee_network_visualization import (
    build_brain_summary_payload,
    build_semantic_payload,
    build_visualization_payload,
)
from cognee.tests.unit.modules.visualization.test_preprocessor import _alice_like_graph


def test_payload_is_json_serializable_including_provenance_index():
    payload = build_visualization_payload(_alice_like_graph())

    encoded = json.dumps(payload)

    # provenance_index is keyed off the same nodes that source_pipeline /
    # source_task above, so the fixture must actually populate it — an empty
    # dict here would mean this test passed for the wrong reason.
    assert payload["provenance_index"]
    assert json.loads(encoded)["provenance_index"] == payload["provenance_index"]


def test_payload_carries_every_preprocessed_graph_field():
    """Not the eight HTML tokens — asdict(pre) in full.

    pipeline_stages, edge_classes, bundles, provenance_index and
    has_meaningful_topological_rank never become template tokens on the HTML
    path (the view modules read them off `pre` directly through emit_js), so
    a curated subset would silently drop them from the JSON client.
    """
    payload = build_visualization_payload(_alice_like_graph())

    for field in (
        "nodes",
        "links",
        "color_maps",
        "schema_graph",
        "schema_data",
        "pipeline_stages",
        "edge_classes",
        "bundles",
        "provenance_index",
        "has_meaningful_topological_rank",
        "memory_map",
        "search_events",
    ):
        assert field in payload, f"missing {field}"


def test_search_events_default_to_an_empty_list():
    payload = build_visualization_payload(_alice_like_graph())

    assert payload["search_events"] == []


def test_search_events_pass_through_when_given():
    events = [{"kind": "search", "time": "2026-06-10T10:31:02", "qa_id": "q1"}]

    payload = build_visualization_payload(_alice_like_graph(), search_events=events)

    assert payload["search_events"] == events


def test_semantic_payload_has_no_embeddings_to_lay_out():
    """No embedding fetch is wired into this fixture, so the honest answer is
    null/null — not an exception, and not a fabricated layout.

    Run via asyncio.run rather than the asyncio marker so this file carries
    no plugin dependency beyond what the rest of the suite already uses.
    """
    import asyncio

    result = asyncio.run(build_semantic_payload(_alice_like_graph()))

    assert result == {"semantic_positions": None, "semantic_clusters": None}
    json.dumps(result)


def test_brain_summary_payload_matches_the_business_export_contract():
    """{"name", "nodes", "links", "node_set_colors"} — the exact per-dataset
    shape a static Business-view export already embeds under a top-level
    ``brainsData`` object keyed by dataset id (see CLO-401's description of
    that export), so a client built against it needs no adaptation to read
    this from a live endpoint. Nothing more, nothing less: schema_graph,
    memory_map and the rest of PreprocessedGraph are deliberately absent —
    they are one dataset's worth of detail, and this shape gets called once
    per dataset a user can read."""
    payload = build_brain_summary_payload("billing", _alice_like_graph())

    assert set(payload) == {"name", "nodes", "links", "node_set_colors"}
    assert payload["name"] == "billing"
    assert payload["nodes"]
    assert payload["links"]
    json.dumps(payload)
