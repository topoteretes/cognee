"""build_brains_summary_payload: the overview that never reads a graph.

Patches at the two boundaries the payload is assembled from — the relational
node-set read and the cached graph counts — because both are covered by their
own tests (test_get_datasets_graph_counts.py) and neither is what this adds.
What is new, and what this pins, is the assembly: the per-dataset shape, node
sets parsed out of the way ingestion writes them, and colors that agree with
the ones GET /visualize/brains hands out for the same node sets — the whole
point of the endpoint being a cheaper *substitute* for it.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.visualize.visualize import (  # noqa: F401
    build_brains_summary_payload as _build_brains_summary_payload,
)
from cognee.modules.data.methods import DatasetGraphCounts
from cognee.modules.visualization.cognee_network_visualization import build_brain_summary_payload

visualize_module = sys.modules["cognee.api.v1.visualize.visualize"]
counts_module = sys.modules["cognee.modules.data.methods.get_datasets_graph_counts"]


def _dataset(name: str):
    return SimpleNamespace(id=uuid4(), name=name, owner_id=uuid4())


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


def _discarding_engine():
    """A relational engine whose session accepts the count-cache write and
    drops it: what gets cached is get_datasets_graph_counts' own test's
    business, not this file's."""

    @asynccontextmanager
    async def get_async_session():
        yield SimpleNamespace(add=lambda _instance: None, commit=AsyncMock())

    return SimpleNamespace(get_async_session=get_async_session)


def _patches(datasets, node_sets, counts=None):
    if counts is None:
        counts = {dataset.id: DatasetGraphCounts() for dataset in datasets}
    return (
        patch.object(
            visualize_module,
            "get_all_user_permission_datasets",
            AsyncMock(return_value=datasets),
        ),
        patch.object(
            visualize_module, "_fetch_dataset_node_sets", AsyncMock(return_value=node_sets)
        ),
        patch.object(visualize_module, "get_datasets_graph_counts", AsyncMock(return_value=counts)),
    )


@pytest.mark.asyncio
async def test_each_readable_dataset_gets_the_documented_shape_keyed_by_id():
    billing, support = _dataset("billing"), _dataset("support")
    counts = {
        billing.id: DatasetGraphCounts(pipeline_run_id=uuid4(), num_nodes=42, num_edges=99),
        support.id: DatasetGraphCounts(),
    }

    ctx_a, ctx_b, ctx_c = _patches(
        [billing, support], {billing.id: [["slack"], ["notion"]]}, counts
    )
    with ctx_a, ctx_b, ctx_c:
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    assert set(payload) == {str(billing.id), str(support.id)}
    assert set(payload[str(billing.id)]) == {
        "name",
        "source_names",
        "node_count",
        "node_set_colors",
    }
    assert payload[str(billing.id)]["name"] == "billing"
    assert payload[str(billing.id)]["source_names"] == ["notion", "slack"]
    assert payload[str(billing.id)]["node_count"] == 42
    # A dataset that has never been cognified is present with a zero count,
    # not missing — a switcher still has to list it.
    assert payload[str(support.id)]["node_count"] == 0
    assert payload[str(support.id)]["source_names"] == []


@pytest.mark.asyncio
async def test_a_dataset_the_caller_cannot_read_never_reaches_the_payload():
    """Scoping is entirely delegated to get_all_user_permission_datasets — this
    endpoint does no filtering of its own. Pin that contract here: if a caller
    is not authorized for a dataset, get_all_user_permission_datasets simply
    never returns it, and the payload must not mention its id."""
    readable = _dataset("billing")
    unreadable = _dataset("someone_elses_dataset")

    ctx_a, ctx_b, ctx_c = _patches([readable], {})
    with ctx_a, ctx_b, ctx_c:
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    assert set(payload) == {str(readable.id)}
    assert str(unreadable.id) not in payload


@pytest.mark.asyncio
async def test_no_datasets_is_an_empty_payload_not_an_error():
    with patch.object(
        visualize_module, "get_all_user_permission_datasets", AsyncMock(return_value=[])
    ):
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    assert payload == {}


def _summary_patches(dataset):
    """Everything above the count cache, so the cost tests below can let the
    real get_datasets_graph_counts run."""
    return (
        patch.object(
            visualize_module,
            "get_all_user_permission_datasets",
            AsyncMock(return_value=[dataset]),
        ),
        patch.object(
            visualize_module, "_fetch_dataset_node_sets", AsyncMock(return_value={dataset.id: []})
        ),
    )


@pytest.mark.asyncio
async def test_a_warm_count_cache_makes_the_summary_cost_no_graph_access_at_all():
    """Half the reason this endpoint exists. Driven through the REAL
    get_datasets_graph_counts: patching that out would put the graph access
    behind the mock and prove nothing about it."""
    dataset = _dataset("billing")
    run_id = uuid4()
    cached = SimpleNamespace(id=run_id, num_nodes=42, num_edges=99, created_at=None)
    datasets_patch, node_sets_patch = _summary_patches(dataset)

    with (
        datasets_patch,
        node_sets_patch,
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: SimpleNamespace(pipeline_run_id=run_id)}),
        ),
        patch.object(
            counts_module, "_get_cached_metrics", AsyncMock(return_value={run_id: cached})
        ),
        patch.object(counts_module, "get_graph_engine", AsyncMock()) as graph_engine,
        patch.object(visualize_module, "fetch_dataset_graph_data", AsyncMock()) as fetch_graph,
    ):
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    graph_engine.assert_not_called()
    fetch_graph.assert_not_called()
    assert payload[str(dataset.id)]["node_count"] == 42


@pytest.mark.asyncio
async def test_a_cold_count_cache_costs_one_count_query_and_no_traversal():
    """The other half, stated honestly: the first call after an uncached
    cognify run does open the graph engine — for a count, once per run — not
    for the bounded node/link fetch /brains does on every single call."""
    dataset = _dataset("billing")
    run_id = uuid4()
    get_graph_metrics = AsyncMock(return_value={"num_nodes": 42, "num_edges": 99})
    get_graph_data = AsyncMock()
    datasets_patch, node_sets_patch = _summary_patches(dataset)

    with (
        datasets_patch,
        node_sets_patch,
        patch.object(
            counts_module,
            "_get_latest_cognify_runs",
            AsyncMock(return_value={dataset.id: SimpleNamespace(pipeline_run_id=run_id)}),
        ),
        patch.object(counts_module, "_get_cached_metrics", AsyncMock(return_value={})),
        patch.object(counts_module, "set_database_global_context_variables", _no_op_context),
        patch.object(
            counts_module,
            "get_graph_engine",
            AsyncMock(
                return_value=SimpleNamespace(
                    get_graph_metrics=get_graph_metrics, get_graph_data=get_graph_data
                )
            ),
        ),
        patch.object(counts_module, "get_relational_engine", lambda: _discarding_engine()),
        patch.object(visualize_module, "fetch_dataset_graph_data", AsyncMock()) as fetch_graph,
    ):
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    get_graph_metrics.assert_awaited_once_with(include_optional=False)
    get_graph_data.assert_not_awaited()
    fetch_graph.assert_not_called()
    assert payload[str(dataset.id)]["node_count"] == 42


@pytest.mark.asyncio
async def test_colors_match_the_ones_brains_gives_the_same_node_sets():
    dataset = _dataset("billing")

    ctx_a, ctx_b, ctx_c = _patches([dataset], {dataset.id: [["slack"], ["notion"]]})
    with ctx_a, ctx_b, ctx_c:
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    # The same two node sets, as /brains would see them on graph nodes.
    graph_data = (
        [
            ("n1", {"type": "Entity", "name": "a", "source_node_set": "slack"}),
            ("n2", {"type": "Entity", "name": "b", "source_node_set": "notion"}),
        ],
        [],
    )
    brains_colors = build_brain_summary_payload("billing", graph_data)["node_set_colors"]

    assert payload[str(dataset.id)]["node_set_colors"] == brains_colors


@pytest.mark.asyncio
async def test_a_multi_set_document_colors_the_joined_key_and_lists_names_apart():
    """Ingestion writes a multi-set document's sets to the graph comma-joined,
    so that joined string is the color key — while each name is its own
    source."""
    dataset = _dataset("billing")

    ctx_a, ctx_b, ctx_c = _patches([dataset], {dataset.id: [["slack", "notion"]]})
    with ctx_a, ctx_b, ctx_c:
        payload = await visualize_module.build_brains_summary_payload(user=MagicMock())

    assert payload[str(dataset.id)]["source_names"] == ["notion", "slack"]
    assert list(payload[str(dataset.id)]["node_set_colors"]) == ["slack, notion"]


def test_node_sets_are_parsed_from_the_json_string_ingestion_writes():
    assert visualize_module._parse_node_set('["slack", "notion"]') == ["slack", "notion"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        ("", []),
        ("[]", []),
        ("   ", []),
        ("not json", ["not json"]),
        ('"slack"', ["slack"]),
        (["slack", "notion"], ["slack", "notion"]),
        (["slack", None, 7, "  notion  "], ["slack", "notion"]),
        ('{"slack": 1}', []),
        (42, []),
    ],
)
def test_malformed_node_set_rows_degrade_instead_of_breaking_the_overview(raw, expected):
    """One unparsable row must not take down a payload covering every dataset
    the caller can read."""
    assert visualize_module._parse_node_set(raw) == expected
