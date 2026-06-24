from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.validate.models import (
    IssueSeverity,
    ValidationReport,
    ValidationStatus,
)
from cognee.api.v1.validate.checks import (
    check_dangling_edges,
    check_graph_vector_sync,
    check_isolated_nodes,
    check_uncognified_data,
)


# ── Model tests ──


def test_report_starts_healthy():
    report = ValidationReport(dataset="test")
    assert report.status == ValidationStatus.HEALTHY
    assert report.issues == []


def test_warning_degrades_to_degraded():
    report = ValidationReport()
    report.add_issue(IssueSeverity.WARNING, "test_warning", detail="something")
    assert report.status == ValidationStatus.DEGRADED


def test_error_sets_unhealthy():
    report = ValidationReport()
    report.add_issue(IssueSeverity.ERROR, "test_error", count=1)
    assert report.status == ValidationStatus.UNHEALTHY


def test_error_overrides_degraded():
    report = ValidationReport()
    report.add_issue(IssueSeverity.WARNING, "w1")
    report.add_issue(IssueSeverity.ERROR, "e1")
    assert report.status == ValidationStatus.UNHEALTHY


def test_info_keeps_healthy():
    report = ValidationReport()
    report.add_issue(IssueSeverity.INFO, "informational", count=5)
    assert report.status == ValidationStatus.HEALTHY


# ── Check tests ──


@pytest.mark.asyncio
async def test_dangling_edges_detected():
    nodes = [("n1", {"type": "Entity"}), ("n2", {"type": "Entity"})]
    edges = [
        ("n1", "n2", "relates_to", {}),
        ("n1", "n999", "points_to", {}),  # n999 doesn't exist
    ]
    graph_engine = AsyncMock()
    graph_engine.get_graph_data = AsyncMock(return_value=(nodes, edges))

    report = ValidationReport()
    await check_dangling_edges(report, graph_engine)

    assert report.status == ValidationStatus.UNHEALTHY
    assert len(report.issues) == 1
    assert report.issues[0].type == "dangling_edges"
    assert report.issues[0].count == 1


@pytest.mark.asyncio
async def test_no_dangling_edges():
    nodes = [("n1", {"type": "Entity"}), ("n2", {"type": "Entity"})]
    edges = [("n1", "n2", "relates_to", {})]
    graph_engine = AsyncMock()
    graph_engine.get_graph_data = AsyncMock(return_value=(nodes, edges))

    report = ValidationReport()
    await check_dangling_edges(report, graph_engine)

    assert report.status == ValidationStatus.HEALTHY
    assert len(report.issues) == 0


@pytest.mark.asyncio
async def test_isolated_nodes_detected():
    nodes = [
        ("n1", {"type": "Entity"}),
        ("n2", {"type": "Entity"}),
        ("n3", {"type": "Entity"}),
    ]
    edges = [("n1", "n2", "relates_to", {})]  # n3 is isolated
    graph_engine = AsyncMock()
    graph_engine.get_graph_data = AsyncMock(return_value=(nodes, edges))

    report = ValidationReport()
    await check_isolated_nodes(report, graph_engine)

    assert report.issues[0].type == "isolated_nodes"
    assert report.issues[0].count == 1


@pytest.mark.asyncio
async def test_structural_types_excluded_from_isolation():
    nodes = [
        ("n1", {"type": "Entity"}),
        ("n2", {"type": "Entity"}),
        ("chunk1", {"type": "DocumentChunk"}),  # structural, should not be flagged
    ]
    edges = [("n1", "n2", "relates_to", {})]
    graph_engine = AsyncMock()
    graph_engine.get_graph_data = AsyncMock(return_value=(nodes, edges))

    report = ValidationReport()
    await check_isolated_nodes(report, graph_engine)

    assert len(report.issues) == 0


@pytest.mark.asyncio
async def test_graph_vector_sync_detects_orphaned_vectors():
    nodes = [("n1", {"type": "Entity"})]
    edges = []
    graph_engine = AsyncMock()
    graph_engine.get_graph_data = AsyncMock(return_value=(nodes, edges))

    orphan_result = MagicMock()
    orphan_result.id = "n_orphan"

    real_result = MagicMock()
    real_result.id = "n1"

    vector_engine = AsyncMock()
    vector_engine.has_collection = AsyncMock(side_effect=lambda name: name == "Entity_name")
    vector_engine.retrieve = AsyncMock(return_value=[real_result, orphan_result])

    report = ValidationReport()
    await check_graph_vector_sync(report, graph_engine, vector_engine)

    orphan_issues = [i for i in report.issues if i.type == "orphaned_vector_entries"]
    assert len(orphan_issues) == 1
    assert orphan_issues[0].count == 1


@pytest.mark.asyncio
async def test_uncognified_data_detected():
    dataset_id = uuid4()

    data_record = SimpleNamespace(
        id=uuid4(),
        pipeline_status={"add_pipeline": {str(dataset_id): "Completed"}},
    )

    with patch("cognee.api.v1.validate.checks.get_relational_engine") as mock_engine:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [data_record]
        mock_session.execute = AsyncMock(return_value=mock_result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.return_value.get_async_session.return_value = ctx

        report = ValidationReport()
        await check_uncognified_data(report, dataset_id)

    assert report.summary["data_items"] == 1
    assert report.summary["uncognified_data_items"] == 1
    assert report.issues[0].type == "uncognified_data"


@pytest.mark.asyncio
async def test_uncognified_data_all_processed():
    dataset_id = uuid4()
    ds_str = str(dataset_id)

    data_record = SimpleNamespace(
        id=uuid4(),
        pipeline_status={"cognify_pipeline": {ds_str: "Completed"}},
    )

    with patch("cognee.api.v1.validate.checks.get_relational_engine") as mock_engine:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [data_record]
        mock_session.execute = AsyncMock(return_value=mock_result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.return_value.get_async_session.return_value = ctx

        report = ValidationReport()
        await check_uncognified_data(report, dataset_id)

    assert report.summary["uncognified_data_items"] == 0
    assert len(report.issues) == 0


@pytest.mark.asyncio
async def test_empty_graph_still_checks_uncognified():
    """An empty graph should still report uncognified data items."""
    report = ValidationReport()
    report.summary["graph_nodes"] = 0

    dataset_id = uuid4()
    data_record = SimpleNamespace(id=uuid4(), pipeline_status={})

    with patch("cognee.api.v1.validate.checks.get_relational_engine") as mock_engine:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [data_record]
        mock_session.execute = AsyncMock(return_value=mock_result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_engine.return_value.get_async_session.return_value = ctx

        await check_uncognified_data(report, dataset_id)

    assert report.issues[0].type == "uncognified_data"
