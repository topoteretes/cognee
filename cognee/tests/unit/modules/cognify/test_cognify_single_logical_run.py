"""Every dataset gets exactly ONE cognify_pipeline run, with per-item routing.

cognify() makes a single run_pipeline call whose ``tasks`` is a resolver closure;
each data item resolves to the task list its kind requires (DLT manifests →
the deterministic DLT list, everything else → the standard list). run_tasks
executes mixed items under a single run lifecycle: one start record,
per-item task lists, one terminal status — never two runs (or two pipeline
names) for one dataset.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.cognify.cognify  # noqa: F401 — ensure the module (not the re-exported function) is importable via sys.modules
import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.cognify.routing import CognifyRoute, cognify_route_for
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunStarted,
)

# `from cognee.api.v1.cognify import cognify` would resolve to the re-exported
# cognify FUNCTION; grab the actual module object for patching.
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


def _manifest_item():
    return SimpleNamespace(external_metadata={"source": "dlt_source"}, extension=None)


def _legacy_item():
    # Pre-manifest per-row DLT record: unsupported, must fail loudly.
    return SimpleNamespace(id="legacy-1", external_metadata={"source": "dlt"}, extension=None)


def _text_item():
    return SimpleNamespace(external_metadata=None, extension="txt")


class TestCognifyRouting:
    def test_manifest_routes_to_dlt_source(self):
        assert cognify_route_for(_manifest_item()) is CognifyRoute.DLT_SOURCE

    def test_legacy_row_record_raises_instead_of_routing(self):
        """No legacy route exists; silently routing standard would LLM-process
        structured rows, so classification (and thus routing) fails loudly."""
        with pytest.raises(ValueError, match="no longer supported"):
            cognify_route_for(_legacy_item())

    def test_plain_document_routes_standard(self):
        assert cognify_route_for(_text_item()) is CognifyRoute.STANDARD


class TestCognifyMakesOneCall:
    """cognify() issues ONE executor call; tasks IS the per-item resolver."""

    async def _run_cognify(self, tasks="STANDARD_TASKS", **cognify_kwargs):
        calls = []

        async def _fake_executor(**kwargs):
            calls.append(kwargs)
            return {"ds": "run_info"}

        with (
            patch.object(
                cognify_module, "get_pipeline_executor", lambda run_in_background: _fake_executor
            ),
            patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value=tasks)),
            patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")),
        ):
            result = await cognify_module.cognify(
                datasets=["ds"],
                chunk_size=1024,
                config={"ontology_config": {"ontology_resolver": None}},
                **cognify_kwargs,
            )
        return calls, result

    @pytest.mark.asyncio
    async def test_single_call_under_cognify_name_with_resolver(self):
        calls, result = await self._run_cognify()

        (call,) = calls  # exactly one executor invocation, mixed or not
        assert call["pipeline_name"] == "cognify_pipeline"
        assert call["datasets"] == ["ds"]
        assert result == {"ds": "run_info"}

        # tasks IS the resolver: a plain list is just the degenerate case.
        resolver = call["tasks"]
        assert callable(resolver)
        assert resolver(_manifest_item()) == "DLT_TASKS"
        assert resolver(_text_item()) == "STANDARD_TASKS"

    @pytest.mark.asyncio
    async def test_unmapped_route_raises_instead_of_defaulting(self):
        """Every route is wired explicitly — a route with no task list is a
        programming error and raises, never silently runs the standard list."""
        calls, _ = await self._run_cognify()
        (call,) = calls
        resolver = call["tasks"]

        with patch.object(cognify_module, "cognify_route_for", return_value="UNMAPPED_ROUTE"):
            with pytest.raises(KeyError):
                resolver(_text_item())

    @pytest.mark.asyncio
    async def test_temporal_swaps_standard_route_only(self):
        """temporal_cognify replaces the fallback list; manifests still route DLT."""
        calls = []

        async def _fake_executor(**kwargs):
            calls.append(kwargs)
            return {}

        with (
            patch.object(
                cognify_module, "get_pipeline_executor", lambda run_in_background: _fake_executor
            ),
            patch.object(
                cognify_module, "get_temporal_tasks", new=AsyncMock(return_value="TEMPORAL_TASKS")
            ),
            patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")),
        ):
            await cognify_module.cognify(
                datasets=["ds"],
                temporal_cognify=True,
                chunk_size=1024,
                config={"ontology_config": {"ontology_resolver": None}},
            )

        (call,) = calls
        resolver = call["tasks"]
        assert resolver(_text_item()) == "TEMPORAL_TASKS"
        assert resolver(_manifest_item()) == "DLT_TASKS"


def _mock_run_tasks_plumbing(monkeypatch, dataset):
    session = MagicMock()
    session.get = AsyncMock(return_value=dataset)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock(spec=["get_async_session"])  # no push_to_s3 attribute
    engine.get_async_session.return_value = session_ctx
    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: engine)

    db_ctx = MagicMock()
    db_ctx.return_value.__aenter__ = AsyncMock(return_value=None)
    db_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", db_ctx)

    log_start = AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4()))
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", log_start)
    log_complete = AsyncMock()
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", log_complete)
    log_error = AsyncMock()
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", log_error)
    monkeypatch.setattr(
        run_tasks_module, "get_graph_engine", AsyncMock(return_value=SimpleNamespace())
    )
    return log_start, log_complete, log_error


@pytest.mark.asyncio
async def test_run_tasks_resolver_shares_one_run_lifecycle(monkeypatch):
    """Items resolved to different task lists share one run record and status."""
    dataset = SimpleNamespace(id=uuid4(), name="mixed_ds", owner_id=uuid4())
    log_start, log_complete, log_error = _mock_run_tasks_plumbing(monkeypatch, dataset)
    run_id = log_start.return_value.pipeline_run_id
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)

    processed = []

    async def _fake_item_run(data_item, ds, item_tasks, *args, **kwargs):
        processed.append((data_item, item_tasks))
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _fake_item_run)

    events = []
    async for event in run_tasks_module.run_tasks.__wrapped__(
        tasks=lambda item: "DLT_TASKS" if item.startswith("m") else "STANDARD_TASKS",
        dataset_id=dataset.id,
        data=["m1", "m2", "r1"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
    ):
        events.append(event)

    # One run record started and completed — never one per task list.
    assert log_start.await_count == 1
    assert log_complete.await_count == 1
    assert log_error.await_count == 0
    assert [type(e) for e in events] == [PipelineRunStarted, PipelineRunCompleted]
    assert all(e.pipeline_run_id == run_id for e in events)
    # log_pipeline_run_start received the real item list, not None.
    assert log_start.await_args.args[3] == ["m1", "m2", "r1"]

    # Every item ran with its own resolved task list.
    assert sorted(processed) == [
        ("m1", "DLT_TASKS"),
        ("m2", "DLT_TASKS"),
        ("r1", "STANDARD_TASKS"),
    ]


@pytest.mark.asyncio
async def test_run_tasks_validates_each_distinct_resolved_list_once(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    _mock_run_tasks_plumbing(monkeypatch, dataset)
    monkeypatch.setattr(
        run_tasks_module, "run_tasks_data_item", AsyncMock(return_value={"run_info": "ok"})
    )

    validated = []
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", validated.append)

    list_a, list_b = ["A"], ["B"]
    async for _ in run_tasks_module.run_tasks.__wrapped__(
        tasks=lambda item: list_a if item == "3" else list_b,
        dataset_id=dataset.id,
        data=["1", "2", "3"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="p",
    ):
        pass

    # Three items, two distinct lists → exactly two validations.
    assert len(validated) == 2
    assert list_a in validated and list_b in validated


@pytest.mark.asyncio
async def test_task_resolver_composes_with_pipeline_cache(monkeypatch):
    """The partition-era mutual exclusion is gone: a resolver and the
    pipeline cache are accepted together (qualification still runs per
    dataset inside _run_body)."""
    import cognee.modules.pipelines.operations.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), [])),
    )

    events = [
        event
        async for event in pipeline_module.run_pipeline(
            tasks=lambda item: ["LIST"],
            use_pipeline_cache=True,
            datasets=["ds"],
        )
    ]
    assert events == []  # no datasets resolved — and, critically, no ValueError


@pytest.mark.asyncio
async def test_run_pipeline_requires_tasks():
    import cognee.modules.pipelines.operations.pipeline as pipeline_module

    with pytest.raises(ValueError, match="requires tasks"):
        async for _ in pipeline_module.run_pipeline(tasks=None):
            pass
