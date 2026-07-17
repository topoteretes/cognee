"""Unit tests for cross-dataset reuse in run_tasks_data_item_incremental.

When a data item already completed a pipeline for another dataset and
`cross_dataset_reuse` is enabled, the incremental path must link the existing
artifacts instead of re-running the tasks — and fall back to full processing
whenever linking is not possible.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
)

run_tasks_data_item_module = importlib.import_module(
    "cognee.modules.pipelines.operations.run_tasks_data_item"
)
graph_methods_module = importlib.import_module("cognee.modules.graph.methods")

PIPELINE_NAME = "cognify_pipeline"


class _FakeResult:
    def __init__(self, data_point):
        self._data_point = data_point

    def scalar_one_or_none(self):
        return self._data_point


class _FakeSession:
    def __init__(self, data_point):
        self._data_point = data_point
        self.merged = []
        self.committed = False

    async def execute(self, _statement):
        return _FakeResult(self._data_point)

    async def merge(self, obj):
        self.merged.append(obj)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, data_point):
        self._data_point = data_point
        self.sessions = []

    def get_async_session(self):
        session = _FakeSession(self._data_point)
        self.sessions.append(session)
        return session


def _make_data(data_id, pipeline_status):
    data = Data(id=data_id)
    data.pipeline_status = pipeline_status
    return data


def _run_incremental(data_point, dataset, cross_dataset_reuse):
    return run_tasks_data_item_module.run_tasks_data_item_incremental(
        data_item=data_point,
        dataset=dataset,
        tasks=[],
        pipeline_name=PIPELINE_NAME,
        pipeline_id=uuid4(),
        pipeline_run_id=uuid4(),
        ctx=None,
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        cross_dataset_reuse=cross_dataset_reuse,
    )


@pytest.fixture
def telemetry_recorder(monkeypatch):
    """Replace run_tasks_with_telemetry with a recording no-op pipeline."""
    calls = []

    async def _fake_telemetry(**kwargs):
        calls.append(kwargs)
        if False:
            yield None

    monkeypatch.setattr(run_tasks_data_item_module, "run_tasks_with_telemetry", _fake_telemetry)
    return calls


@pytest.fixture
def link_recorder(monkeypatch):
    """Replace link_data_to_dataset with a recorder whose result is settable."""
    recorder = SimpleNamespace(calls=[], result=True)

    async def _fake_link(**kwargs):
        recorder.calls.append(kwargs)
        return recorder.result

    monkeypatch.setattr(graph_methods_module, "link_data_to_dataset", _fake_link)
    return recorder


@pytest.mark.asyncio
async def test_completed_for_other_dataset_links_and_skips_processing(
    monkeypatch, telemetry_recorder, link_recorder
):
    data_id = uuid4()
    source_dataset_id = uuid4()
    target_dataset = SimpleNamespace(id=uuid4(), name="dataset-b")

    data_point = _make_data(
        data_id,
        {PIPELINE_NAME: {str(source_dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
    )
    engine = _FakeEngine(data_point)
    monkeypatch.setattr(run_tasks_data_item_module, "get_relational_engine", lambda: engine)

    results = [
        result
        async for result in _run_incremental(data_point, target_dataset, cross_dataset_reuse=True)
    ]

    assert len(link_recorder.calls) == 1
    link_call = link_recorder.calls[0]
    assert link_call["data"] is data_point
    assert link_call["source_dataset_id"] == source_dataset_id
    assert link_call["target_dataset"] is target_dataset

    assert telemetry_recorder == [], "the task pipeline must not run for a linked item"

    assert len(results) == 1
    assert isinstance(results[0]["run_info"], PipelineRunAlreadyCompleted)
    assert results[0]["data_id"] == data_id

    # The item must be marked completed for the target dataset.
    assert (
        data_point.pipeline_status[PIPELINE_NAME][str(target_dataset.id)]
        == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
    )
    assert any(session.committed for session in engine.sessions)


@pytest.mark.asyncio
async def test_failed_link_falls_back_to_full_processing(
    monkeypatch, telemetry_recorder, link_recorder
):
    link_recorder.result = False

    data_id = uuid4()
    source_dataset_id = uuid4()
    target_dataset = SimpleNamespace(id=uuid4(), name="dataset-b")

    data_point = _make_data(
        data_id,
        {PIPELINE_NAME: {str(source_dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
    )
    engine = _FakeEngine(data_point)
    monkeypatch.setattr(run_tasks_data_item_module, "get_relational_engine", lambda: engine)

    results = [
        result
        async for result in _run_incremental(data_point, target_dataset, cross_dataset_reuse=True)
    ]

    assert len(link_recorder.calls) == 1
    assert len(telemetry_recorder) == 1, "linking failed, so the full pipeline must run"
    assert isinstance(results[-1]["run_info"], PipelineRunCompleted)


@pytest.mark.asyncio
async def test_reuse_disabled_processes_normally(monkeypatch, telemetry_recorder, link_recorder):
    data_id = uuid4()
    source_dataset_id = uuid4()
    target_dataset = SimpleNamespace(id=uuid4(), name="dataset-b")

    data_point = _make_data(
        data_id,
        {PIPELINE_NAME: {str(source_dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
    )
    engine = _FakeEngine(data_point)
    monkeypatch.setattr(run_tasks_data_item_module, "get_relational_engine", lambda: engine)

    results = [
        result
        async for result in _run_incremental(data_point, target_dataset, cross_dataset_reuse=False)
    ]

    assert link_recorder.calls == []
    assert len(telemetry_recorder) == 1
    assert isinstance(results[-1]["run_info"], PipelineRunCompleted)


@pytest.mark.asyncio
async def test_no_prior_completion_processes_normally(
    monkeypatch, telemetry_recorder, link_recorder
):
    data_id = uuid4()
    target_dataset = SimpleNamespace(id=uuid4(), name="dataset-b")

    data_point = _make_data(data_id, {})
    engine = _FakeEngine(data_point)
    monkeypatch.setattr(run_tasks_data_item_module, "get_relational_engine", lambda: engine)

    results = [
        result
        async for result in _run_incremental(data_point, target_dataset, cross_dataset_reuse=True)
    ]

    assert link_recorder.calls == []
    assert len(telemetry_recorder) == 1
    assert isinstance(results[-1]["run_info"], PipelineRunCompleted)


@pytest.mark.asyncio
async def test_completed_for_same_dataset_skips_without_linking(
    monkeypatch, telemetry_recorder, link_recorder
):
    data_id = uuid4()
    target_dataset = SimpleNamespace(id=uuid4(), name="dataset-b")

    data_point = _make_data(
        data_id,
        {PIPELINE_NAME: {str(target_dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}},
    )
    engine = _FakeEngine(data_point)
    monkeypatch.setattr(run_tasks_data_item_module, "get_relational_engine", lambda: engine)

    results = [
        result
        async for result in _run_incremental(data_point, target_dataset, cross_dataset_reuse=True)
    ]

    assert link_recorder.calls == [], "within-dataset skip must win over cross-dataset linking"
    assert telemetry_recorder == []
    assert len(results) == 1
    assert isinstance(results[0]["run_info"], PipelineRunAlreadyCompleted)
