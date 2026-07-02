"""Unit tests for the provenance-enabled gate in handle_task (issue #3632).

The pipeline runtime stamps provenance onto every task output unless provenance
is disabled. handle_task now reads ``get_base_config().provenance_enabled`` once
and only calls ``_stamp_provenance`` when it is True:

    provenance_enabled = get_base_config().provenance_enabled
    ...
    if provenance_enabled:
        _stamp_provenance(result_data, ...)

These tests exercise that gate with an inline simulation of the call site — the
same convention used by test_provenance_stamping.py and
test_topological_rank_stamping.py to avoid the cognee.__init__ import chain. The
config flag itself (default on, disable-able via PROVENANCE_ENABLED) is covered
in test_edge_provenance.py.
"""

from uuid import uuid4
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DataPoint(BaseModel):
    """Minimal DataPoint replica."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: object = Field(default_factory=uuid4)
    version: int = 1
    source_pipeline: Optional[str] = None
    source_task: Optional[str] = None


def _stamp_provenance(data, pipeline_name, task_name, visited=None):
    """Copy of the stamping logic from run_tasks_base.py."""
    if visited is None:
        visited = set()

    if isinstance(data, DataPoint):
        obj_id = id(data)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if data.source_pipeline is None:
            data.source_pipeline = pipeline_name
        if data.source_task is None:
            data.source_task = task_name

        for field_name in data.model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance(field_value, pipeline_name, task_name, visited)

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance(item, pipeline_name, task_name, visited)


def _simulate_handle_task(output, provenance_enabled, task_name="extract_graph"):
    """Mirror of handle_task's gated call site."""
    if provenance_enabled:
        _stamp_provenance(output, "test_pipeline", task_name, visited=set())
    return output


# ── Tests ──


def test_provenance_stamped_when_enabled():
    dp = DataPoint(id=uuid4())
    _simulate_handle_task(dp, provenance_enabled=True)
    assert dp.source_pipeline == "test_pipeline"
    assert dp.source_task == "extract_graph"


def test_provenance_suppressed_when_disabled():
    dp = DataPoint(id=uuid4())
    _simulate_handle_task(dp, provenance_enabled=False)
    assert dp.source_pipeline is None
    assert dp.source_task is None


def test_disabled_gate_leaves_nested_datapoints_untouched():
    class ChunkDP(DataPoint):
        contains: Optional[list] = None

    entity = DataPoint(id=uuid4())
    chunk = ChunkDP(id=uuid4(), contains=[entity])

    _simulate_handle_task(chunk, provenance_enabled=False)

    assert chunk.source_task is None
    assert entity.source_task is None


def test_enabled_gate_stamps_nested_datapoints():
    class ChunkDP(DataPoint):
        contains: Optional[list] = None

    entity = DataPoint(id=uuid4())
    chunk = ChunkDP(id=uuid4(), contains=[entity])

    _simulate_handle_task(chunk, provenance_enabled=True)

    assert chunk.source_task == "extract_graph"
    assert entity.source_task == "extract_graph"
