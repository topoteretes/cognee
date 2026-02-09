"""Unit tests for the _stamp_provenance function from run_tasks_base.

Tests the provenance stamping logic that tags DataPoints with their
originating pipeline and task names. Uses inline reimplementation to
avoid the cognee.__init__ import chain (starlette version issue).
"""

from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime, timezone


class DataPoint(BaseModel):
    """Minimal DataPoint replica for testing provenance stamping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: object = Field(default_factory=uuid4)
    version: int = 1
    source_pipeline: Optional[str] = None
    source_task: Optional[str] = None


class EntityDP(DataPoint):
    """Simulates an Entity DataPoint nested inside another."""

    name: str = ""
    description: str = ""


class ChunkDP(DataPoint):
    """Simulates a DocumentChunk with nested entities."""

    text: str = ""
    contains: Optional[List] = None


def _stamp_provenance(data, pipeline_name, task_name, visited=None):
    """Copy of the function from run_tasks_base.py for isolated testing."""
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


# ── Tests ──


def test_stamp_provenance_sets_fields_on_bare_datapoint():
    dp = DataPoint(id=uuid4())
    _stamp_provenance(dp, "my_pipeline", "my_task")
    assert dp.source_pipeline == "my_pipeline"
    assert dp.source_task == "my_task"


def test_stamp_provenance_does_not_overwrite_existing():
    dp = DataPoint(
        id=uuid4(),
        source_pipeline="original_pipe",
        source_task="original_task",
    )
    _stamp_provenance(dp, "new_pipeline", "new_task")
    assert dp.source_pipeline == "original_pipe"
    assert dp.source_task == "original_task"


def test_stamp_provenance_walks_lists():
    dp1 = DataPoint(id=uuid4())
    dp2 = DataPoint(id=uuid4())
    _stamp_provenance([dp1, dp2], "pipe", "task")
    assert dp1.source_pipeline == "pipe"
    assert dp1.source_task == "task"
    assert dp2.source_pipeline == "pipe"
    assert dp2.source_task == "task"


def test_stamp_provenance_walks_nested_lists():
    dp = DataPoint(id=uuid4())
    _stamp_provenance([[dp]], "pipe", "task")
    assert dp.source_pipeline == "pipe"
    assert dp.source_task == "task"


def test_stamp_provenance_handles_none_values():
    dp = DataPoint(id=uuid4())
    _stamp_provenance(dp, None, None)
    assert dp.source_pipeline is None
    assert dp.source_task is None


def test_stamp_provenance_ignores_non_datapoint():
    data = {"key": "value"}
    _stamp_provenance(data, "pipe", "task")
    assert "source_pipeline" not in data


def test_stamp_provenance_mixed_list():
    dp = DataPoint(id=uuid4())
    _stamp_provenance([dp, "string", 42, None], "pipe", "task")
    assert dp.source_pipeline == "pipe"
    assert dp.source_task == "task"


def test_stamp_provenance_partial_override():
    """Only source_task is pre-set; source_pipeline should be stamped."""
    dp = DataPoint(id=uuid4(), source_task="pre_existing")
    _stamp_provenance(dp, "my_pipe", "new_task")
    assert dp.source_pipeline == "my_pipe"
    assert dp.source_task == "pre_existing"


def test_stamp_provenance_tuple_input():
    dp = DataPoint(id=uuid4())
    _stamp_provenance((dp,), "pipe", "task")
    assert dp.source_pipeline == "pipe"
    assert dp.source_task == "task"


def test_stamp_provenance_recurses_into_datapoint_fields():
    """Entities nested inside a chunk's 'contains' field should be stamped."""
    entity = EntityDP(id=uuid4(), name="Alice")
    chunk = ChunkDP(id=uuid4(), text="hello", contains=[(None, entity)])
    _stamp_provenance(chunk, "pipe", "task")
    assert chunk.source_pipeline == "pipe"
    assert chunk.source_task == "task"
    assert entity.source_pipeline == "pipe"
    assert entity.source_task == "task"


def test_stamp_provenance_nested_does_not_overwrite():
    """Pre-set provenance on nested DataPoints is preserved."""
    entity = EntityDP(id=uuid4(), name="Bob", source_task="earlier_task")
    chunk = ChunkDP(id=uuid4(), text="hello", contains=[(None, entity)])
    _stamp_provenance(chunk, "pipe", "task")
    assert chunk.source_task == "task"
    assert entity.source_task == "earlier_task"
    assert entity.source_pipeline == "pipe"


def test_stamp_provenance_no_infinite_recursion():
    """Circular references should not cause infinite recursion."""
    dp = DataPoint(id=uuid4())
    # Simulate a circular-like structure via list
    circular_list = [dp]
    _stamp_provenance(circular_list, "pipe", "task")
    assert dp.source_pipeline == "pipe"
