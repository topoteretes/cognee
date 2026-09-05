"""Unit tests for topological_rank stamping in _stamp_provenance.

DataPoint.topological_rank defaults to 0 (a sentinel — the field is never
written anywhere else in the codebase). When a pipeline runs, each task's
output should be stamped with a 1-based pipeline-stage index so the
visualization can lay nodes out left-to-right in pipeline order.

Uses inline reimplementation to avoid the cognee.__init__ import chain
(matches the pattern in test_provenance_stamping.py).
"""

from uuid import uuid4
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataPoint(BaseModel):
    """Minimal DataPoint replica for testing rank stamping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: object = Field(default_factory=uuid4)
    version: int = 1
    topological_rank: Optional[int] = 0
    source_pipeline: Optional[str] = None
    source_task: Optional[str] = None


class ChunkDP(DataPoint):
    text: str = ""
    contains: Optional[List] = None


def _stamp_provenance(
    data,
    pipeline_name,
    task_name,
    visited=None,
    task_index=None,
):
    """Copy of the rank-aware stamping logic from run_tasks_base.py."""
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

        if task_index is not None and task_index > 0:
            current_rank = getattr(data, "topological_rank", None)
            if current_rank is None or current_rank == 0:
                data.topological_rank = task_index

        for field_name in type(data).model_fields:
            field_value = getattr(data, field_name, None)
            if field_value is not None:
                _stamp_provenance(
                    field_value,
                    pipeline_name,
                    task_name,
                    visited,
                    task_index,
                )

    elif isinstance(data, (list, tuple)):
        for item in data:
            _stamp_provenance(item, pipeline_name, task_name, visited, task_index)


def _simulate_pipeline_run(task_names: List[str], outputs_per_task):
    """Simulate handle_task's rank-assignment loop.

    Returns the produced DataPoints in order so each task's rank can be asserted.

    ``outputs_per_task`` is a list of DataPoints (or lists of DataPoints) — one
    entry per task. Each entry is what that task "produces".
    """
    sequence: List[str] = []
    visited: set = set()
    produced = []

    for task_name, output in zip(task_names, outputs_per_task):
        if task_name not in sequence:
            sequence.append(task_name)
        task_index = sequence.index(task_name) + 1
        _stamp_provenance(output, "test_pipeline", task_name, visited, task_index)
        produced.append(output)

    return produced, sequence


# ── Tests ──


def test_rank_stamped_on_single_task_output():
    dp = DataPoint(id=uuid4())
    _stamp_provenance(dp, "p", "task_one", task_index=1)
    assert dp.topological_rank == 1


def test_rank_increments_across_tasks_in_pipeline_order():
    dp1, dp2, dp3 = DataPoint(id=uuid4()), DataPoint(id=uuid4()), DataPoint(id=uuid4())
    produced, sequence = _simulate_pipeline_run(
        ["classify_documents", "extract_chunks_from_documents", "extract_graph_from_data"],
        [dp1, dp2, dp3],
    )
    assert sequence == [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_graph_from_data",
    ]
    assert produced[0].topological_rank == 1
    assert produced[1].topological_rank == 2
    assert produced[2].topological_rank == 3


def test_same_task_repeated_keeps_same_rank():
    """When an upstream task streams multiple results, downstream task's
    handle_task is called multiple times. Each invocation must yield the
    same rank for that task — rank is position in pipeline, not call count."""
    dp_a = DataPoint(id=uuid4())
    dp_b = DataPoint(id=uuid4())
    dp_c = DataPoint(id=uuid4())

    sequence: List[str] = []
    visited: set = set()

    # Streaming pattern: chunk task fires once, extract task fires twice
    for task_name, output in [
        ("chunk", dp_a),
        ("extract", dp_b),
        ("extract", dp_c),
    ]:
        if task_name not in sequence:
            sequence.append(task_name)
        idx = sequence.index(task_name) + 1
        _stamp_provenance(output, "p", task_name, visited, idx)

    assert dp_a.topological_rank == 1
    assert dp_b.topological_rank == 2
    assert dp_c.topological_rank == 2


def test_rank_does_not_overwrite_explicit_nonzero():
    """A DataPoint produced upstream and re-encountered downstream keeps
    its earlier rank."""
    dp = DataPoint(id=uuid4(), topological_rank=2)
    _stamp_provenance(dp, "p", "later_task", task_index=5)
    assert dp.topological_rank == 2


def test_rank_zero_treated_as_unset():
    """topological_rank defaults to 0 on DataPoint; 0 must be treated as
    'never stamped' so legacy data still gets a real rank on first stamp."""
    dp = DataPoint(id=uuid4(), topological_rank=0)
    _stamp_provenance(dp, "p", "task", task_index=3)
    assert dp.topological_rank == 3


def test_rank_none_treated_as_unset():
    dp = DataPoint(id=uuid4(), topological_rank=None)
    _stamp_provenance(dp, "p", "task", task_index=4)
    assert dp.topological_rank == 4


def test_rank_not_written_when_task_index_none():
    """Pipelines that run without a PipelineContext (task_index=None) must
    not touch topological_rank."""
    dp = DataPoint(id=uuid4(), topological_rank=0)
    _stamp_provenance(dp, "p", "task", task_index=None)
    assert dp.topological_rank == 0


def test_rank_propagates_to_nested_datapoints():
    """A chunk containing entities — both should be stamped with the same rank
    on first pass."""
    entity = DataPoint(id=uuid4())
    chunk = ChunkDP(id=uuid4(), text="hi", contains=[(None, entity)])
    _stamp_provenance(chunk, "p", "task", task_index=7)
    assert chunk.topological_rank == 7
    assert entity.topological_rank == 7


def test_rank_walks_lists():
    dps = [DataPoint(id=uuid4()) for _ in range(3)]
    _stamp_provenance(dps, "p", "task", task_index=9)
    for dp in dps:
        assert dp.topological_rank == 9


def test_visited_set_prevents_rerank_in_later_task():
    """When a DataPoint has already been visited in a prior task, subsequent
    tasks must skip it entirely — rank set in stage 1 stays at 1 even if
    stage 2 encounters it."""
    dp = DataPoint(id=uuid4())
    visited: set = set()

    _stamp_provenance(dp, "p", "stage_one", visited, task_index=1)
    _stamp_provenance(dp, "p", "stage_two", visited, task_index=2)

    assert dp.topological_rank == 1
    assert dp.source_task == "stage_one"
