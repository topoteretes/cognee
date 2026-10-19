"""``RememberResult`` reads a run's per-item results through ``data_ids``.

A ``PipelineRunCompleted`` carries no ``payload``; the items live in
``data_ingestion_info``. ``remember()`` used to walk that list by hand with
its own errored filter — it now takes the ``data_ids`` stamped on the run
info, or derives them with the same helper when the field is absent.
"""

from uuid import uuid4

from cognee.api.v1.remember.remember import RememberResult
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)


def _run_info(cls, **kwargs):
    return cls(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="notes", **kwargs)


def _resolved(run_info):
    result = RememberResult(status="running", dataset_name="x")
    result._resolve({run_info.dataset_id: run_info})
    return result


def test_items_come_from_the_per_item_results_when_data_ids_is_not_stamped():
    """cognify's run info has no data_ids: same ids, same errored filter."""
    stored, existing, failed = uuid4(), uuid4(), uuid4()
    run_info = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": stored},
            {"run_info": _run_info(PipelineRunAlreadyCompleted), "data_id": existing},
            {"run_info": _run_info(PipelineRunCompleted), "data_id": stored},  # duplicate
            {"run_info": _run_info(PipelineRunCompleted)},  # processed, id unknown
            {"run_info": _run_info(PipelineRunErrored), "data_id": failed},
        ],
    )

    result = _resolved(run_info)

    assert result.items == [{"id": str(stored)}, {"id": str(existing)}]
    assert result.items_processed == 4
    assert result.status == "completed"
    assert result.dataset_name == "notes"


def test_items_take_the_data_ids_stamped_on_the_run_info():
    """add() already computed the list; remember() does not re-derive it."""
    stamped = uuid4()
    run_info = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": _run_info(PipelineRunCompleted), "data_id": uuid4()}],
        data_ids=[stamped],
    )

    result = _resolved(run_info)

    assert result.items == [{"id": str(stamped)}]
    assert result.items_processed == 1
