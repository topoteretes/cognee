"""Tests for what a run records in ``run_info["data"]`` and ``run_info["errored_data"]``.

``data`` names the documents the run completed, as a list of ``data_id``
strings; ``errored_data`` names the ones that failed. Both are read from the
per-item results the item runners yield, so a document skipped as already
completed is not listed and a row grows with the work done, not the corpus
(#4363). No input text is ever stored.
"""

from uuid import uuid4

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.utils import errored_run_info_data_ids, summarize_run_info_data

RUN, DATASET = uuid4(), uuid4()


def _item(status, data_id):
    extra = {"payload": "boom"} if status is PipelineRunErrored else {}
    run_info = status(pipeline_run_id=RUN, dataset_id=DATASET, dataset_name="d", **extra)
    return {"run_info": run_info, "data_id": data_id}


def test_before_any_item_ran_no_ids_are_recorded():
    assert summarize_run_info_data(None) == []
    assert summarize_run_info_data(["a raw input"]) == []
    assert errored_run_info_data_ids(None) == []


def test_only_documents_completed_in_this_run_are_listed():
    done, skipped, failed = uuid4(), uuid4(), uuid4()
    results = [
        _item(PipelineRunCompleted, done),
        _item(PipelineRunAlreadyCompleted, skipped),
        _item(PipelineRunErrored, failed),
    ]

    assert summarize_run_info_data(None, results) == [str(done)]
    assert errored_run_info_data_ids(results) == [str(failed)]


def test_an_item_that_failed_before_naming_its_document_is_not_listed():
    results = [_item(PipelineRunErrored, None), "not a per-item result"]

    assert errored_run_info_data_ids(results) == []
    assert summarize_run_info_data(None, results) == []
