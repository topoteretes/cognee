"""Tests for ``summarize_run_info_data``.

Guards against unbounded growth of the ``pipeline_runs`` table: large
payloads passed to ``add``/``cognify`` used to be stored verbatim in
``run_info["data"]`` on every run, with no reader and no size limit. The
helper bounds that payload. Lists of ``Data`` records (what ``cognify`` passes:
the dataset's every document, most of them skipped by an incremental run) are
summarized by what the run did with them, so a row grows with the work done
rather than with the corpus (#4363).
"""

from uuid import uuid4

from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.utils import summarize_run_info_data
from cognee.modules.pipelines.utils.summarize_run_info_data import MAX_RUN_INFO_DATA_CHARS

RUN, DATASET = uuid4(), uuid4()


def test_empty_data_is_summarized_as_none():
    assert summarize_run_info_data(None) == "None"
    assert summarize_run_info_data("") == "None"
    assert summarize_run_info_data([]) == "None"


def _results(records, *, processed=(), skipped=(), errored=()):
    """Per-item results as the item runners yield them: ``{"run_info", "data_id"}``."""
    rows = []
    for index in processed:
        rows.append(
            {
                "run_info": PipelineRunCompleted(
                    pipeline_run_id=RUN, dataset_id=DATASET, dataset_name="d"
                ),
                "data_id": records[index].id,
            }
        )
    for index in skipped:
        rows.append(
            {
                "run_info": PipelineRunAlreadyCompleted(
                    pipeline_run_id=RUN, dataset_id=DATASET, dataset_name="d"
                ),
                "data_id": records[index].id,
            }
        )
    for index in errored:
        rows.append(
            {
                "run_info": PipelineRunErrored(
                    pipeline_run_id=RUN, dataset_id=DATASET, dataset_name="d", payload="boom"
                ),
                "data_id": records[index].id,
            }
        )
    return rows


def test_a_data_list_before_any_item_ran_is_summarized_as_its_count():
    # The STARTED row: nothing has been processed yet, so only the size is known.
    records = [Data(id=uuid4(), name="a"), Data(id=uuid4(), name="b")]
    assert summarize_run_info_data(records) == {"dataset_data_count": 2}


def test_a_data_list_is_summarized_by_what_the_run_did_with_it():
    records = [Data(id=uuid4(), name=str(i)) for i in range(5)]
    results = _results(records, processed=(0, 1), skipped=(2, 3), errored=(4,))
    assert summarize_run_info_data(records, results) == {
        "dataset_data_count": 5,
        "processed_data_ids": [str(records[0].id), str(records[1].id)],
        "errored_data_ids": [str(records[4].id)],
        "skipped_completed": 2,
    }


def test_a_run_that_skips_every_document_stores_no_ids():
    # The #4363 case: an incremental cognify over a large dataset re-reads every
    # document and skips the completed ones; the row must not grow with the corpus.
    records = [Data(id=uuid4(), name=str(i)) for i in range(2_000)]
    summary = summarize_run_info_data(records, _results(records, skipped=range(2_000)))
    assert summary == {
        "dataset_data_count": 2_000,
        "processed_data_ids": [],
        "errored_data_ids": [],
        "skipped_completed": 2_000,
    }
    assert len(str(summary)) < 200


def test_an_item_that_failed_before_naming_its_document_is_not_listed():
    records = [Data(id=uuid4(), name="a")]
    # run_tasks records a gathered exception without a data_id; the error text
    # lives in run_info["error"], so the summary just does not list an id.
    results = [
        {
            "run_info": PipelineRunErrored(
                pipeline_run_id=RUN, dataset_id=DATASET, dataset_name="d", payload="boom"
            )
        }
    ]
    assert summarize_run_info_data(records, results)["errored_data_ids"] == []


def test_small_payload_is_preserved_verbatim():
    text = "Session trace: a small amount of text"
    assert summarize_run_info_data(text) == text


def test_large_payload_is_truncated_and_bounded():
    payload = "x" * (MAX_RUN_INFO_DATA_CHARS * 100)
    result = summarize_run_info_data(payload)

    assert result.startswith("x" * MAX_RUN_INFO_DATA_CHARS)
    assert f"[truncated, {len(payload)} chars total]" in result
    # The stored value must stay close to the cap, not scale with the input.
    assert len(result) < MAX_RUN_INFO_DATA_CHARS + 64
