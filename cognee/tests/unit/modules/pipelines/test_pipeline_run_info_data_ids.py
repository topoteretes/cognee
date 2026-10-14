"""``PipelineRunInfo.data_ids`` exposes the ids of the data items a run processed.

``add()`` always returned a ``PipelineRunCompleted`` whose per-item results
(``data_ingestion_info``) carried each stored row's id, but the shape was an
undocumented list of dicts, so callers re-derived ids from content instead.
The accessor makes the ids a first-class, serialized part of the result.
"""

import json
from uuid import UUID, uuid4

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
    PipelineRunStarted,
    extract_data_ids,
)


def _run_info(cls, **kwargs):
    return cls(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds", **kwargs)


def test_data_ids_follow_result_order_and_skip_errored_items():
    first, second, skipped = uuid4(), uuid4(), uuid4()
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": first},
            # Dedup hit: the row already existed with this content — still the
            # id the caller wants back.
            {"run_info": _run_info(PipelineRunAlreadyCompleted), "data_id": second},
            {"run_info": _run_info(PipelineRunErrored), "data_id": skipped},
        ],
    )

    assert run.data_ids == [first, second]


def test_data_ids_tolerate_string_ids_duplicates_and_junk_entries():
    data_id = uuid4()
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": str(data_id)},
            {"run_info": _run_info(PipelineRunCompleted), "data_id": data_id},  # duplicate
            {"run_info": _run_info(PipelineRunCompleted)},  # no data_id
            {"run_info": _run_info(PipelineRunCompleted), "data_id": "not-a-uuid"},
            "not a dict",
        ],
    )

    assert run.data_ids == [data_id]


def test_run_infos_without_per_item_results_have_no_data_ids():
    assert _run_info(PipelineRunStarted, payload=["some text"]).data_ids == []
    assert _run_info(PipelineRunCompleted).data_ids == []
    assert extract_data_ids(None) == []
    assert extract_data_ids("nonsense") == []


def test_data_ids_are_serialized_with_the_model():
    """The HTTP add route returns the run info as its body; the ids ride along."""
    data_id = uuid4()
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": _run_info(PipelineRunCompleted), "data_id": data_id}],
    )

    assert run.model_dump()["data_ids"] == [data_id]
    assert json.loads(run.model_dump_json())["data_ids"] == [str(data_id)]
    assert [UUID(value) for value in json.loads(run.model_dump_json())["data_ids"]] == [data_id]


def test_data_ids_skip_errored_items_after_a_json_round_trip():
    """A run info rebuilt from JSON has dict run_infos and string ids."""
    stored, failed = uuid4(), uuid4()
    run = _run_info(
        PipelineRunErrored,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": stored},
            {"run_info": _run_info(PipelineRunErrored), "data_id": failed},
        ],
    )

    rebuilt = PipelineRunInfo.model_validate(json.loads(run.model_dump_json()))

    assert rebuilt.data_ids == [stored]
