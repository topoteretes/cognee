"""``PipelineRunInfo.data_ids`` exposes the ids of the data items an ``add()`` stored.

``add()`` always returned a ``PipelineRunCompleted`` whose per-item results
(``data_ingestion_info``) carried each stored data item's id, but the shape was
an undocumented list of dicts, so callers re-derived ids from content instead.
``add()`` now fills ``data_ids`` from those results; every other pipeline
leaves it ``None``.
"""

import json
from uuid import UUID, uuid4

from cognee.api.v1.add.add import _extract_data_ids
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
    PipelineRunStarted,
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

    assert _extract_data_ids(run.data_ingestion_info) == [first, second]


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

    assert _extract_data_ids(run.data_ingestion_info) == [data_id]


def test_data_ids_stay_none_unless_add_fills_them():
    """A cognify run also has per-item results, but its run info reports no data_ids."""
    cognify_run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunAlreadyCompleted), "data_id": uuid4()}
        ],
    )

    assert cognify_run.data_ids is None
    assert _run_info(PipelineRunStarted, payload=["some text"]).data_ids is None
    assert _extract_data_ids(None) == []
    assert _extract_data_ids("nonsense") == []


def test_data_ids_are_serialized_with_the_model():
    """The HTTP add route returns the run info as its body; the ids ride along."""
    data_id = uuid4()
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": _run_info(PipelineRunCompleted), "data_id": data_id}],
        data_ids=[data_id],
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

    assert _extract_data_ids(rebuilt.data_ingestion_info) == [stored]
