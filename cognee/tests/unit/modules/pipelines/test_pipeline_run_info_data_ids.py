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


def test_data_ids_include_every_data_item_an_entry_lists():
    single, first, second = uuid4(), uuid4(), uuid4()
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": single},
            {
                "run_info": _run_info(PipelineRunCompleted),
                "data_id": first,
                "data_ids": [first, second, single],
            },
        ],
    )

    assert run.data_ids == [single, first, second]


def test_run_infos_without_per_item_results_have_no_data_ids():
    assert _run_info(PipelineRunStarted, payload=["some text"]).data_ids is None
    assert _run_info(PipelineRunCompleted).data_ids is None
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


def test_run_infos_without_per_item_results_leave_data_ids_out_of_the_output():
    """Nested run infos and runs that just started carry no "data_ids": []."""
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": _run_info(PipelineRunCompleted), "data_id": uuid4()}],
    )

    assert "data_ids" not in json.loads(run.model_dump_json())["data_ingestion_info"][0]["run_info"]
    assert "data_ids" not in json.loads(_run_info(PipelineRunStarted).model_dump_json())
    # Per-item results without ids (incremental_loading=False) still say so.
    no_ids = _run_info(PipelineRunCompleted, data_ingestion_info=[])
    assert json.loads(no_ids.model_dump_json())["data_ids"] == []
    # Excluding the value must not hide the computed field from the OpenAPI schema.
    assert "data_ids" in PipelineRunInfo.model_json_schema(mode="serialization")["properties"]


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
