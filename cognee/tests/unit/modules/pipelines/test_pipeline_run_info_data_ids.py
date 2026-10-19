"""``PipelineRunInfo.data_ids`` carries the ids of the Data rows a run named.

``add()`` always returned a ``PipelineRunCompleted`` whose per-item results
(``data_ingestion_info``) carried each stored row's id, but the shape was an
undocumented list of dicts, so callers re-derived ids from content instead.
``extract_data_ids`` reads that list; ``add()`` stamps the result on the run
info it returns, where it serializes as a plain field.
"""

import json
from uuid import UUID, uuid4

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunInfo,
    PipelineRunProgress,
    PipelineRunStarted,
    extract_data_ids,
)


def _run_info(cls, **kwargs):
    return cls(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds", **kwargs)


def test_extract_data_ids_follows_result_order_and_skips_errored_items():
    first, second, skipped = uuid4(), uuid4(), uuid4()
    info = [
        {"run_info": _run_info(PipelineRunCompleted), "data_id": first},
        # Dedup hit: the row already existed with this content — still the
        # id an add() caller wants back.
        {"run_info": _run_info(PipelineRunAlreadyCompleted), "data_id": second},
        {"run_info": _run_info(PipelineRunErrored), "data_id": skipped},
    ]

    assert extract_data_ids(info) == [first, second]


def test_extract_data_ids_can_leave_out_already_completed_items():
    """cognify counts a skipped-as-up-to-date item as NOT its own work."""
    processed, skipped = uuid4(), uuid4()
    info = [
        {"run_info": _run_info(PipelineRunCompleted), "data_id": processed},
        {"run_info": _run_info(PipelineRunAlreadyCompleted), "data_id": skipped},
    ]

    assert extract_data_ids(info, include_already_completed=False) == [processed]


def test_extract_data_ids_reads_run_infos_in_their_dict_form():
    """A JSON round-trip leaves the nested run infos as dicts; the errored
    filter must still apply to them."""
    kept, errored = uuid4(), uuid4()
    run = _run_info(
        PipelineRunErrored,
        data_ingestion_info=[
            {"run_info": _run_info(PipelineRunCompleted), "data_id": kept},
            {"run_info": _run_info(PipelineRunErrored), "data_id": errored},
        ],
    )
    revived = PipelineRunInfo.model_validate(json.loads(run.model_dump_json()))
    assert isinstance(revived.data_ingestion_info[0]["run_info"], dict)

    assert extract_data_ids(revived.data_ingestion_info) == [kept]


def test_extract_data_ids_tolerates_string_ids_duplicates_and_junk_entries():
    data_id = uuid4()
    info = [
        {"run_info": _run_info(PipelineRunCompleted), "data_id": str(data_id)},
        {"run_info": _run_info(PipelineRunCompleted), "data_id": data_id},  # duplicate
        {"run_info": _run_info(PipelineRunCompleted)},  # no data_id
        {"run_info": _run_info(PipelineRunCompleted), "data_id": None},
        {"run_info": _run_info(PipelineRunCompleted), "data_id": "not-a-uuid"},
        "not a dict",
    ]

    assert extract_data_ids(info) == [data_id]
    assert extract_data_ids(None) == []
    assert extract_data_ids("nonsense") == []


def test_extract_data_ids_reads_an_item_that_ended_up_under_several_rows():
    one, two, three = uuid4(), uuid4(), uuid4()
    info = [
        {"run_info": _run_info(PipelineRunCompleted), "data_id": one, "data_ids": [one, two]},
        {"run_info": _run_info(PipelineRunCompleted), "data_id": three},
    ]

    assert extract_data_ids(info) == [one, two, three]


def test_data_ids_default_to_none_and_stay_out_of_the_serialized_form():
    """Nothing but add() fills the field, so cognify/custom-pipeline results,
    progress ticks and the per-item run infos nested in a result carry no
    data_ids key at all — not even an empty one."""
    nested = _run_info(PipelineRunCompleted)
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": nested, "data_id": uuid4()}],
    )

    assert run.data_ids is None
    assert "data_ids" not in run.model_dump()
    dumped = json.loads(run.model_dump_json())
    assert "data_ids" not in dumped
    assert "data_ids" not in dumped["data_ingestion_info"][0]["run_info"]
    assert "data_ids" not in _run_info(PipelineRunStarted).model_dump()
    assert "data_ids" not in _run_info(PipelineRunProgress).model_dump()


def test_data_ids_serialize_with_the_model_once_set():
    """The HTTP add route returns the run info as its body; the ids ride along,
    only on the run info they were stamped on, and survive a JSON round-trip."""
    data_id = uuid4()
    nested = _run_info(PipelineRunCompleted)
    run = _run_info(
        PipelineRunCompleted,
        data_ingestion_info=[{"run_info": nested, "data_id": data_id}],
    )
    run.data_ids = extract_data_ids(run.data_ingestion_info)

    assert run.model_dump()["data_ids"] == [data_id]
    dumped = json.loads(run.model_dump_json())
    assert dumped["data_ids"] == [str(data_id)]
    assert "data_ids" not in dumped["data_ingestion_info"][0]["run_info"]
    assert PipelineRunInfo.model_validate(dumped).data_ids == [data_id]
    assert all(isinstance(value, UUID) for value in PipelineRunInfo.model_validate(dumped).data_ids)


def test_data_ids_is_one_plain_field_in_the_schema():
    """No computed field, so the OpenAPI schema does not split into
    PipelineRunInfo-Input / -Output variants."""
    assert "data_ids" in PipelineRunInfo.model_json_schema()["properties"]
    assert "data_ids" in PipelineRunInfo.model_json_schema(mode="validation")["properties"]
    assert "data_ids" not in PipelineRunInfo.model_computed_fields
