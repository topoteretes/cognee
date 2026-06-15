"""Tests for ``summarize_run_info_data``.

Guards against unbounded growth of the ``pipeline_runs`` table: large
payloads passed to ``add``/``cognify`` used to be stored verbatim in
``run_info["data"]`` on every run, with no reader and no size limit. The
helper now bounds that payload while preserving the existing behaviour for
empty input and lists of ``Data`` records.
"""

from uuid import uuid4

from cognee.modules.data.models import Data
from cognee.modules.pipelines.utils import summarize_run_info_data
from cognee.modules.pipelines.utils.summarize_run_info_data import MAX_RUN_INFO_DATA_CHARS


def test_empty_data_is_summarized_as_none():
    assert summarize_run_info_data(None) == "None"
    assert summarize_run_info_data("") == "None"
    assert summarize_run_info_data([]) == "None"


def test_list_of_data_records_is_reduced_to_ids():
    records = [Data(id=uuid4(), name="a"), Data(id=uuid4(), name="b")]
    result = summarize_run_info_data(records)
    assert result == [str(records[0].id), str(records[1].id)]


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
