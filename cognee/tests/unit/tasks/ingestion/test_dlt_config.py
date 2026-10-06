"""``dlt_config`` groups the DLT-only options of add()/remember().

The bare keyword spellings stay accepted; the two never merge silently, and
the CSV loader receives the options through its ``preferred_loaders`` entry.
"""

import pytest

pytest.importorskip("dlt")

from cognee.tasks.ingestion.dlt_config import (  # noqa: E402
    DLT_OPTION_NAMES,
    resolve_dlt_options,
    with_csv_loader_options,
)


def test_nothing_given_is_nothing():
    assert resolve_dlt_options(None, {}) == {}
    assert resolve_dlt_options({}, {"node_set": ["x"]}) == {}


def test_grouped_and_loose_options_combine():
    options = resolve_dlt_options(
        {"temporal_columns": {"orders": ["order_date"]}},
        {"primary_key": "id", "node_set": ["ignored"]},
    )
    assert options == {"temporal_columns": {"orders": ["order_date"]}, "primary_key": "id"}


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="temporal_column"):
        resolve_dlt_options({"temporal_column": {}}, {})


def test_wrong_type_raises():
    with pytest.raises(ValueError, match="max_rows_per_table"):
        resolve_dlt_options({"max_rows_per_table": "many"}, {})


def test_same_option_both_ways_raises():
    with pytest.raises(ValueError, match="primary_key"):
        resolve_dlt_options({"primary_key": "id"}, {"primary_key": "id"})


def test_options_reach_the_csv_loader_entry():
    loaders = with_csv_loader_options(None, {"primary_key": "id"})
    assert loaders == {"dlt_csv_loader": {"primary_key": "id"}}

    loaders = with_csv_loader_options(
        {"dlt_csv_loader": {"write_disposition": "merge"}, "pypdf_loader": {}},
        {"temporal_columns": {}},
    )
    assert loaders == {
        "dlt_csv_loader": {"write_disposition": "merge", "temporal_columns": {}},
        "pypdf_loader": {},
    }


def test_no_options_leave_preferred_loaders_untouched():
    assert with_csv_loader_options(None, {}) is None
    assert with_csv_loader_options({"csv_loader": {}}, {}) == {"csv_loader": {}}


def test_option_also_under_the_loader_entry_raises():
    with pytest.raises(ValueError, match="primary_key"):
        with_csv_loader_options({"dlt_csv_loader": {"primary_key": "a"}}, {"primary_key": "b"})


def test_option_names_are_the_model_fields():
    assert DLT_OPTION_NAMES == {
        "primary_key",
        "write_disposition",
        "query",
        "max_rows_per_table",
        "column_value_columns",
        "temporal_columns",
    }
