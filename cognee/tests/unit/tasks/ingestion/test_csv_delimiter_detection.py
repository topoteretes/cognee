"""Delimiter detection for the DLT CSV route (create_dlt_source_from_csv).

Pure: no dlt pipeline runs. Only the detection helper is exercised, on small
fixtures that mirror the real failure — a semicolon-delimited file whose text
column is full of commas made pandas' default comma split ragged.
"""

import pytest

from cognee.tasks.ingestion.create_dlt_source import detect_csv_delimiter


def _write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_comma_is_the_default(tmp_path):
    path = _write(tmp_path, "id,name\n1,Ann\n2,Bob\n")
    assert detect_csv_delimiter(path) == ","


def test_semicolon_with_comma_heavy_column(tmp_path):
    # Mixed quoting, as produced by pandas: only cells containing quotes are quoted.
    path = _write(
        tmp_path,
        "transaction_id;date;items;discount\n"
        "TX-1;2025-02-03;[{'product': 'RAM', 'qty': 4}, {'product': 'Mouse', 'qty': 9}];0\n"
        "TX-2;2025-01-03;[{'product': 'Laptop', 'qty': 4}];263.84\n"
        "TX-3;2025-02-03;\"[{'product': 'HP 24\"\" Monitor', 'qty': 9}, {'product': 'X'}]\";1\n",
    )
    assert detect_csv_delimiter(path) == ";"


@pytest.mark.parametrize("delimiter", ["\t", "|"])
def test_tab_and_pipe(tmp_path, delimiter):
    path = _write(tmp_path, delimiter.join(["a", "b", "c"]) + "\n" + delimiter.join("123") + "\n")
    assert detect_csv_delimiter(path) == delimiter


def test_unparseable_or_single_column_falls_back_to_comma(tmp_path):
    assert detect_csv_delimiter(_write(tmp_path, "")) == ","
    assert detect_csv_delimiter(_write(tmp_path, "just one column\nper line\n")) == ","
