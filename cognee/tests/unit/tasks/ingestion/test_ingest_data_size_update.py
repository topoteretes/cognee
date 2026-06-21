"""Regression tests for #3160 — update path must set mapped data_size column."""

from sqlalchemy import inspect

from cognee.modules.data.models import Data


def test_data_model_uses_data_size_not_file_size():
    columns = set(inspect(Data).columns.keys())
    assert "data_size" in columns
    assert "file_size" not in columns

    data_point = Data(data_size=100)
    data_point.data_size = 200
    assert data_point.data_size == 200
