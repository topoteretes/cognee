import ast
from pathlib import Path

from cognee.modules.data.models import Data


def test_data_model_has_data_size_column_not_file_size():
    columns = {column.name for column in Data.__table__.columns}
    assert "data_size" in columns
    assert "file_size" not in columns


def test_ingest_data_update_branch_assigns_data_size():
    """Regression test for #3160: the update branch in store_data_to_dataset
    must assign to `data_point.data_size`, the actual mapped column, not the
    unmapped `file_size` attribute the create branch never used.
    """
    source_path = Path(__file__).parents[4] / "tasks" / "ingestion" / "ingest_data.py"
    tree = ast.parse(source_path.read_text())

    assigned_attrs = {
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "data_point"
    }

    assert "data_size" in assigned_attrs
    assert "file_size" not in assigned_attrs
