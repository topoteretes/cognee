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


def test_ingest_data_update_branch_preserves_label_when_absent():
    """A re-ingest without a label must not clear a stored one: every
    `data_point.label = ...` in the update branch has to sit under a
    `current_label is not None` guard, so absent means "leave unchanged".
    """
    source_path = Path(__file__).parents[4] / "tasks" / "ingestion" / "ingest_data.py"
    tree = ast.parse(source_path.read_text())

    def label_assigns(node):
        return [
            sub
            for sub in ast.walk(node)
            if isinstance(sub, ast.Assign)
            and isinstance(sub.targets[0], ast.Attribute)
            and sub.targets[0].attr == "label"
            and isinstance(sub.targets[0].value, ast.Name)
            and sub.targets[0].value.id == "data_point"
        ]

    guarded = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "current_label"
            and isinstance(node.test.ops[0], ast.IsNot)
        ):
            guarded.update(id(assign) for assign in label_assigns(node))

    all_assigns = label_assigns(tree)
    assert all_assigns, "expected a data_point.label assignment in the update branch"
    unguarded = [assign for assign in all_assigns if id(assign) not in guarded]
    assert not unguarded, (
        f"{len(unguarded)} unguarded data_point.label assignment(s): a re-ingest "
        "without a label would clear the stored label"
    )
