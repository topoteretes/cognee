"""Enable delete for old tutorial notebooks

Revision ID: 1a58b986e6e1
Revises: 46a6ce2bd2b2
Create Date: 2025-12-17 11:04:44.414259

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1a58b986e6e1"
down_revision: Union[str, None] = "e1ec1dcb50b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def change_tutorial_deletable_flag(deletable: bool) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "notebooks" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("notebooks")}
    required_columns = {"name", "deletable"}
    if not required_columns.issubset(columns):
        return

    notebooks = sa.table(
        "notebooks",
        sa.Column("name", sa.String()),
        sa.Column("deletable", sa.Boolean()),
    )

    tutorial_name = "Python Development with Cognee Tutorial 🧠"

    bind.execute(
        notebooks.update().where(notebooks.c.name == tutorial_name).values(deletable=deletable)
    )


def upgrade() -> None:
    change_tutorial_deletable_flag(True)


def downgrade() -> None:
    change_tutorial_deletable_flag(False)
