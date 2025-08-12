"""incremental_loading

Revision ID: 1daae0df1866
Revises: b9274c27a25a
Create Date: 2025-08-12 13:14:12.515935

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.mutable import MutableDict

# revision identifiers, used by Alembic.
revision: str = "1daae0df1866"
down_revision: Union[str, None] = "b9274c27a25a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("data", sa.Column("pipeline_status", MutableDict.as_mutable(sa.JSON)))


def downgrade() -> None:
    op.drop_column("data", "pipeline_status")
