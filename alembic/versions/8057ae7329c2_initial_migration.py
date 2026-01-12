"""Initial migration

Revision ID: 8057ae7329c2
Revises:
Create Date: 2024-10-02 12:55:20.989372

"""

from typing import Sequence, Union
from sqlalchemy.util import await_only
from cognee.infrastructure.databases.relational import get_relational_engine

# revision identifiers, used by Alembic.
revision: str = "8057ae7329c2"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
