"""Add default user

Revision ID: 482cd6517ce4
Revises: 8057ae7329c2
Create Date: 2024-10-16 22:17:18.634638

"""

from typing import Sequence, Union

from sqlalchemy.util import await_only

from cognee.modules.users.methods import create_default_user, delete_user

from fastapi_users.exceptions import UserAlreadyExists


# revision identifiers, used by Alembic.
revision: str = "482cd6517ce4"
down_revision: Union[str, None] = "8057ae7329c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "8057ae7329c2"


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
