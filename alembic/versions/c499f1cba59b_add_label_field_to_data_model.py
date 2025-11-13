"""add label field to Data model

Revision ID: c499f1cba59b
Revises: 211ab850ef3d
Create Date: 2025-11-12 22:54:28.517081
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c499f1cba59b'
down_revision: Union[str, None] = '211ab850ef3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the label column to Data table"""
    op.add_column('data', sa.Column('label', sa.String(), nullable=True))


def downgrade() -> None:
    """Remove the label column from Data table"""
    op.drop_column('data', 'label')
