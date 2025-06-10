"""Add processing_status column to data table

Revision ID: e51f431ce327
Revises: 1d0bb7fede17
Create Date: 2025-06-10 23:00:51.295547

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e51f431ce327'
down_revision: Union[str, None] = '1d0bb7fede17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'data',
        sa.Column(
            'processing_status',
            sa.Enum('unprocessed', 'processing', 'processed', 'error', name='file_processing_status'),
            nullable=False,
            server_default='unprocessed',
        )
    )
    op.create_index(op.f('ix_data_processing_status'), 'data', ['processing_status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_data_processing_status'), table_name='data')
    op.drop_column('data', 'processing_status')
