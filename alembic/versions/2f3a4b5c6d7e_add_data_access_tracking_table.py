"""add_data_access_tracking_table

Revision ID: 2f3a4b5c6d7e
Revises: 1daae0df1866
Create Date: 2025-10-11 12:46:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '2f3a4b5c6d7e'
down_revision: Union[str, None] = '1daae0df1866'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create data_access_tracking reference table for tracking Data model access.
    
    This supports issue #1335: cleanup_unused_data functionality at the Data level.
    By tracking access at the Data level, we can properly clean up related graph
    and vector database entries when Data entries are deleted.
    
    The reference table approach avoids frequent writes on the main Data table
    while maintaining efficient access tracking.
    """
    # Create data_access_tracking table
    op.create_table(
        'data_access_tracking',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('data_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('access_count', sa.Integer(), nullable=False, server_default=sa.text('1')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['data_id'], ['data.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('data_id', name='uq_data_access_tracking_data_id')
    )
    
    # Create index on data_id for efficient lookups
    op.create_index(
        'ix_data_access_tracking_data_id',
        'data_access_tracking',
        ['data_id']
    )
    
    # Create index on last_accessed for efficient cleanup queries
    op.create_index(
        'ix_data_access_tracking_last_accessed',
        'data_access_tracking',
        ['last_accessed']
    )


def downgrade() -> None:
    """
    Remove data_access_tracking table and its indexes.
    """
    op.drop_index('ix_data_access_tracking_last_accessed', table_name='data_access_tracking')
    op.drop_index('ix_data_access_tracking_data_id', table_name='data_access_tracking')
    op.drop_table('data_access_tracking')
