"""add_file_processing_status_tracking

Revision ID: 1d1826f911f8
Revises: ab7e313804ae
Create Date: 2025-07-05 16:13:49.621227

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from cognee.infrastructure.databases.relational.get_relational_engine import get_relational_engine


# revision identifiers, used by Alembic.
revision: str = '1d1826f911f8'
down_revision: Union[str, None] = 'ab7e313804ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    db_engine = get_relational_engine()
    
    if db_engine.engine.dialect.name == "postgresql":
        # Create enum type
        fileprocessingstatus_enum = postgresql.ENUM(
            'UNPROCESSED', 'PROCESSING', 'PROCESSED', 'ERROR',
            name='fileprocessingstatus'
        )
        fileprocessingstatus_enum.create(op.get_bind(), checkfirst=True)
            
        # Add column with default
        op.add_column('data', sa.Column('processing_status', fileprocessingstatus_enum, server_default='UNPROCESSED'))
            
        # Update existing records to PROCESSED status
        op.execute("UPDATE data SET processing_status = 'PROCESSED' WHERE processing_status IS NULL")
            
        # Add index
        op.create_index('idx_data_processing_status', 'data', ['processing_status'])


def downgrade() -> None:
    db_engine = get_relational_engine()
    
    if db_engine.engine.dialect.name == "postgresql":
        op.drop_index('idx_data_processing_status', table_name='data')
        op.drop_column('data', 'processing_status')
        op.execute("DROP TYPE IF EXISTS fileprocessingstatus")
