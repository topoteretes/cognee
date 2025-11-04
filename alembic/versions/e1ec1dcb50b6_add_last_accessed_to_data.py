"""add_last_accessed_to_data

Revision ID: e1ec1dcb50b6
Revises: 211ab850ef3d
Create Date: 2025-11-04 21:45:52.642322

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1ec1dcb50b6'
down_revision: Union[str, None] = '211ab850ef3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

  
def upgrade() -> None:  
    op.add_column('data',   
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=True)  
    )  
    # Optionally initialize with created_at values for existing records  
    op.execute("UPDATE data SET last_accessed = created_at")  
  
  
def downgrade() -> None:  
    op.drop_column('data', 'last_accessed')
