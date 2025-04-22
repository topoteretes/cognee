"""add default user

Revision ID: 482cd6517ce4
Revises: 8057ae7329c2
Create Date: 2023-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
import uuid

# revision identifiers, used by Alembic.
revision = '482cd6517ce4'
down_revision = '8057ae7329c2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create a default admin user
    users = table('users',
        column('id', sa.String),
        column('email', sa.String),
        column('hashed_password', sa.String),
        column('is_active', sa.Boolean),
        column('is_superuser', sa.Boolean),
        column('is_verified', sa.Boolean)
    )
    
    # Default hashed password for 'admin@example.com' (this is a placeholder)
    # In a real scenario, this would be properly hashed
    hashed_password = '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW'  # 'password'
    
    op.bulk_insert(users, [
        {
            'id': str(uuid.uuid4()),
            'email': 'admin@example.com',
            'hashed_password': hashed_password,
            'is_active': True,
            'is_superuser': True,
            'is_verified': True
        }
    ])


def downgrade() -> None:
    # Remove the default admin user
    op.execute("DELETE FROM users WHERE email = 'admin@example.com'")
