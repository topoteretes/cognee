"""Add pipeline run status

Revision ID: 1d0bb7fede17
Revises: 482cd6517ce4
Create Date: 2025-05-19 10:58:15.993314
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "1d0bb7fede17"
down_revision: Union[str, None] = "482cd6517ce4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "482cd6517ce4"


def upgrade() -> None:
    # The dialect of the database being migrated (op.get_bind()), never of the
    # globally configured engine: an adapter's create_database() runs this chain
    # on ITS database, which may not be the configured one.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "ALTER TYPE pipelinerunstatus ADD VALUE IF NOT EXISTS 'DATASET_PROCESSING_INITIATED'"
        )


def downgrade() -> None:
    pass
