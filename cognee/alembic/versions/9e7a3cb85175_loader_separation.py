"""loader_separation

Revision ID: 9e7a3cb85175
Revises: 1daae0df1866
Create Date: 2025-08-14 19:18:11.406907

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9e7a3cb85175"
down_revision: Union[str, None] = "1daae0df1866"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    # Define table with all necessary columns including primary key
    data = sa.table(
        "data",
        sa.Column("id", sa.UUID, primary_key=True),  # Critical for SQLite
        sa.Column("original_extension", sa.String()),
        sa.Column("original_mime_type", sa.String()),
        sa.Column("original_data_location", sa.String()),
        sa.Column("extension", sa.String()),
        sa.Column("mime_type", sa.String()),
        sa.Column("raw_data_location", sa.String()),
    )

    original_extension_column = _get_column(insp, "data", "original_extension")
    if not original_extension_column:
        op.add_column("data", sa.Column("original_extension", sa.String(), nullable=True))
        if op.get_context().dialect.name == "sqlite":
            # If column doesn't exist create new original_extension column and update from values of extension column
            with op.batch_alter_table("data") as batch_op:
                batch_op.execute(
                    data.update().values(
                        original_extension=data.c.extension,
                    )
                )
        else:
            conn = op.get_bind()
            conn.execute(data.update().values(original_extension=data.c.extension))

    original_mime_type = _get_column(insp, "data", "original_mime_type")
    if not original_mime_type:
        # If column doesn't exist create new original_mime_type column and update from values of mime_type column
        op.add_column("data", sa.Column("original_mime_type", sa.String(), nullable=True))
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table("data") as batch_op:
                batch_op.execute(
                    data.update().values(
                        original_mime_type=data.c.mime_type,
                    )
                )
        else:
            conn = op.get_bind()
            conn.execute(data.update().values(original_mime_type=data.c.mime_type))

    loader_engine = _get_column(insp, "data", "loader_engine")
    if not loader_engine:
        op.add_column("data", sa.Column("loader_engine", sa.String(), nullable=True))

    original_data_location = _get_column(insp, "data", "original_data_location")
    if not original_data_location:
        # If column doesn't exist create new original data column and update from values of raw_data_location column
        op.add_column("data", sa.Column("original_data_location", sa.String(), nullable=True))
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table("data") as batch_op:
                batch_op.execute(
                    data.update().values(
                        original_data_location=data.c.raw_data_location,
                    )
                )
        else:
            conn = op.get_bind()
            conn.execute(data.update().values(original_data_location=data.c.raw_data_location))

    raw_content_hash = _get_column(insp, "data", "raw_content_hash")
    if not raw_content_hash:
        op.add_column("data", sa.Column("raw_content_hash", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("data", "raw_content_hash")
    op.drop_column("data", "original_data_location")
    op.drop_column("data", "loader_engine")
    op.drop_column("data", "original_mime_type")
    op.drop_column("data", "original_extension")
