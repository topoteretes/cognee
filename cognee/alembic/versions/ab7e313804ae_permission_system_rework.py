"""permission_system_rework

Revision ID: ab7e313804ae
Revises: 1d0bb7fede17
Create Date: 2025-06-16 15:20:43.118246

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy import UUID
from datetime import datetime, timezone
from uuid import uuid4

# revision identifiers, used by Alembic.
revision: str = "ab7e313804ae"
down_revision: Union[str, None] = "1d0bb7fede17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _now():
    return datetime.now(timezone.utc)


def _define_dataset_table() -> sa.Table:
    # Note: We can't use any Cognee model info to gather data (as it can change) in database so we must use our own table
    #       definition or load what is in the database
    table = sa.Table(
        "datasets",
        sa.MetaData(),
        sa.Column("id", UUID, primary_key=True, default=uuid4),
        sa.Column("name", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
        sa.Column("owner_id", UUID, sa.ForeignKey("principals.id"), index=True),
    )

    return table


def _define_data_table() -> sa.Table:
    # Note: We can't use any Cognee model info to gather data (as it can change) in database so we must use our own table
    #       definition or load what is in the database
    table = sa.Table(
        "data",
        sa.MetaData(),
        sa.Column("id", UUID, primary_key=True, default=uuid4),
        sa.Column("name", sa.String),
        sa.Column("extension", sa.String),
        sa.Column("mime_type", sa.String),
        sa.Column("raw_data_location", sa.String),
        sa.Column("owner_id", UUID, index=True),
        sa.Column("content_hash", sa.String),
        sa.Column("external_metadata", sa.JSON),
        sa.Column("node_set", sa.JSON, nullable=True),  # list of strings
        sa.Column("token_count", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
    )

    return table


def _ensure_permission(conn, permission_name) -> str:
    """
    Return the permission.id for the given name, creating the row if needed.
    """
    permissions_table = sa.Table(
        "permissions",
        sa.MetaData(),
        sa.Column("id", UUID, primary_key=True, index=True, default=uuid4),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            onupdate=lambda: datetime.now(timezone.utc),
        ),
        sa.Column("name", sa.String, unique=True, nullable=False, index=True),
    )
    row = conn.execute(
        sa.select(permissions_table).filter(permissions_table.c.name == permission_name)
    ).fetchone()

    if row is None:
        permission_id = uuid4()

        op.bulk_insert(
            permissions_table,
            [
                {
                    "id": permission_id,
                    "name": permission_name,
                    "created_at": _now(),
                }
            ],
        )
        return permission_id

    return row.id


def _build_acl_row(*, user_id, target_id, permission_id, target_col) -> dict:
    """Create a dict with the correct column names for the ACL row."""
    return {
        "id": uuid4(),
        "created_at": _now(),
        "principal_id": user_id,
        target_col: target_id,
        "permission_id": permission_id,
    }


def _create_dataset_permission(conn, user_id, dataset_id, permission_name):
    perm_id = _ensure_permission(conn, permission_name)
    return _build_acl_row(
        user_id=user_id, target_id=dataset_id, permission_id=perm_id, target_col="dataset_id"
    )


def _create_data_permission(conn, user_id, data_id, permission_name):
    perm_id = _ensure_permission(conn, permission_name)
    return _build_acl_row(
        user_id=user_id, target_id=data_id, permission_id=perm_id, target_col="data_id"
    )


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    dataset_id_column = _get_column(insp, "acls", "dataset_id")
    if not dataset_id_column:
        # Recreate ACLs table with default permissions set to datasets instead of documents
        op.drop_table("acls")

        acls_table = op.create_table(
            "acls",
            sa.Column("id", UUID, primary_key=True, default=uuid4),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                onupdate=lambda: datetime.now(timezone.utc),
            ),
            sa.Column("principal_id", UUID, sa.ForeignKey("principals.id")),
            sa.Column("permission_id", UUID, sa.ForeignKey("permissions.id")),
            sa.Column("dataset_id", UUID, sa.ForeignKey("datasets.id", ondelete="CASCADE")),
        )

        # Note: We can't use any Cognee model info to gather data (as it can change) in database so we must use our own table
        #       definition or load what is in the database
        dataset_table = _define_dataset_table()
        datasets = conn.execute(sa.select(dataset_table)).fetchall()

        if not datasets:
            return

        acl_list = []

        for dataset in datasets:
            acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "read"))
            acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "write"))
            acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "share"))
            acl_list.append(
                _create_dataset_permission(conn, dataset.owner_id, dataset.id, "delete")
            )

        if acl_list:
            op.bulk_insert(acls_table, acl_list)


def downgrade() -> None:
    conn = op.get_bind()

    op.drop_table("acls")

    acls_table = op.create_table(
        "acls",
        sa.Column("id", UUID, primary_key=True, nullable=False, default=uuid4),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)
        ),
        sa.Column("principal_id", UUID, sa.ForeignKey("principals.id")),
        sa.Column("permission_id", UUID, sa.ForeignKey("permissions.id")),
        sa.Column("data_id", UUID, sa.ForeignKey("data.id", ondelete="CASCADE")),
    )

    # Note: We can't use any Cognee model info to gather data (as it can change) in database so we must use our own table
    #       definition or load what is in the database
    data_table = _define_data_table()
    data = conn.execute(sa.select(data_table)).fetchall()

    if not data:
        return

    acl_list = []
    for single_data in data:
        acl_list.append(_create_data_permission(conn, single_data.owner_id, single_data.id, "read"))
        acl_list.append(
            _create_data_permission(conn, single_data.owner_id, single_data.id, "write")
        )

    if acl_list:
        op.bulk_insert(acls_table, acl_list)
