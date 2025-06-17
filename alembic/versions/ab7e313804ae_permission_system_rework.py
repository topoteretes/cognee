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


def _ensure_permission(conn, permission_name) -> str:
    """
    Return the permission.id for the given name, creating the row if needed.
    """
    from cognee.modules.users.models import Permission

    row = conn.execute(sa.select(Permission).filter(Permission.name == permission_name)).fetchone()

    if row is None:
        permission_id = uuid4()
        op.bulk_insert(
            Permission.__table__,
            [
                {
                    "id": permission_id,
                    "name": permission_name,
                    "created_at": _now(),
                    "updated_at": _now(),
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
        "updated_at": _now(),
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


def upgrade() -> None:
    conn = op.get_bind()

    # Recreate ACLs table with default permissions set to datasets instead of documents
    op.drop_table("acls")

    acls_table = op.create_table(
        "acls",
        sa.Column("id", UUID, primary_key=True, nullable=False, default=uuid4),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("principal_id", UUID, sa.ForeignKey("principals.id"), nullable=True),
        sa.Column("permission_id", UUID, sa.ForeignKey("permissions.id"), nullable=True),
        sa.Column(
            "dataset_id",
            UUID,
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    from cognee.modules.data.models import Dataset

    datasets = conn.execute(sa.select(Dataset)).fetchall()

    if not datasets:
        return

    acl_list = []

    for dataset in datasets:
        acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "read"))
        acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "write"))
        acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "share"))
        acl_list.append(_create_dataset_permission(conn, dataset.owner_id, dataset.id, "delete"))

    if acl_list:
        op.bulk_insert(acls_table, acl_list)


def downgrade() -> None:
    conn = op.get_bind()

    op.drop_table("acls")

    acls_table = op.create_table(
        "acls",
        sa.Column("id", UUID, primary_key=True, nullable=False, default=uuid4),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("principal_id", UUID, sa.ForeignKey("principals.id"), nullable=True),
        sa.Column("permission_id", UUID, sa.ForeignKey("permissions.id"), nullable=True),
        sa.Column(
            "data_id",
            UUID,
            sa.ForeignKey("data.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    from cognee.modules.data.models import Data

    data = conn.execute(sa.select(Data)).fetchall()

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
