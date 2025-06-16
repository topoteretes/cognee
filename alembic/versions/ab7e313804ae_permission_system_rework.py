"""permission_system_rework

Revision ID: ab7e313804ae
Revises: 1d0bb7fede17
Create Date: 2025-06-16 15:20:43.118246

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime, timezone
from uuid import uuid4

# revision identifiers, used by Alembic.
revision: str = "ab7e313804ae"
down_revision: Union[str, None] = "1d0bb7fede17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_dataset_permission(conn, user_id, dataset_id, permission_name) -> dict:
    from cognee.modules.users.models import Permission

    permission = conn.execute(
        sa.select(Permission).filter(Permission.name == permission_name)
    ).fetchone()

    if permission is None:
        permission = Permission(name=permission_name)

    return {
        "id": uuid4(),
        "create_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "principal_id": user_id,
        "dataset_id": dataset_id,
        "permission_id": permission.id,
    }


def _uuid_type():
    """Return a UUID-compatible column type for the current dialect."""
    if op.get_bind().dialect.name == "postgresql":
        return postgresql.UUID(as_uuid=True)
    # SQLite (and others): fall back to CHAR(36) – application inserts uuid4()
    return sa.CHAR(36)


def upgrade() -> None:
    conn = op.get_bind()

    # Recreate ACLs table with default permissions set to datasets instead of documents
    op.drop_table("acls")

    uuid_type = _uuid_type()
    op.create_table(
        "acls",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("principal_id", uuid_type, sa.ForeignKey("principals.id"), nullable=True),
        sa.Column("permission_id", uuid_type, sa.ForeignKey("permissions.id"), nullable=True),
        sa.Column(
            "dataset_id",
            uuid_type,
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
        from cognee.modules.users.models import ACL

        op.bulk_insert(ACL.__table__, acl_list)


def downgrade() -> None:
    # op.drop_table('acls')
    # op.create_table('acls')
    pass
