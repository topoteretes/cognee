"""add_integration_credentials_table

Revision ID: b2c4d6e8f0a1
Revises: aa753a730673
Create Date: 2026-07-20 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b2c4d6e8f0a1"
down_revision: Union[str, None] = "aa753a730673"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "integration_credentials" in insp.get_table_names():
        return

    op.create_table(
        "integration_credentials",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_account_id", sa.String(), nullable=True),
        sa.Column("account_label", sa.String(), nullable=True),
        sa.Column("auth_type", sa.String(), nullable=False, server_default="oauth2"),
        sa.Column("scopes", sa.String(), nullable=True),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        sa.Column("encryption_version", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("key_id", sa.String(), nullable=False, server_default="1"),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sync_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_integration_credentials_user_id", "integration_credentials", ["user_id"])
    op.create_index(
        "ix_integration_credentials_provider_account",
        "integration_credentials",
        ["provider", "provider_account_id"],
        unique=True,
    )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "integration_credentials" in insp.get_table_names():
        op.drop_table("integration_credentials")
