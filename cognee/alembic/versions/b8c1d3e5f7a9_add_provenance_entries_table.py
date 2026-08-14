"""Add provenance_entries table

Revision ID: b8c1d3e5f7a9
Revises: f2b4c6d8e0a1
Create Date: 2026-08-14 00:00:00.000000

Append-only, tamper-evident audit ledger for the opt-in provenance-tracking
cognify task (env PROVENANCE_TRACKING, default off). One relational table;
joins to graph provenance via source_ref_key. See cognee/modules/provenance/.

Note: the design doc named this revision a1b2c3d4e5f6, but that id was already
taken by add_label_column_to_data, so a fresh id is used.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b8c1d3e5f7a9"
down_revision: Union[str, None] = "f2b4c6d8e0a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The table may already exist: Base.metadata.create_all() on the startup
    # path creates it once cognee.modules.provenance.models has been imported.
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "provenance_entries" in inspector.get_table_names():
        return

    op.create_table(
        "provenance_entries",
        # identity / PROV-O core
        sa.Column("entity_id", sa.String(length=1024), primary_key=True),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("activity_id", sa.String(length=256), nullable=False),
        sa.Column("agent_id", sa.String(length=256), nullable=False),
        sa.Column("agent_type", sa.String(length=32), nullable=False),
        sa.Column("is_automated", sa.Boolean(), nullable=False),
        sa.Column("role", sa.String(length=128), nullable=True),
        # audit-grade source
        sa.Column("source_document", sa.Text(), nullable=False),
        sa.Column("source_location", sa.Text(), nullable=True),
        sa.Column("source_quote", sa.Text(), nullable=True),
        sa.Column("source_ref_key", sa.String(length=256), nullable=True),
        # temporal (ISO-8601 strings — hashed material must be dialect-stable)
        sa.Column("timestamp", sa.String(length=64), nullable=False),
        sa.Column("first_seen", sa.String(length=64), nullable=True),
        sa.Column("last_updated", sa.String(length=64), nullable=True),
        sa.Column("activity_started_at_time", sa.String(length=64), nullable=True),
        sa.Column("activity_ended_at_time", sa.String(length=64), nullable=True),
        sa.Column("valid_from", sa.String(length=64), nullable=True),
        sa.Column("valid_until", sa.String(length=64), nullable=True),
        # quality / hash chain
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("credibility", sa.Float(), nullable=True),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("sequence_id", sa.Integer(), nullable=True),
        sa.Column("previous_checksum", sa.String(length=64), nullable=True),
        # lineage
        sa.Column("parent_entity_id", sa.String(length=1024), nullable=True),
        sa.Column("used_entities", sa.JSON(), nullable=False),
        sa.Column("previous_version_id", sa.String(length=1024), nullable=True),
        sa.Column("derived_from_id", sa.String(length=1024), nullable=True),
        # governance
        sa.Column("acted_on_behalf_of", sa.String(length=256), nullable=True),
        sa.Column("informed_by_activities", sa.JSON(), nullable=False),
        sa.Column("revision_type", sa.String(length=64), nullable=True),
        sa.Column("supersedes", sa.String(length=1024), nullable=True),
        sa.Column("bundle_id", sa.String(length=256), nullable=True),
        # invalidation tombstone
        sa.Column("invalidated", sa.Boolean(), nullable=False),
        sa.Column("invalidated_at_time", sa.String(length=64), nullable=True),
        sa.Column("invalidated_by", sa.String(length=256), nullable=True),
        sa.Column("invalidation_reason", sa.Text(), nullable=True),
        # chunk extras + free-form
        sa.Column("start_index", sa.Integer(), nullable=True),
        sa.Column("end_index", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("version", sa.String(length=16), nullable=False),
    )

    op.create_index("idx_provenance_entity_type", "provenance_entries", ["entity_type"])
    op.create_index("idx_provenance_parent", "provenance_entries", ["parent_entity_id"])
    op.create_index("idx_provenance_prev_version", "provenance_entries", ["previous_version_id"])
    op.create_index("idx_provenance_derived_from", "provenance_entries", ["derived_from_id"])
    op.create_index("idx_provenance_bundle", "provenance_entries", ["bundle_id"])
    op.create_index("idx_provenance_source_ref_key", "provenance_entries", ["source_ref_key"])
    op.create_index("idx_provenance_invalidated", "provenance_entries", ["invalidated"])
    # THE chain guard: two writers cannot both claim the same slot.
    op.create_index(
        "ux_provenance_sequence_id",
        "provenance_entries",
        ["sequence_id"],
        unique=True,
        postgresql_where=sa.text("sequence_id IS NOT NULL"),
        sqlite_where=sa.text("sequence_id IS NOT NULL"),
    )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "provenance_entries" in inspector.get_table_names():
        op.drop_table("provenance_entries")
