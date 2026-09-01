"""Initial migration

Revision ID: 8057ae7329c2
Revises:
Create Date: 2024-10-02 12:55:20.989372

Historically this revision was EMPTY: the chain was a delta log and the base
tables only ever came from ``Base.metadata.create_all()`` + ``stamp head`` —
an unverified assertion that the models and the chain describe the same
schema (the assertion whose silent break caused the Aug-2026 cloud mis-stamp
incident). It now carries the FROZEN base schema, making the chain a
self-sufficient creation history: ``alembic upgrade head`` on an empty
database builds the real schema, so the fresh-database bootstrap runs the
chain instead of stamping, and ``alembic_version`` only ever records
revisions that actually executed.

The DDL below is a frozen snapshot rendered by Alembic's autogenerate engine
from the certified model surface at chain head d1e2f3a4b5c6 (2026-09-01; a
real chain-migrated database was verified to match the models exactly).
FROZEN means frozen: never regenerate this body from live models — a future
model change ships as a NEW migration at the head of the chain, exactly as
before. Every table (with its indexes) is created only when absent, so the
body is idempotent, safe on partially-existing legacy databases, and inert
on any database where this revision is already recorded (the body never
runs there — Alembic considers it applied).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8057ae7329c2"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _existing = set(sa.inspect(op.get_bind()).get_table_names())

    if "data" not in _existing:
        op.create_table(
            "data",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("label", sa.String(), nullable=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("extension", sa.String(), nullable=True),
            sa.Column("mime_type", sa.String(), nullable=True),
            sa.Column("original_extension", sa.String(), nullable=True),
            sa.Column("original_mime_type", sa.String(), nullable=True),
            sa.Column("loader_engine", sa.String(), nullable=True),
            sa.Column("raw_data_location", sa.String(), nullable=True),
            sa.Column("original_data_location", sa.String(), nullable=True),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("legacy_id", sa.UUID(), nullable=True),
            sa.Column("content_hash", sa.String(), nullable=True),
            sa.Column("raw_content_hash", sa.String(), nullable=True),
            sa.Column("external_metadata", sa.JSON(), nullable=True),
            sa.Column("system_metadata", sa.JSON(), nullable=True),
            sa.Column("node_set", sa.JSON(), nullable=True),
            sa.Column("pipeline_status", sa.JSON(), nullable=True),
            sa.Column("token_count", sa.Integer(), nullable=True),
            sa.Column("data_size", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
            sa.Column("importance_weight", sa.Float(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "data_dataset_content_lookup",
            "data",
            ["dataset_id", "owner_id", "content_hash"],
            unique=False,
        )
        op.create_index(op.f("ix_data_dataset_id"), "data", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_data_legacy_id"), "data", ["legacy_id"], unique=False)
        op.create_index(op.f("ix_data_owner_id"), "data", ["owner_id"], unique=False)
        op.create_index(op.f("ix_data_tenant_id"), "data", ["tenant_id"], unique=False)

    if "datasets" not in _existing:
        op.create_table(
            "datasets",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("name", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_datasets_owner_id"), "datasets", ["owner_id"], unique=False)
        op.create_index(op.f("ix_datasets_tenant_id"), "datasets", ["tenant_id"], unique=False)

    if "edges" not in _existing:
        op.create_table(
            "edges",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("slug", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("data_id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("pipeline_run_id", sa.UUID(), nullable=True),
            sa.Column("source_node_id", sa.UUID(), nullable=False),
            sa.Column("destination_node_id", sa.UUID(), nullable=False),
            sa.Column("relationship_name", sa.Text(), nullable=False),
            sa.Column("label", sa.Text(), nullable=True),
            sa.Column("attributes", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_edges_data_id"), "edges", ["data_id"], unique=False)
        op.create_index(op.f("ix_edges_dataset_id"), "edges", ["dataset_id"], unique=False)
        op.create_index(
            op.f("ix_edges_pipeline_run_id"), "edges", ["pipeline_run_id"], unique=False
        )

    if "global_database_version" not in _existing:
        op.create_table(
            "global_database_version",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("cognee_version", sa.String(), nullable=True),
            sa.Column("global_migration_revision", sa.String(), nullable=True),
            sa.Column("global_migration_last_error", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if "graph_metrics" not in _existing:
        op.create_table(
            "graph_metrics",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column(
                "has_full_metrics", sa.Boolean(), server_default=sa.false(), nullable=False
            ),
            sa.Column("num_tokens", sa.Integer(), nullable=True),
            sa.Column("num_nodes", sa.Integer(), nullable=True),
            sa.Column("num_edges", sa.Integer(), nullable=True),
            sa.Column("mean_degree", sa.Float(), nullable=True),
            sa.Column("edge_density", sa.Float(), nullable=True),
            sa.Column("num_connected_components", sa.Integer(), nullable=True),
            sa.Column("sizes_of_connected_components", sa.JSON(), nullable=True),
            sa.Column("num_selfloops", sa.Integer(), nullable=True),
            sa.Column("diameter", sa.Integer(), nullable=True),
            sa.Column("avg_shortest_path_length", sa.Float(), nullable=True),
            sa.Column("avg_clustering", sa.Float(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=True,
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if "graph_relationship_ledger" not in _existing:
        op.create_table(
            "graph_relationship_ledger",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("source_node_id", sa.UUID(), nullable=False),
            sa.Column("destination_node_id", sa.UUID(), nullable=False),
            sa.Column("creator_function", sa.String(), nullable=False),
            sa.Column("node_label", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "idx_graph_relationship_id", "graph_relationship_ledger", ["id"], unique=False
        )
        op.create_index(
            "idx_graph_relationship_ledger_destination_node_id",
            "graph_relationship_ledger",
            ["destination_node_id"],
            unique=False,
        )
        op.create_index(
            "idx_graph_relationship_ledger_source_node_id",
            "graph_relationship_ledger",
            ["source_node_id"],
            unique=False,
        )

    if "integration_credentials" not in _existing:
        op.create_table(
            "integration_credentials",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("workspace_id", sa.UUID(), nullable=True),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("provider_account_id", sa.String(), nullable=True),
            sa.Column("account_label", sa.String(), nullable=True),
            sa.Column("auth_type", sa.String(), nullable=False),
            sa.Column("scopes", sa.String(), nullable=True),
            sa.Column("provider_metadata", sa.JSON(), nullable=True),
            sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
            sa.Column("nonce", sa.LargeBinary(), nullable=False),
            sa.Column("encryption_version", sa.SmallInteger(), nullable=False),
            sa.Column("key_id", sa.String(), nullable=False),
            sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("sync_status", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_integration_credentials_provider_account",
            "integration_credentials",
            ["provider", "provider_account_id"],
            unique=True,
        )
        op.create_index(
            op.f("ix_integration_credentials_user_id"),
            "integration_credentials",
            ["user_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_integration_credentials_workspace_id"),
            "integration_credentials",
            ["workspace_id"],
            unique=False,
        )

    if "nodes" not in _existing:
        op.create_table(
            "nodes",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("slug", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("data_id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("pipeline_run_id", sa.UUID(), nullable=True),
            sa.Column("label", sa.Text(), nullable=True),
            sa.Column("type", sa.Text(), nullable=False),
            sa.Column("indexed_fields", sa.JSON(), nullable=False),
            sa.Column("attributes", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("index_node_dataset_data", "nodes", ["dataset_id", "data_id"], unique=False)
        op.create_index("index_node_dataset_slug", "nodes", ["dataset_id", "slug"], unique=False)
        op.create_index(op.f("ix_nodes_dataset_id"), "nodes", ["dataset_id"], unique=False)
        op.create_index(
            op.f("ix_nodes_pipeline_run_id"), "nodes", ["pipeline_run_id"], unique=False
        )

    if "permissions" not in _existing:
        op.create_table(
            "permissions",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_permissions_id"), "permissions", ["id"], unique=False)
        op.create_index(op.f("ix_permissions_name"), "permissions", ["name"], unique=True)

    if "pipeline_runs" not in _existing:
        op.create_table(
            "pipeline_runs",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "status",
                sa.Enum(
                    "DATASET_PROCESSING_INITIATED",
                    "DATASET_PROCESSING_STARTED",
                    "DATASET_PROCESSING_COMPLETED",
                    "DATASET_PROCESSING_ERRORED",
                    name="pipelinerunstatus",
                ),
                nullable=True,
            ),
            sa.Column("pipeline_run_id", sa.UUID(), nullable=True),
            sa.Column("pipeline_name", sa.String(), nullable=True),
            sa.Column("pipeline_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("run_info", sa.JSON(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
            sa.Column("operation_name", sa.String(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("outcome", sa.String(), nullable=True),
            sa.Column("error_class", sa.String(), nullable=True),
            sa.Column("tokens_in", sa.Integer(), nullable=True),
            sa.Column("tokens_out", sa.Integer(), nullable=True),
            sa.Column("origin", sa.String(), nullable=True),
            sa.Column("session_id", sa.String(), nullable=True),
            sa.Column("parent_operation_id", sa.UUID(), nullable=True),
            sa.Column("background", sa.Boolean(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_pipeline_runs_created_at_id", "pipeline_runs", ["created_at", "id"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_dataset_id"), "pipeline_runs", ["dataset_id"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_operation_name"),
            "pipeline_runs",
            ["operation_name"],
            unique=False,
        )
        op.create_index(
            op.f("ix_pipeline_runs_outcome"), "pipeline_runs", ["outcome"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_parent_operation_id"),
            "pipeline_runs",
            ["parent_operation_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_pipeline_runs_pipeline_id"), "pipeline_runs", ["pipeline_id"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_pipeline_run_id"),
            "pipeline_runs",
            ["pipeline_run_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_pipeline_runs_session_id"), "pipeline_runs", ["session_id"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_user_id"), "pipeline_runs", ["user_id"], unique=False
        )

    if "principals" not in _existing:
        op.create_table(
            "principals",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("type", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_principals_id"), "principals", ["id"], unique=False)

    if "provenance_entries" not in _existing:
        op.create_table(
            "provenance_entries",
            sa.Column("entity_id", sa.String(length=1024), nullable=False),
            sa.Column("entity_type", sa.String(length=64), nullable=False),
            sa.Column("activity_id", sa.String(length=256), nullable=False),
            sa.Column("agent_id", sa.String(length=256), nullable=False),
            sa.Column("agent_type", sa.String(length=32), nullable=False),
            sa.Column("is_automated", sa.Boolean(), nullable=False),
            sa.Column("role", sa.String(length=128), nullable=True),
            sa.Column("source_document", sa.Text(), nullable=False),
            sa.Column("source_location", sa.Text(), nullable=True),
            sa.Column("source_quote", sa.Text(), nullable=True),
            sa.Column("source_ref_key", sa.String(length=256), nullable=True),
            sa.Column("timestamp", sa.String(length=64), nullable=False),
            sa.Column("first_seen", sa.String(length=64), nullable=True),
            sa.Column("last_updated", sa.String(length=64), nullable=True),
            sa.Column("activity_started_at_time", sa.String(length=64), nullable=True),
            sa.Column("activity_ended_at_time", sa.String(length=64), nullable=True),
            sa.Column("valid_from", sa.String(length=64), nullable=True),
            sa.Column("valid_until", sa.String(length=64), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("credibility", sa.Float(), nullable=True),
            sa.Column("checksum", sa.String(length=64), nullable=True),
            sa.Column("sequence_id", sa.Integer(), nullable=True),
            sa.Column("previous_checksum", sa.String(length=64), nullable=True),
            sa.Column("parent_entity_id", sa.String(length=1024), nullable=True),
            sa.Column("used_entities", sa.JSON(), nullable=False),
            sa.Column("previous_version_id", sa.String(length=1024), nullable=True),
            sa.Column("derived_from_id", sa.String(length=1024), nullable=True),
            sa.Column("acted_on_behalf_of", sa.String(length=256), nullable=True),
            sa.Column("informed_by_activities", sa.JSON(), nullable=False),
            sa.Column("revision_type", sa.String(length=64), nullable=True),
            sa.Column("supersedes", sa.String(length=1024), nullable=True),
            sa.Column("bundle_id", sa.String(length=256), nullable=True),
            sa.Column("invalidated", sa.Boolean(), nullable=False),
            sa.Column("invalidated_at_time", sa.String(length=64), nullable=True),
            sa.Column("invalidated_by", sa.String(length=256), nullable=True),
            sa.Column("invalidation_reason", sa.Text(), nullable=True),
            sa.Column("start_index", sa.Integer(), nullable=True),
            sa.Column("end_index", sa.Integer(), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("version", sa.String(length=16), nullable=False),
            sa.PrimaryKeyConstraint("entity_id"),
        )
        op.create_index("idx_provenance_bundle", "provenance_entries", ["bundle_id"], unique=False)
        op.create_index(
            "idx_provenance_derived_from", "provenance_entries", ["derived_from_id"], unique=False
        )
        op.create_index(
            "idx_provenance_entity_type", "provenance_entries", ["entity_type"], unique=False
        )
        op.create_index(
            "idx_provenance_invalidated", "provenance_entries", ["invalidated"], unique=False
        )
        op.create_index(
            "idx_provenance_parent", "provenance_entries", ["parent_entity_id"], unique=False
        )
        op.create_index(
            "idx_provenance_prev_version",
            "provenance_entries",
            ["previous_version_id"],
            unique=False,
        )
        op.create_index(
            "idx_provenance_source_ref_key", "provenance_entries", ["source_ref_key"], unique=False
        )
        op.create_index(
            "ux_provenance_sequence_id",
            "provenance_entries",
            ["sequence_id"],
            unique=True,
            postgresql_where=sa.text("sequence_id IS NOT NULL"),
            sqlite_where=sa.text("sequence_id IS NOT NULL"),
        )

    if "queries" not in _existing:
        op.create_table(
            "queries",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("text", sa.String(), nullable=True),
            sa.Column("query_type", sa.String(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_queries_created_at"), "queries", ["created_at"], unique=False)
        op.create_index(op.f("ix_queries_dataset_id"), "queries", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_queries_user_id"), "queries", ["user_id"], unique=False)

    if "results" not in _existing:
        op.create_table(
            "results",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("value", sa.Text(), nullable=True),
            sa.Column("query_id", sa.UUID(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_results_created_at"), "results", ["created_at"], unique=False)
        op.create_index(op.f("ix_results_dataset_id"), "results", ["dataset_id"], unique=False)
        op.create_index(op.f("ix_results_user_id"), "results", ["user_id"], unique=False)

    if "session_model_usage" not in _existing:
        op.create_table(
            "session_model_usage",
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("model", sa.Text(), nullable=False),
            sa.Column("tokens_in", sa.Integer(), nullable=False),
            sa.Column("tokens_out", sa.Integer(), nullable=False),
            sa.Column("cost_usd", sa.Float(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("session_id", "user_id", "model"),
        )
        op.create_index(
            op.f("ix_session_model_usage_user_id"), "session_model_usage", ["user_id"], unique=False
        )

    if "session_records" not in _existing:
        op.create_table(
            "session_records",
            sa.Column("session_id", sa.String(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("tokens_in", sa.Integer(), nullable=False),
            sa.Column("tokens_out", sa.Integer(), nullable=False),
            sa.Column("cost_usd", sa.Float(), nullable=False),
            sa.Column("error_count", sa.Integer(), nullable=False),
            sa.Column("last_model", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("session_id", "user_id"),
        )
        op.create_index(
            op.f("ix_session_records_dataset_id"), "session_records", ["dataset_id"], unique=False
        )
        op.create_index(
            op.f("ix_session_records_last_activity_at"),
            "session_records",
            ["last_activity_at"],
            unique=False,
        )
        op.create_index(
            op.f("ix_session_records_status"), "session_records", ["status"], unique=False
        )
        op.create_index(
            op.f("ix_session_records_user_id"), "session_records", ["user_id"], unique=False
        )

    if "sync_operations" not in _existing:
        op.create_table(
            "sync_operations",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("run_id", sa.Text(), nullable=True),
            sa.Column(
                "status",
                sa.Enum(
                    "STARTED", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED", name="syncstatus"
                ),
                nullable=True,
            ),
            sa.Column("progress_percentage", sa.Integer(), nullable=True),
            sa.Column("dataset_ids", sa.JSON(), nullable=True),
            sa.Column("dataset_names", sa.JSON(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("total_records_to_sync", sa.Integer(), nullable=True),
            sa.Column("total_records_to_download", sa.Integer(), nullable=True),
            sa.Column("total_records_to_upload", sa.Integer(), nullable=True),
            sa.Column("records_downloaded", sa.Integer(), nullable=True),
            sa.Column("records_uploaded", sa.Integer(), nullable=True),
            sa.Column("bytes_downloaded", sa.Integer(), nullable=True),
            sa.Column("bytes_uploaded", sa.Integer(), nullable=True),
            sa.Column("dataset_sync_hashes", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("retry_count", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_sync_operations_run_id"), "sync_operations", ["run_id"], unique=True
        )
        op.create_index(
            op.f("ix_sync_operations_user_id"), "sync_operations", ["user_id"], unique=False
        )

    if "tool_connections" not in _existing:
        op.create_table(
            "tool_connections",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("provider", sa.String(), nullable=False),
            sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
            sa.Column("nonce", sa.LargeBinary(), nullable=False),
            sa.Column("encryption_version", sa.SmallInteger(), nullable=False),
            sa.Column("key_id", sa.String(), nullable=False),
            sa.Column("options", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "name", name="uq_tool_connections_user_name"),
        )
        op.create_index(
            op.f("ix_tool_connections_user_id"), "tool_connections", ["user_id"], unique=False
        )

    if "tool_write_proposals" not in _existing:
        op.create_table(
            "tool_write_proposals",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("connection_name", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("instruction", sa.Text(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=True),
            sa.Column("sql", sa.Text(), nullable=False),
            sa.Column("target_table", sa.String(), nullable=True),
            sa.Column("estimated_rows", sa.Integer(), nullable=True),
            sa.Column("applied_rows", sa.Integer(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_tool_write_proposals_status"), "tool_write_proposals", ["status"], unique=False
        )
        op.create_index(
            op.f("ix_tool_write_proposals_user_id"),
            "tool_write_proposals",
            ["user_id"],
            unique=False,
        )

    if "acls" not in _existing:
        op.create_table(
            "acls",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("principal_id", sa.UUID(), nullable=True),
            sa.Column("permission_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["permission_id"],
                ["permissions.id"],
            ),
            sa.ForeignKeyConstraint(
                ["principal_id"],
                ["principals.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    if "dataset_configurations" not in _existing:
        op.create_table(
            "dataset_configurations",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("graph_schema", sa.JSON(), nullable=True),
            sa.Column("custom_prompt", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("dataset_id"),
        )

    if "dataset_database" not in _existing:
        op.create_table(
            "dataset_database",
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("vector_database_name", sa.String(), nullable=False),
            sa.Column("graph_database_name", sa.String(), nullable=False),
            sa.Column("vector_database_provider", sa.String(), nullable=False),
            sa.Column("graph_database_provider", sa.String(), nullable=False),
            sa.Column("graph_dataset_database_handler", sa.String(), nullable=False),
            sa.Column("vector_dataset_database_handler", sa.String(), nullable=False),
            sa.Column("vector_database_url", sa.String(), nullable=True),
            sa.Column("graph_database_url", sa.String(), nullable=True),
            sa.Column("vector_database_key", sa.String(), nullable=True),
            sa.Column("graph_database_key", sa.String(), nullable=True),
            sa.Column("cognee_version", sa.String(), nullable=True),
            sa.Column("migration_revision", sa.String(), nullable=True),
            sa.Column("migration_last_error", sa.String(), nullable=True),
            sa.Column(
                "graph_database_connection_info",
                sa.JSON(),
                server_default=sa.text("'{}'"),
                nullable=False,
            ),
            sa.Column(
                "vector_database_connection_info",
                sa.JSON(),
                server_default=sa.text("'{}'"),
                nullable=False,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["owner_id"], ["principals.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("dataset_id"),
        )
        op.create_index(
            op.f("ix_dataset_database_dataset_id"), "dataset_database", ["dataset_id"], unique=False
        )
        op.create_index(
            op.f("ix_dataset_database_owner_id"), "dataset_database", ["owner_id"], unique=False
        )

    if "principal_configuration" not in _existing:
        op.create_table(
            "principal_configuration",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("configuration", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["owner_id"], ["principals.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_principal_configuration_owner_id"),
            "principal_configuration",
            ["owner_id"],
            unique=False,
        )

    if "tenants" not in _existing:
        op.create_table(
            "tenants",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.ForeignKeyConstraint(
                ["id"],
                ["principals.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_tenants_name"), "tenants", ["name"], unique=False)
        op.create_index(op.f("ix_tenants_owner_id"), "tenants", ["owner_id"], unique=False)

    if "user_api_key" not in _existing:
        op.create_table(
            "user_api_key",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("api_key", sa.String(), nullable=False),
            sa.Column("label", sa.String(), nullable=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["principals.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_user_api_key_user_id"), "user_api_key", ["user_id"], unique=False)

    if "roles" not in _existing:
        op.create_table(
            "roles",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(["id"], ["principals.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["tenant_id"],
                ["tenants.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_id_name"),
        )
        op.create_index(op.f("ix_roles_name"), "roles", ["name"], unique=False)

    if "tenant_default_permissions" not in _existing:
        op.create_table(
            "tenant_default_permissions",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.Column("permission_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("tenant_id", "permission_id"),
        )

    if "users" not in _existing:
        op.create_table(
            "users",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
            sa.Column("parent_user_id", sa.UUID(), nullable=True),
            sa.Column("email", sa.String(length=320), nullable=False),
            sa.Column("hashed_password", sa.String(length=1024), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("is_superuser", sa.Boolean(), nullable=False),
            sa.Column("is_verified", sa.Boolean(), nullable=False),
            sa.ForeignKeyConstraint(["id"], ["principals.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["parent_user_id"], ["users.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(
                ["tenant_id"],
                ["tenants.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    if "role_default_permissions" not in _existing:
        op.create_table(
            "role_default_permissions",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("role_id", sa.UUID(), nullable=False),
            sa.Column("permission_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("role_id", "permission_id"),
        )

    if "user_default_permissions" not in _existing:
        op.create_table(
            "user_default_permissions",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("permission_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("user_id", "permission_id"),
        )

    if "user_roles" not in _existing:
        op.create_table(
            "user_roles",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("role_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(
                ["role_id"],
                ["roles.id"],
            ),
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
            ),
            sa.PrimaryKeyConstraint("user_id", "role_id"),
        )

    if "user_tenants" not in _existing:
        op.create_table(
            "user_tenants",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.ForeignKeyConstraint(
                ["tenant_id"],
                ["tenants.id"],
            ),
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
            ),
            sa.PrimaryKeyConstraint("user_id", "tenant_id"),
        )


def downgrade() -> None:
    # Downgrading to base has never dropped the base schema (this revision was
    # empty for its whole life); keep that contract.
    pass
