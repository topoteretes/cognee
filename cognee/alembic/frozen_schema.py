"""The FROZEN base schema of cognee's relational database.

This module is the single source of the frozen surface two revisions build
from: the initial revision ``8057ae7329c2`` creates every table here that does
not exist yet, and the reconcile revision ``1c22e6cb5aec`` converges an
existing database to it (missing tables, columns and indexes). It is the
certified model surface at chain head ``d1e2f3a4b5c6`` (32 tables), rendered
by Alembic's autogenerate and verified against a real chain-migrated database,
plus the one chain-provided index that surface lacks
(``ix_pipeline_runs_dataset_pipeline_created_at`` from ``d1e2f3a4b5c6``:
``PipelineRun.__table_args__`` is assigned twice, so the model never declared
it).

FROZEN MEANS FROZEN. Never regenerate this from live models: a live-model
surface silently absorbs any model change whose migration is missing — the
one failure the migration/model lockstep guard cannot detect, because chain
and models would then agree. Every future schema change ships as a NEW
revision at the head. ``tests/unit/test_frozen_schema_seal.py`` pins this
module's fingerprint; changing it is a deliberate re-certification.

Portable across cognee's two dialects: enum columns are ``postgresql.ENUM``
with ``create_type=False`` on Postgres (the types are created up front with
``checkfirst`` — they outlive ``DROP TABLE``) and ``sa.Enum`` elsewhere;
server defaults are SQLAlchemy constructs, never compiled literals.
"""

import hashlib
import json

import sqlalchemy as sa

ENUM_TYPES = {
    "pipelinerunstatus": (
        "DATASET_PROCESSING_INITIATED",
        "DATASET_PROCESSING_STARTED",
        "DATASET_PROCESSING_COMPLETED",
        "DATASET_PROCESSING_ERRORED",
    ),
    "syncstatus": ("STARTED", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"),
}


def _enum(dialect_name: str, name: str):
    values = ENUM_TYPES[name]
    if dialect_name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.ENUM(*values, name=name, create_type=False)
    return sa.Enum(*values, name=name)


def frozen_metadata(dialect_name: str) -> sa.MetaData:
    """A fresh ``MetaData`` holding the frozen surface for ``dialect_name``.
    Built per call (32 tables, milliseconds): callers never share or mutate it,
    and one process can migrate databases of different dialects."""
    metadata = sa.MetaData()

    def _index(name: str, table: str, columns: list, **kw) -> sa.Index:
        return sa.Index(name, *(metadata.tables[table].c[column] for column in columns), **kw)

    sa.Table(
        "data",
        metadata,
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
    _index(
        "data_dataset_content_lookup",
        "data",
        ["dataset_id", "owner_id", "content_hash"],
        unique=False,
    )
    _index("ix_data_dataset_id", "data", ["dataset_id"], unique=False)
    _index("ix_data_legacy_id", "data", ["legacy_id"], unique=False)
    _index("ix_data_owner_id", "data", ["owner_id"], unique=False)
    _index("ix_data_tenant_id", "data", ["tenant_id"], unique=False)

    sa.Table(
        "datasets",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.Column("tenant_id", sa.UUID(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_datasets_owner_id", "datasets", ["owner_id"], unique=False)
    _index("ix_datasets_tenant_id", "datasets", ["tenant_id"], unique=False)

    sa.Table(
        "edges",
        metadata,
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
    _index("ix_edges_data_id", "edges", ["data_id"], unique=False)
    _index("ix_edges_dataset_id", "edges", ["dataset_id"], unique=False)
    _index("ix_edges_pipeline_run_id", "edges", ["pipeline_run_id"], unique=False)

    sa.Table(
        "global_database_version",
        metadata,
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("cognee_version", sa.String(), nullable=True),
        sa.Column("global_migration_revision", sa.String(), nullable=True),
        sa.Column("global_migration_last_error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    sa.Table(
        "graph_metrics",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("has_full_metrics", sa.Boolean(), server_default=sa.false(), nullable=False),
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
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    sa.Table(
        "graph_relationship_ledger",
        metadata,
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
    _index("idx_graph_relationship_id", "graph_relationship_ledger", ["id"], unique=False)
    _index(
        "idx_graph_relationship_ledger_destination_node_id",
        "graph_relationship_ledger",
        ["destination_node_id"],
        unique=False,
    )
    _index(
        "idx_graph_relationship_ledger_source_node_id",
        "graph_relationship_ledger",
        ["source_node_id"],
        unique=False,
    )

    sa.Table(
        "integration_credentials",
        metadata,
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
    _index(
        "ix_integration_credentials_provider_account",
        "integration_credentials",
        ["provider", "provider_account_id"],
        unique=True,
    )
    _index(
        "ix_integration_credentials_user_id",
        "integration_credentials",
        ["user_id"],
        unique=False,
    )
    _index(
        "ix_integration_credentials_workspace_id",
        "integration_credentials",
        ["workspace_id"],
        unique=False,
    )

    sa.Table(
        "nodes",
        metadata,
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
    _index("index_node_dataset_data", "nodes", ["dataset_id", "data_id"], unique=False)
    _index("index_node_dataset_slug", "nodes", ["dataset_id", "slug"], unique=False)
    _index("ix_nodes_dataset_id", "nodes", ["dataset_id"], unique=False)
    _index("ix_nodes_pipeline_run_id", "nodes", ["pipeline_run_id"], unique=False)

    sa.Table(
        "permissions",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_permissions_id", "permissions", ["id"], unique=False)
    _index("ix_permissions_name", "permissions", ["name"], unique=True)

    sa.Table(
        "pipeline_runs",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", _enum(dialect_name, "pipelinerunstatus"), nullable=True),
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
    _index("ix_pipeline_runs_created_at_id", "pipeline_runs", ["created_at", "id"], unique=False)
    _index("ix_pipeline_runs_dataset_id", "pipeline_runs", ["dataset_id"], unique=False)
    _index(
        "ix_pipeline_runs_operation_name",
        "pipeline_runs",
        ["operation_name"],
        unique=False,
    )
    _index("ix_pipeline_runs_outcome", "pipeline_runs", ["outcome"], unique=False)
    _index(
        "ix_pipeline_runs_parent_operation_id",
        "pipeline_runs",
        ["parent_operation_id"],
        unique=False,
    )
    _index("ix_pipeline_runs_pipeline_id", "pipeline_runs", ["pipeline_id"], unique=False)
    _index(
        "ix_pipeline_runs_pipeline_run_id",
        "pipeline_runs",
        ["pipeline_run_id"],
        unique=False,
    )
    _index("ix_pipeline_runs_session_id", "pipeline_runs", ["session_id"], unique=False)
    _index("ix_pipeline_runs_user_id", "pipeline_runs", ["user_id"], unique=False)

    sa.Table(
        "principals",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_principals_id", "principals", ["id"], unique=False)

    sa.Table(
        "provenance_entries",
        metadata,
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
    _index("idx_provenance_bundle", "provenance_entries", ["bundle_id"], unique=False)
    _index("idx_provenance_derived_from", "provenance_entries", ["derived_from_id"], unique=False)
    _index("idx_provenance_entity_type", "provenance_entries", ["entity_type"], unique=False)
    _index("idx_provenance_invalidated", "provenance_entries", ["invalidated"], unique=False)
    _index("idx_provenance_parent", "provenance_entries", ["parent_entity_id"], unique=False)
    _index(
        "idx_provenance_prev_version",
        "provenance_entries",
        ["previous_version_id"],
        unique=False,
    )
    _index("idx_provenance_source_ref_key", "provenance_entries", ["source_ref_key"], unique=False)
    _index(
        "ux_provenance_sequence_id",
        "provenance_entries",
        ["sequence_id"],
        unique=True,
        postgresql_where=sa.text("sequence_id IS NOT NULL"),
        sqlite_where=sa.text("sequence_id IS NOT NULL"),
    )

    sa.Table(
        "queries",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("text", sa.String(), nullable=True),
        sa.Column("query_type", sa.String(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("dataset_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_queries_created_at", "queries", ["created_at"], unique=False)
    _index("ix_queries_dataset_id", "queries", ["dataset_id"], unique=False)
    _index("ix_queries_user_id", "queries", ["user_id"], unique=False)

    sa.Table(
        "results",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("query_id", sa.UUID(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("dataset_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_results_created_at", "results", ["created_at"], unique=False)
    _index("ix_results_dataset_id", "results", ["dataset_id"], unique=False)
    _index("ix_results_user_id", "results", ["user_id"], unique=False)

    sa.Table(
        "session_model_usage",
        metadata,
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("session_id", "user_id", "model"),
    )
    _index("ix_session_model_usage_user_id", "session_model_usage", ["user_id"], unique=False)

    sa.Table(
        "session_records",
        metadata,
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
    _index("ix_session_records_dataset_id", "session_records", ["dataset_id"], unique=False)
    _index(
        "ix_session_records_last_activity_at",
        "session_records",
        ["last_activity_at"],
        unique=False,
    )
    _index("ix_session_records_status", "session_records", ["status"], unique=False)
    _index("ix_session_records_user_id", "session_records", ["user_id"], unique=False)

    sa.Table(
        "sync_operations",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("status", _enum(dialect_name, "syncstatus"), nullable=True),
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
    _index("ix_sync_operations_run_id", "sync_operations", ["run_id"], unique=True)
    _index("ix_sync_operations_user_id", "sync_operations", ["user_id"], unique=False)

    sa.Table(
        "tool_connections",
        metadata,
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
    _index("ix_tool_connections_user_id", "tool_connections", ["user_id"], unique=False)

    sa.Table(
        "tool_write_proposals",
        metadata,
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
    _index("ix_tool_write_proposals_status", "tool_write_proposals", ["status"], unique=False)
    _index(
        "ix_tool_write_proposals_user_id",
        "tool_write_proposals",
        ["user_id"],
        unique=False,
    )

    sa.Table(
        "acls",
        metadata,
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

    sa.Table(
        "dataset_configurations",
        metadata,
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

    sa.Table(
        "dataset_database",
        metadata,
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
    _index("ix_dataset_database_dataset_id", "dataset_database", ["dataset_id"], unique=False)
    _index("ix_dataset_database_owner_id", "dataset_database", ["owner_id"], unique=False)

    sa.Table(
        "principal_configuration",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["principals.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    _index(
        "ix_principal_configuration_owner_id",
        "principal_configuration",
        ["owner_id"],
        unique=False,
    )

    sa.Table(
        "tenants",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["id"],
            ["principals.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_tenants_name", "tenants", ["name"], unique=False)
    _index("ix_tenants_owner_id", "tenants", ["owner_id"], unique=False)

    sa.Table(
        "user_api_key",
        metadata,
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
    _index("ix_user_api_key_user_id", "user_api_key", ["user_id"], unique=False)

    sa.Table(
        "roles",
        metadata,
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
    _index("ix_roles_name", "roles", ["name"], unique=False)

    sa.Table(
        "tenant_default_permissions",
        metadata,
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("permission_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id", "permission_id"),
    )

    sa.Table(
        "users",
        metadata,
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=True),
        sa.Column("parent_user_id", sa.UUID(), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("hashed_password", sa.String(length=1024), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False),
        sa.Column("is_verified", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["id"], ["principals.id"], ondelete="CASCADE"),
        # Named as 7c5d4e2f8a91 names it, so its existence guard recognizes
        # this constraint instead of adding a second, auto-named copy.
        sa.ForeignKeyConstraint(
            ["parent_user_id"],
            ["users.id"],
            ondelete="SET NULL",
            name="fk_users_parent_user_id_users",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    _index("ix_users_email", "users", ["email"], unique=True)

    sa.Table(
        "role_default_permissions",
        metadata,
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("role_id", sa.UUID(), nullable=False),
        sa.Column("permission_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("role_id", "permission_id"),
    )

    sa.Table(
        "user_default_permissions",
        metadata,
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("permission_id", sa.UUID(), nullable=False),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "permission_id"),
    )

    sa.Table(
        "user_roles",
        metadata,
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

    sa.Table(
        "user_tenants",
        metadata,
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

    # Chain-provided (d1e2f3a4b5c6), absent from the model surface — see the
    # module docstring.
    _index(
        "ix_pipeline_runs_dataset_pipeline_created_at",
        "pipeline_runs",
        ["dataset_id", "pipeline_name", "created_at"],
        unique=False,
    )

    return metadata


def create_enum_types_if_missing(bind) -> None:
    """Postgres enum types outlive ``DROP TABLE``; create them with checkfirst
    so a recreated database does not collide with the types it left behind."""
    if bind.dialect.name != "postgresql":
        return
    from sqlalchemy.dialects import postgresql

    for name, values in ENUM_TYPES.items():
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)


def create_missing_tables(bind, metadata: sa.MetaData) -> list[str]:
    """Create every frozen table that does not exist yet, whole (with its
    indexes and constraints). Returns the names created. Existing tables are
    left exactly as they are — the reconcile handles their columns."""
    existing = set(sa.inspect(bind).get_table_names())
    created = []
    for table in metadata.sorted_tables:
        if table.name not in existing:
            table.create(bind)
            created.append(table.name)
    return created


def _missing_column(column: sa.Column, dialect_name: str) -> sa.Column:
    """A fresh Column for ADD COLUMN: same type, nullability, foreign keys and
    server default VALUE (a Column can belong to one Table only). A foreign key
    keeps its frozen constraint name; SQLite's batch mode can only add NAMED
    constraints, so an unnamed one gets the chain's own convention there."""
    foreign_keys = []
    for fk in column.foreign_keys:
        name = fk.constraint.name if fk.constraint is not None else None
        if name is None and dialect_name == "sqlite":
            name = f"fk_{column.table.name}_{column.name}_{fk.column.table.name}"
        foreign_keys.append(sa.ForeignKey(fk.target_fullname, ondelete=fk.ondelete, name=name))
    return sa.Column(
        column.name,
        column.type,
        *foreign_keys,
        nullable=column.nullable,
        server_default=None if column.server_default is None else column.server_default.arg,
    )


def _needs_table_rebuild(column: sa.Column) -> bool:
    """SQLite's ADD COLUMN accepts only constant defaults; a function default
    (``now()``) needs the batch-mode table rebuild."""
    return column.server_default is not None and isinstance(
        column.server_default.arg, sa.sql.functions.FunctionElement
    )


def reconcile(op, metadata: sa.MetaData) -> None:
    """Converge the database on ``bind`` to the frozen surface, additively and
    idempotently: missing tables created whole, missing columns and indexes
    added to existing tables, nothing dropped or altered. Plans first and
    raises BEFORE any change if a primary key, or a NOT NULL column without a
    server default on a populated table, is missing — those need a dedicated
    migration. ``op`` is Alembic's operations proxy for the running revision:
    it supplies the bind and ``add_column`` / ``batch_alter_table`` (the latter
    required on SQLite, whose ADD COLUMN rejects function defaults)."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    missing_columns: dict[sa.Table, list[sa.Column]] = {}
    for table in metadata.sorted_tables:
        if table.name in existing:
            present = {column["name"] for column in inspector.get_columns(table.name)}
            absent = [column for column in table.columns if column.name not in present]
            if absent:
                missing_columns[table] = absent

    blocked = []
    for table, columns in missing_columns.items():
        for column in columns:
            if column.primary_key:
                blocked.append(f"{table.name}.{column.name} (primary key)")
            elif not column.nullable and column.server_default is None:
                populated = bind.execute(sa.text(f'SELECT 1 FROM "{table.name}" LIMIT 1')).first()
                if populated is not None:
                    blocked.append(f"{table.name}.{column.name}")
    if blocked:
        raise RuntimeError(
            "Cannot reconcile: these columns are missing and cannot be added by ADD COLUMN "
            "(a primary key, or NOT NULL without a server default on a populated table); "
            f"they need a dedicated migration: {blocked}"
        )

    create_enum_types_if_missing(bind)
    create_missing_tables(bind, metadata)
    for table, columns in missing_columns.items():
        if bind.dialect.name == "sqlite":
            recreate = "always" if any(_needs_table_rebuild(c) for c in columns) else "auto"
            with op.batch_alter_table(table.name, recreate=recreate) as batch_op:
                for column in columns:
                    batch_op.add_column(_missing_column(column, "sqlite"))
        else:
            for column in columns:
                op.add_column(table.name, _missing_column(column, bind.dialect.name))

    # Indexes last: some sit on columns added just above. A fresh inspector —
    # the one above predates those changes.
    inspector = sa.inspect(bind)
    for table in metadata.sorted_tables:
        if table.name in existing:
            present = {index["name"] for index in inspector.get_indexes(table.name)}
            for index in table.indexes:
                if index.name not in present:
                    index.create(bind)


def fingerprint() -> str:
    """A dialect-independent digest of the frozen surface — tables, columns
    (type, nullability, server default), primary keys, foreign keys, unique
    constraints and indexes — pinned by the seal test."""
    metadata = frozen_metadata("sqlite")
    description = {
        table.name: {
            "columns": {
                c.name: [
                    str(c.type),
                    bool(c.nullable),
                    None if c.server_default is None else str(c.server_default.arg),
                ]
                for c in table.columns
            },
            "pk": [c.name for c in table.primary_key.columns],
            "fks": sorted(
                [
                    [list(k.column_keys), k.referred_table.name, k.name or ""]
                    for k in table.foreign_key_constraints
                ]
            ),
            "uniques": sorted(
                [
                    sorted(c.name for c in k.columns)
                    for k in table.constraints
                    if isinstance(k, sa.UniqueConstraint)
                ]
            ),
            "indexes": sorted(
                [[i.name, [c.name for c in i.columns], bool(i.unique)] for i in table.indexes]
            ),
        }
        for table in metadata.sorted_tables
    }
    canonical = json.dumps(description, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
