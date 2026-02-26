"""New consolidated initial schema

Creates the full current database schema in a single migration. Intended for
fresh installations where the complete schema should be bootstrapped via
``alembic upgrade deadbeef0001`` rather than relying on
``Base.metadata.create_all()``.

Existing installations that already have the schema from the original
migration chain are unaffected: every ``create_table`` call is guarded by an
existence check and will be skipped when the table is already present.

Revision ID: deadbeef0001
Revises: (none – this is an alternative root for fresh installs)
Create Date: 2026-02-26 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "deadbeef0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = ("schema_init",)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = set(inspector.get_table_names())
    dialect = op.get_context().dialect.name

    # -------------------------------------------------------------------------
    # principals
    # Base table for single-table inheritance used by users, tenants and roles.
    # -------------------------------------------------------------------------
    if "principals" not in existing_tables:
        op.create_table(
            "principals",
            sa.Column("id", sa.UUID(), primary_key=True, index=True),
            sa.Column("type", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # permissions
    # -------------------------------------------------------------------------
    if "permissions" not in existing_tables:
        op.create_table(
            "permissions",
            sa.Column("id", sa.UUID(), primary_key=True, index=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(op.f("ix_permissions_name"), "permissions", ["name"], unique=True)

    # -------------------------------------------------------------------------
    # tenants  (joined-table inheritance from principals)
    # -------------------------------------------------------------------------
    if "tenants" not in existing_tables:
        op.create_table(
            "tenants",
            sa.Column("id", sa.UUID(), sa.ForeignKey("principals.id"), primary_key=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("owner_id", sa.UUID(), nullable=True),
        )
        op.create_index(op.f("ix_tenants_name"), "tenants", ["name"], unique=True)
        op.create_index(op.f("ix_tenants_owner_id"), "tenants", ["owner_id"], unique=False)

    # -------------------------------------------------------------------------
    # users  (joined-table inheritance from principals; fastapi-users columns)
    # -------------------------------------------------------------------------
    if "users" not in existing_tables:
        op.create_table(
            "users",
            sa.Column(
                "id",
                sa.UUID(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            # fastapi-users columns
            sa.Column("email", sa.String(length=320), nullable=False),
            sa.Column("hashed_password", sa.String(length=1024), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("is_superuser", sa.Boolean(), nullable=False),
            sa.Column("is_verified", sa.Boolean(), nullable=False),
            # cognee columns
            sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id"), nullable=True),
        )
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # -------------------------------------------------------------------------
    # roles  (joined-table inheritance from principals)
    # -------------------------------------------------------------------------
    if "roles" not in existing_tables:
        op.create_table(
            "roles",
            sa.Column(
                "id",
                sa.UUID(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id"), nullable=False),
            sa.UniqueConstraint("tenant_id", "name", name="uq_roles_tenant_id_name"),
        )
        op.create_index(op.f("ix_roles_name"), "roles", ["name"], unique=False)

    # -------------------------------------------------------------------------
    # datasets
    # -------------------------------------------------------------------------
    if "datasets" not in existing_tables:
        op.create_table(
            "datasets",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("name", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
        )
        op.create_index(op.f("ix_datasets_owner_id"), "datasets", ["owner_id"], unique=False)
        op.create_index(op.f("ix_datasets_tenant_id"), "datasets", ["tenant_id"], unique=False)

    # -------------------------------------------------------------------------
    # data
    # -------------------------------------------------------------------------
    if "data" not in existing_tables:
        op.create_table(
            "data",
            sa.Column("id", sa.UUID(), primary_key=True),
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
            sa.Column("content_hash", sa.String(), nullable=True),
            sa.Column("raw_content_hash", sa.String(), nullable=True),
            sa.Column("external_metadata", sa.JSON(), nullable=True),
            sa.Column("node_set", sa.JSON(), nullable=True),
            sa.Column(
                "pipeline_status",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
            sa.Column("token_count", sa.Integer(), nullable=True),
            sa.Column("data_size", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(op.f("ix_data_owner_id"), "data", ["owner_id"], unique=False)
        op.create_index(op.f("ix_data_tenant_id"), "data", ["tenant_id"], unique=False)

    # -------------------------------------------------------------------------
    # dataset_data  (junction table: datasets ↔ data)
    # -------------------------------------------------------------------------
    if "dataset_data" not in existing_tables:
        op.create_table(
            "dataset_data",
            sa.Column(
                "dataset_id",
                sa.UUID(),
                sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "data_id",
                sa.UUID(),
                sa.ForeignKey("data.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # acls
    # -------------------------------------------------------------------------
    if "acls" not in existing_tables:
        op.create_table(
            "acls",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("principal_id", sa.UUID(), sa.ForeignKey("principals.id"), nullable=True),
            sa.Column("permission_id", sa.UUID(), sa.ForeignKey("permissions.id"), nullable=True),
            sa.Column(
                "dataset_id",
                sa.UUID(),
                sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # user_roles  (junction table: users ↔ roles)
    # -------------------------------------------------------------------------
    if "user_roles" not in existing_tables:
        op.create_table(
            "user_roles",
            sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), primary_key=True),
            sa.Column("role_id", sa.UUID(), sa.ForeignKey("roles.id"), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # user_tenants  (junction table: users ↔ tenants)
    # -------------------------------------------------------------------------
    if "user_tenants" not in existing_tables:
        op.create_table(
            "user_tenants",
            sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), primary_key=True),
            sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id"), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # role_default_permissions  (junction table: roles ↔ permissions)
    # -------------------------------------------------------------------------
    if "role_default_permissions" not in existing_tables:
        op.create_table(
            "role_default_permissions",
            sa.Column(
                "role_id",
                sa.UUID(),
                sa.ForeignKey("roles.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "permission_id",
                sa.UUID(),
                sa.ForeignKey("permissions.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # user_default_permissions  (junction table: users ↔ permissions)
    # -------------------------------------------------------------------------
    if "user_default_permissions" not in existing_tables:
        op.create_table(
            "user_default_permissions",
            sa.Column(
                "user_id",
                sa.UUID(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "permission_id",
                sa.UUID(),
                sa.ForeignKey("permissions.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # tenant_default_permissions  (junction table: tenants ↔ permissions)
    # -------------------------------------------------------------------------
    if "tenant_default_permissions" not in existing_tables:
        op.create_table(
            "tenant_default_permissions",
            sa.Column(
                "tenant_id",
                sa.UUID(),
                sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "permission_id",
                sa.UUID(),
                sa.ForeignKey("permissions.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # dataset_database
    # -------------------------------------------------------------------------
    if "dataset_database" not in existing_tables:
        op.create_table(
            "dataset_database",
            sa.Column(
                "owner_id",
                sa.UUID(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column(
                "dataset_id",
                sa.UUID(),
                sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                primary_key=True,
            ),
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
            sa.Column(
                "graph_database_connection_info",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
            sa.Column(
                "vector_database_connection_info",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(
            op.f("ix_dataset_database_owner_id"), "dataset_database", ["owner_id"], unique=False
        )
        op.create_index(
            op.f("ix_dataset_database_dataset_id"),
            "dataset_database",
            ["dataset_id"],
            unique=False,
        )

    # -------------------------------------------------------------------------
    # principal_configuration
    # -------------------------------------------------------------------------
    if "principal_configuration" not in existing_tables:
        op.create_table(
            "principal_configuration",
            sa.Column("id", sa.UUID(), primary_key=True, index=True),
            sa.Column(
                "owner_id",
                sa.UUID(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("configuration", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(
            op.f("ix_principal_configuration_owner_id"),
            "principal_configuration",
            ["owner_id"],
            unique=False,
        )

    # -------------------------------------------------------------------------
    # queries
    # -------------------------------------------------------------------------
    if "queries" not in existing_tables:
        op.create_table(
            "queries",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("text", sa.String(), nullable=True),
            sa.Column("query_type", sa.String(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # results
    # -------------------------------------------------------------------------
    if "results" not in existing_tables:
        op.create_table(
            "results",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("value", sa.Text(), nullable=True),
            sa.Column("query_id", sa.UUID(), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(op.f("ix_results_user_id"), "results", ["user_id"], unique=False)

    # -------------------------------------------------------------------------
    # notebooks
    # -------------------------------------------------------------------------
    if "notebooks" not in existing_tables:
        op.create_table(
            "notebooks",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("owner_id", sa.UUID(), nullable=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("cells", sa.JSON(), nullable=False),
            sa.Column("deletable", sa.Boolean(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(op.f("ix_notebooks_owner_id"), "notebooks", ["owner_id"], unique=False)

    # -------------------------------------------------------------------------
    # graph_metrics
    # -------------------------------------------------------------------------
    if "graph_metrics" not in existing_tables:
        op.create_table(
            "graph_metrics",
            sa.Column("id", sa.UUID(), primary_key=True),
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
        )

    # -------------------------------------------------------------------------
    # graph_relationship_ledger  (legacy; retained for backwards compatibility)
    # -------------------------------------------------------------------------
    if "graph_relationship_ledger" not in existing_tables:
        op.create_table(
            "graph_relationship_ledger",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("source_node_id", sa.UUID(), nullable=False),
            sa.Column("destination_node_id", sa.UUID(), nullable=False),
            sa.Column("creator_function", sa.String(), nullable=False),
            sa.Column("node_label", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("user_id", sa.UUID(), nullable=True),
        )
        op.create_index(
            "idx_graph_relationship_id", "graph_relationship_ledger", ["id"], unique=False
        )
        op.create_index(
            "idx_graph_relationship_ledger_source_node_id",
            "graph_relationship_ledger",
            ["source_node_id"],
            unique=False,
        )
        op.create_index(
            "idx_graph_relationship_ledger_destination_node_id",
            "graph_relationship_ledger",
            ["destination_node_id"],
            unique=False,
        )

    # -------------------------------------------------------------------------
    # nodes
    # -------------------------------------------------------------------------
    if "nodes" not in existing_tables:
        op.create_table(
            "nodes",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("slug", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("data_id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("label", sa.String(255), nullable=True),
            sa.Column("type", sa.String(255), nullable=False),
            sa.Column("indexed_fields", sa.JSON(), nullable=False),
            sa.Column("attributes", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(op.f("ix_nodes_dataset_id"), "nodes", ["dataset_id"], unique=False)
        op.create_index("index_node_dataset_slug", "nodes", ["dataset_id", "slug"], unique=False)
        op.create_index("index_node_dataset_data", "nodes", ["dataset_id", "data_id"], unique=False)

    # -------------------------------------------------------------------------
    # edges
    # -------------------------------------------------------------------------
    if "edges" not in existing_tables:
        op.create_table(
            "edges",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("slug", sa.UUID(), nullable=False),
            sa.Column("user_id", sa.UUID(), nullable=False),
            sa.Column("data_id", sa.UUID(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=False),
            sa.Column("source_node_id", sa.UUID(), nullable=False),
            sa.Column("destination_node_id", sa.UUID(), nullable=False),
            sa.Column("relationship_name", sa.Text(), nullable=False),
            sa.Column("label", sa.Text(), nullable=True),
            sa.Column("attributes", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(op.f("ix_edges_data_id"), "edges", ["data_id"], unique=False)
        op.create_index(op.f("ix_edges_dataset_id"), "edges", ["dataset_id"], unique=False)

    # -------------------------------------------------------------------------
    # pipelines
    # -------------------------------------------------------------------------
    if "pipelines" not in existing_tables:
        op.create_table(
            "pipelines",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # tasks
    # -------------------------------------------------------------------------
    if "tasks" not in existing_tables:
        op.create_table(
            "tasks",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("name", sa.String(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("executable", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # pipeline_task  (junction table: pipelines ↔ tasks)
    # Note: the ORM model uses column-names "pipeline" / "task" (not the
    # attribute names pipeline_id / task_id), and its ForeignKey strings point
    # at the non-existent tables "pipeline" / "task" rather than "pipelines" /
    # "tasks".  The FK declarations are therefore omitted here to avoid
    # referential errors; the columns are created as plain UUIDs matching the
    # DB column names that create_all() would produce.
    # -------------------------------------------------------------------------
    if "pipeline_task" not in existing_tables:
        op.create_table(
            "pipeline_task",
            sa.Column("pipeline", sa.UUID(), primary_key=True),
            sa.Column("task", sa.UUID(), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # -------------------------------------------------------------------------
    # task_runs
    # -------------------------------------------------------------------------
    if "task_runs" not in existing_tables:
        op.create_table(
            "task_runs",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("task_name", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("run_info", sa.JSON(), nullable=True),
        )

    # -------------------------------------------------------------------------
    # pipeline_runs
    # -------------------------------------------------------------------------
    if "pipeline_runs" not in existing_tables:
        _PIPELINE_RUN_STATUS_VALUES = (
            "DATASET_PROCESSING_INITIATED",
            "DATASET_PROCESSING_STARTED",
            "DATASET_PROCESSING_COMPLETED",
            "DATASET_PROCESSING_ERRORED",
        )
        if dialect == "postgresql":
            pipelinerunstatus_pg = postgresql.ENUM(
                *_PIPELINE_RUN_STATUS_VALUES,
                name="pipelinerunstatus",
            )
            pipelinerunstatus_pg.create(conn, checkfirst=True)
            status_col = sa.Column(
                "status",
                postgresql.ENUM(
                    *_PIPELINE_RUN_STATUS_VALUES,
                    name="pipelinerunstatus",
                    create_type=False,
                ),
                nullable=True,
            )
        else:
            status_col = sa.Column(
                "status",
                sa.Enum(*_PIPELINE_RUN_STATUS_VALUES, name="pipelinerunstatus"),
                nullable=True,
            )

        op.create_table(
            "pipeline_runs",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            status_col,
            sa.Column("pipeline_run_id", sa.UUID(), nullable=True),
            sa.Column("pipeline_name", sa.String(), nullable=True),
            sa.Column("pipeline_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("run_info", sa.JSON(), nullable=True),
        )
        op.create_index(
            op.f("ix_pipeline_runs_pipeline_run_id"),
            "pipeline_runs",
            ["pipeline_run_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_pipeline_runs_pipeline_id"), "pipeline_runs", ["pipeline_id"], unique=False
        )
        op.create_index(
            op.f("ix_pipeline_runs_dataset_id"), "pipeline_runs", ["dataset_id"], unique=False
        )

    # -------------------------------------------------------------------------
    # sync_operations
    # Enum values are lowercase to match the SyncStatus Python enum's .value
    # attributes as defined in the ORM model.
    # -------------------------------------------------------------------------
    if "sync_operations" not in existing_tables:
        _SYNC_STATUS_VALUES = ("started", "in_progress", "completed", "failed", "cancelled")

        if dialect == "postgresql":
            syncstatus_pg = postgresql.ENUM(*_SYNC_STATUS_VALUES, name="syncstatus")
            syncstatus_pg.create(conn, checkfirst=True)
            status_col = sa.Column(
                "status",
                postgresql.ENUM(*_SYNC_STATUS_VALUES, name="syncstatus", create_type=False),
                nullable=True,
            )
        else:
            status_col = sa.Column(
                "status",
                sa.Enum(*_SYNC_STATUS_VALUES, name="syncstatus"),
                nullable=True,
            )

        op.create_table(
            "sync_operations",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column("run_id", sa.Text(), nullable=True),
            status_col,
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
        )
        op.create_index(
            op.f("ix_sync_operations_run_id"), "sync_operations", ["run_id"], unique=True
        )
        op.create_index(
            op.f("ix_sync_operations_user_id"), "sync_operations", ["user_id"], unique=False
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = set(inspector.get_table_names())
    dialect = op.get_context().dialect.name

    # Drop in reverse dependency order.
    _drop_order = [
        "sync_operations",
        "pipeline_runs",
        "task_runs",
        "pipeline_task",
        "tasks",
        "pipelines",
        "edges",
        "nodes",
        "graph_relationship_ledger",
        "graph_metrics",
        "notebooks",
        "results",
        "queries",
        "principal_configuration",
        "dataset_database",
        "tenant_default_permissions",
        "user_default_permissions",
        "role_default_permissions",
        "user_tenants",
        "user_roles",
        "acls",
        "dataset_data",
        "data",
        "datasets",
        "roles",
        "users",
        "tenants",
        "permissions",
        "principals",
    ]

    for table_name in _drop_order:
        if table_name in existing_tables:
            op.drop_table(table_name)

    if dialect == "postgresql":
        sa.Enum(name="pipelinerunstatus").drop(conn, checkfirst=True)
        sa.Enum(name="syncstatus").drop(conn, checkfirst=True)
