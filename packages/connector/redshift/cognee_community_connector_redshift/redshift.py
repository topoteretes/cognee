"""Amazon Redshift data-source connector for Cognee.

Uses Amazon Redshift Data API (via boto3) as the transport layer to enable
seamless data ingestion from Redshift clusters or Redshift Serverless workgroups,
including clusters deployed inside private VPCs.
"""

import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

import dlt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RedshiftConnectorError(Exception):
    """Base class for Redshift connector exceptions."""


class RedshiftAPIError(RedshiftConnectorError):
    """Raised when Redshift Data API execution fails or returns an error."""


class RedshiftAuthError(RedshiftConnectorError):
    """Raised when AWS or Redshift authentication fails."""


class RedshiftTimeoutError(RedshiftConnectorError):
    """Raised when statement execution polling exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Data API Client & Transport
# ---------------------------------------------------------------------------
class RedshiftClient:
    """Synchronous Amazon Redshift Data API client wrapping boto3.

    Handles statement execution, polling, pagination, and error checking.
    Credentials and connection details default to AWS environment variables
    and IAM standard resolution.
    """

    def __init__(
        self,
        database: str,
        workgroup_name: str | None = None,
        cluster_identifier: str | None = None,
        secret_arn: str | None = None,
        db_user: str | None = None,
        region_name: str | None = None,
        boto3_client: Any | None = None,
        timeout_seconds: int = 300,
        poll_interval_seconds: float = 1.0,
    ):
        if not database:
            raise ValueError("database parameter is required")
        if not workgroup_name and not cluster_identifier:
            raise ValueError("Either workgroup_name or cluster_identifier must be provided")

        self.database = database
        self.workgroup_name = workgroup_name
        self.cluster_identifier = cluster_identifier
        self.secret_arn = secret_arn
        self.db_user = db_user
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

        if boto3_client is not None:
            self._client = boto3_client
        else:
            import boto3

            kwargs = {}
            if region_name:
                kwargs["region_name"] = region_name
            self._client = boto3.client("redshift-data", **kwargs)

    def _execution_target_kwargs(self) -> dict[str, Any]:
        """Build target parameter dict for ExecuteStatement."""
        target: dict[str, Any] = {"Database": self.database}
        if self.workgroup_name:
            target["WorkgroupName"] = self.workgroup_name
        elif self.cluster_identifier:
            target["ClusterIdentifier"] = self.cluster_identifier

        if self.secret_arn:
            target["SecretArn"] = self.secret_arn
        elif self.db_user:
            target["DbUser"] = self.db_user

        return target

    def execute_statement(self, sql: str) -> str:
        """Submit a SQL statement to Redshift Data API and return statement ID."""
        kwargs = self._execution_target_kwargs()
        kwargs["Sql"] = sql

        try:
            response = self._client.execute_statement(**kwargs)
            statement_id = response.get("Id")
            if not statement_id:
                raise RedshiftAPIError("ExecuteStatement response did not return a statement Id")
            return statement_id
        except Exception as exc:
            if type(exc).__name__ in ("ClientError", "ParamValidationError"):
                raise RedshiftAPIError(f"Redshift Data API execution failed: {exc}") from exc
            raise

    def poll_statement(self, statement_id: str) -> dict[str, Any]:
        """Poll DescribeStatement until statement finishes, fails, or times out."""
        start_time = time.time()
        poll_interval = self.poll_interval_seconds

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                raise RedshiftTimeoutError(
                    f"Redshift statement {statement_id} timed out after {self.timeout_seconds} seconds"
                )

            try:
                response = self._client.describe_statement(Id=statement_id)
            except Exception as exc:
                raise RedshiftAPIError(f"DescribeStatement call failed: {exc}") from exc

            status = response.get("Status")
            if status == "FINISHED":
                return response
            elif status == "FAILED":
                error_msg = response.get("Error", "Unknown execution error")
                raise RedshiftAPIError(f"Redshift query failed (statement {statement_id}): {error_msg}")
            elif status == "ABORTED":
                raise RedshiftAPIError(f"Redshift query was aborted (statement {statement_id})")

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5.0)

    def fetch_all_pages(self, statement_id: str) -> Iterator[dict[str, Any]]:
        """Paginate through statement results and yield dictionary rows."""
        next_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {"Id": statement_id}
            if next_token:
                kwargs["NextToken"] = next_token

            try:
                response = self._client.get_statement_result(**kwargs)
            except Exception as exc:
                raise RedshiftAPIError(f"GetStatementResult call failed: {exc}") from exc

            column_metadata = response.get("ColumnMetadata", [])
            col_names = [col.get("name", f"col_{idx}") for idx, col in enumerate(column_metadata)]
            records = response.get("Records", [])

            for record in records:
                row_dict: dict[str, Any] = {}
                for idx, col_name in enumerate(col_names):
                    if idx < len(record):
                        val_dict = record[idx]
                        row_dict[col_name] = _extract_column_value(val_dict)
                    else:
                        row_dict[col_name] = None
                yield row_dict

            next_token = response.get("NextToken")
            if not next_token:
                break

    def execute_and_fetch(self, sql: str) -> list[dict[str, Any]]:
        """Convenience method to execute a statement, wait for completion, and return all rows."""
        statement_id = self.execute_statement(sql)
        self.poll_statement(statement_id)
        return list(self.fetch_all_pages(statement_id))


def _extract_column_value(val_dict: dict[str, Any]) -> Any:
        """Extract typed primitive value from Redshift Data API ColumnValue dict."""
        if not isinstance(val_dict, dict):
            return None
        if val_dict.get("isNull"):
            return None

        if "stringValue" in val_dict:
            return val_dict["stringValue"]
        if "longValue" in val_dict:
            return val_dict["longValue"]
        if "doubleValue" in val_dict:
            return val_dict["doubleValue"]
        if "booleanValue" in val_dict:
            return val_dict["booleanValue"]
        if "blobValue" in val_dict:
            return val_dict["blobValue"]

        return next(iter(val_dict.values())) if val_dict else None


# ---------------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------------
def render_table_metadata(
    schema: str,
    table: str,
    table_comment: str | None = None,
    column_comments: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render a standalone metadata document describing table structure and comments."""
    lines = [f"Table: {schema}.{table}"]
    if table_comment:
        lines.append(f"Description: {table_comment}")

    if column_comments:
        lines.append("Columns:")
        for col, col_doc in sorted(column_comments.items()):
            lines.append(f"  - {col}: {col_doc}")

    content = "\n".join(lines)
    return {
        "id": f"meta:{schema}.{table}",
        "title": f"Table Schema: {schema}.{table}",
        "content": content,
        "_deleted": False,
    }


def render_row(
    schema: str,
    table: str,
    primary_key: str,
    row: dict[str, Any],
    soft_delete_col: str | None = None,
) -> dict[str, Any]:
    """Render a data row into a clean dictionary for Cognee ingestion."""
    pk_val = row.get(primary_key)
    if pk_val is None:
        raise ValueError(f"Row is missing primary key column '{primary_key}': {row}")

    is_deleted = False
    if soft_delete_col and soft_delete_col in row:
        val = row[soft_delete_col]
        is_deleted = bool(val) if not isinstance(val, str) else val.lower() in ("true", "1", "t", "yes")

    content_lines = []
    for key, val in sorted(row.items()):
        if soft_delete_col and key == soft_delete_col:
            continue
        content_lines.append(f"{key}: {val}")

    content_text = "\n".join(content_lines) or f"{table} record #{pk_val}"

    return {
        "id": f"{schema}.{table}:{pk_val}",
        "title": f"{table} #{pk_val}",
        "content": content_text,
        "table_name": table,
        "schema_name": schema,
        "_deleted": is_deleted,
    }


def _escape_identifier(identifier: str) -> str:
    """Validate and double-quote a SQL identifier to prevent SQL injection."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("SQL identifier must be a non-empty string")
    cleaned = identifier.strip().strip('"')
    if not re.match(r"^[A-Za-z0-9_]+$", cleaned):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return f'"{cleaned}"'


# ---------------------------------------------------------------------------
# DLT Source and Resource
# ---------------------------------------------------------------------------
@dlt.source(name="redshift_source")
def redshift_source(
    client: RedshiftClient,
    table_name: str,
    primary_key: str,
    schema: str = "public",
    timestamp_column: str | None = None,
    soft_delete_column: str | None = None,
    reconcile_deletions: bool = True,
) -> Any:
    """Create a DLT source for ingesting data from an Amazon Redshift table into Cognee."""
    if not table_name:
        raise ValueError("table_name is required")
    if not primary_key:
        raise ValueError("primary_key is required")

    safe_schema = _escape_identifier(schema)
    safe_table = _escape_identifier(table_name)
    safe_pk = _escape_identifier(primary_key)
    safe_ts = _escape_identifier(timestamp_column) if timestamp_column else None

    resource_name = f"{schema}_{table_name}".lower().replace(".", "_")

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def redshift_table_resource() -> Iterator[dict[str, Any]]:
        state = dlt.current.resource_state()

        # Step 1: Yield table metadata document once
        if not state.get("metadata_emitted"):
            meta_doc = render_table_metadata(
                schema=schema,
                table=table_name,
                table_comment=None,
                column_comments=None,
            )
            yield meta_doc

        # Step 2: Incremental data extraction
        last_timestamp = state.get("last_timestamp")
        last_pk = state.get("last_pk")

        query_parts = [f"SELECT * FROM {safe_schema}.{safe_table}"]
        where_conditions = []

        if safe_ts and last_timestamp is not None:
            where_conditions.append(f"{safe_ts} >= '{last_timestamp}'")

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        if safe_ts:
            query_parts.append(f"ORDER BY {safe_ts} ASC, {safe_pk} ASC")
        else:
            query_parts.append(f"ORDER BY {safe_pk} ASC")

        sql_query = " ".join(query_parts)

        statement_id = client.execute_statement(sql_query)
        client.poll_statement(statement_id)

        candidate_last_timestamp = last_timestamp
        candidate_last_pk = last_pk
        scanned_pks: set[str] = set()

        for row in client.fetch_all_pages(statement_id):
            pk_val = str(row.get(primary_key))
            scanned_pks.add(pk_val)

            if timestamp_column and timestamp_column in row:
                row_ts = str(row[timestamp_column])
                # Skip duplicate rows at the tie-break boundary
                if last_timestamp is not None and row_ts == str(last_timestamp) and last_pk is not None and pk_val <= str(last_pk):
                    continue

                if candidate_last_timestamp is None or row_ts > str(candidate_last_timestamp):
                    candidate_last_timestamp = row_ts
                    candidate_last_pk = pk_val
                elif row_ts == str(candidate_last_timestamp):
                    candidate_last_pk = pk_val

            rendered = render_row(
                schema=schema,
                table=table_name,
                primary_key=primary_key,
                row=row,
                soft_delete_col=soft_delete_column,
            )
            yield rendered

        # Step 3: Deletion Reconciliation via PK Inventory (only when complete)
        known_pks = set(state.get("known_pks") or [])

        if reconcile_deletions and known_pks:
            pk_query = f"SELECT {safe_pk} FROM {safe_schema}.{safe_table}"
            pk_stmt_id = client.execute_statement(pk_query)
            client.poll_statement(pk_stmt_id)

            active_pks: set[str] = set()
            for pk_row in client.fetch_all_pages(pk_stmt_id):
                val = pk_row.get(primary_key)
                if val is not None:
                    active_pks.add(str(val))

            # Emit tombstones for missing PKs
            deleted_pks = known_pks - active_pks
            for missing_pk in sorted(deleted_pks):
                yield {
                    "id": f"{schema}.{table_name}:{missing_pk}",
                    "_deleted": True,
                }
            updated_known_pks = active_pks
        else:
            updated_known_pks = known_pks | scanned_pks

        # Step 4: Atomic Checkpoint Commit (runs ONLY after all pages & queries finish)
        state["metadata_emitted"] = True
        state["last_timestamp"] = candidate_last_timestamp
        state["last_pk"] = candidate_last_pk
        state["known_pks"] = sorted(updated_known_pks)

    return redshift_table_resource
