"""Unit tests for Amazon Redshift community connector."""

from unittest import mock

import dlt.extract.exceptions
import pytest
from cognee_community_connector_redshift.redshift import (
    RedshiftAPIError,
    RedshiftClient,
    RedshiftTimeoutError,
    _escape_identifier,
    _extract_column_value,
    redshift_source,
    render_row,
    render_table_metadata,
)


def test_escape_identifier():
    """Validate and double-quote SQL identifiers."""
    assert _escape_identifier("users") == '"users"'
    assert _escape_identifier("public") == '"public"'

    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        _escape_identifier("users; DROP TABLE users;--")

    with pytest.raises(ValueError, match="SQL identifier must be a non-empty string"):
        _escape_identifier("")


def test_client_init_validation():
    """Client validates required parameters."""
    with pytest.raises(ValueError, match="database parameter is required"):
        RedshiftClient(database="", workgroup_name="wg")

    with pytest.raises(ValueError, match="Either workgroup_name or cluster_identifier"):
        RedshiftClient(database="dev")


def test_client_execution_target_kwargs():
    """Target kwargs correctly resolve serverless vs provisioned and secret vs db_user."""
    client_serverless = RedshiftClient(
        database="dev",
        workgroup_name="my-wg",
        secret_arn="arn:aws:secretsmanager:us-east-1:123456789012:secret:mysecret",
        boto3_client=mock.MagicMock(),
    )
    assert client_serverless._execution_target_kwargs() == {
        "Database": "dev",
        "WorkgroupName": "my-wg",
        "SecretArn": "arn:aws:secretsmanager:us-east-1:123456789012:secret:mysecret",
    }

    client_provisioned = RedshiftClient(
        database="dev",
        cluster_identifier="my-cluster",
        db_user="awsuser",
        boto3_client=mock.MagicMock(),
    )
    assert client_provisioned._execution_target_kwargs() == {
        "Database": "dev",
        "ClusterIdentifier": "my-cluster",
        "DbUser": "awsuser",
    }


def test_column_value_extraction():
    """Extract primitive values from Redshift Data API ColumnValue dicts."""
    assert _extract_column_value({"stringValue": "hello"}) == "hello"
    assert _extract_column_value({"longValue": 42}) == 42
    assert _extract_column_value({"doubleValue": 99.9}) == 99.9
    assert _extract_column_value({"booleanValue": True}) is True
    assert _extract_column_value({"isNull": True}) is None
    assert _extract_column_value(None) is None


def test_execute_statement_success():
    """Execute statement returns statement ID on success."""
    mock_boto = mock.MagicMock()
    mock_boto.execute_statement.return_value = {"Id": "stmt-123"}

    client = RedshiftClient(database="dev", workgroup_name="wg", boto3_client=mock_boto)
    stmt_id = client.execute_statement("SELECT 1")

    assert stmt_id == "stmt-123"
    mock_boto.execute_statement.assert_called_once_with(
        Database="dev", WorkgroupName="wg", Sql="SELECT 1"
    )


def test_poll_statement_status():
    """Polling handles FINISHED, FAILED, and TIMEOUT status cleanly."""
    mock_boto = mock.MagicMock()
    client = RedshiftClient(
        database="dev", workgroup_name="wg", boto3_client=mock_boto, timeout_seconds=2, poll_interval_seconds=0.01
    )

    # 1. Finished
    mock_boto.describe_statement.return_value = {"Status": "FINISHED"}
    res = client.poll_statement("stmt-1")
    assert res["Status"] == "FINISHED"

    # 2. Failed
    mock_boto.describe_statement.return_value = {"Status": "FAILED", "Error": "Syntax error"}
    with pytest.raises(RedshiftAPIError, match="Syntax error"):
        client.poll_statement("stmt-2")

    # 3. Timeout
    mock_boto.describe_statement.return_value = {"Status": "STARTED"}
    with pytest.raises(RedshiftTimeoutError):
        client.poll_statement("stmt-3")


def test_fetch_all_pages_multi_page():
    """Pagination fetches all pages when NextToken is present."""
    mock_boto = mock.MagicMock()
    client = RedshiftClient(database="dev", workgroup_name="wg", boto3_client=mock_boto)

    page1 = {
        "ColumnMetadata": [{"name": "id"}, {"name": "val"}],
        "Records": [[{"longValue": 1}, {"stringValue": "A"}]],
        "NextToken": "token-page-2",
    }
    page2 = {
        "ColumnMetadata": [{"name": "id"}, {"name": "val"}],
        "Records": [[{"longValue": 2}, {"stringValue": "B"}]],
    }
    mock_boto.get_statement_result.side_effect = [page1, page2]

    rows = list(client.fetch_all_pages("stmt-123"))

    assert len(rows) == 2
    assert rows[0] == {"id": 1, "val": "A"}
    assert rows[1] == {"id": 2, "val": "B"}
    assert mock_boto.get_statement_result.call_count == 2


def test_rendering_helpers():
    """Render table metadata and data rows correctly."""
    meta = render_table_metadata(
        schema="public",
        table="users",
        table_comment="User accounts table",
        column_comments={"id": "User primary key", "email": "User email address"},
    )
    assert meta["id"] == "meta:public.users"
    assert "Table: public.users" in meta["content"]
    assert "Description: User accounts table" in meta["content"]
    assert "email: User email address" in meta["content"]

    row_doc = render_row(
        schema="public",
        table="users",
        primary_key="id",
        row={"id": 101, "email": "user@example.com", "is_deleted": False},
        soft_delete_col="is_deleted",
    )
    assert row_doc["id"] == "public.users:101"
    assert row_doc["title"] == "users #101"
    assert "email: user@example.com" in row_doc["content"]
    assert row_doc["_deleted"] is False


def test_redshift_source_execution(monkeypatch):
    """Test full redshift_source dlt resource extraction workflow."""
    mock_boto = mock.MagicMock()
    client = RedshiftClient(database="dev", workgroup_name="wg", boto3_client=mock_boto)

    mock_boto.execute_statement.side_effect = [{"Id": "stmt-data"}, {"Id": "stmt-pk"}]
    mock_boto.describe_statement.return_value = {"Status": "FINISHED"}

    data_page = {
        "ColumnMetadata": [{"name": "id"}, {"name": "updated_at"}, {"name": "name"}],
        "Records": [
            [{"longValue": 1}, {"stringValue": "2026-10-08T10:00:00Z"}, {"stringValue": "Alice"}],
            [{"longValue": 2}, {"stringValue": "2026-10-08T11:00:00Z"}, {"stringValue": "Bob"}],
        ],
    }
    pk_page = {
        "ColumnMetadata": [{"name": "id"}],
        "Records": [[{"longValue": 1}], [{"longValue": 2}]],
    }

    mock_boto.get_statement_result.side_effect = [data_page, pk_page]

    # Mock DLT state
    dlt_state = {}
    monkeypatch.setattr("dlt.current.resource_state", lambda: dlt_state)

    source = redshift_source(
        client=client,
        table_name="users",
        primary_key="id",
        schema="public",
        timestamp_column="updated_at",
    )

    items = list(source)

    # 1 Metadata document + 2 Data rows = 3 items
    assert len(items) == 3
    assert items[0]["id"] == "meta:public.users"
    assert items[1]["id"] == "public.users:1"
    assert items[2]["id"] == "public.users:2"

    # State updated on completion
    assert dlt_state["metadata_emitted"] is True
    assert dlt_state["last_timestamp"] == "2026-10-08T11:00:00Z"
    assert dlt_state["last_pk"] == "2"
    assert dlt_state["known_pks"] == ["1", "2"]


def test_deletion_reconciliation_tombstones(monkeypatch):
    """Test primary key inventory reconciliation yielding tombstones for deleted rows."""
    mock_boto = mock.MagicMock()
    client = RedshiftClient(database="dev", workgroup_name="wg", boto3_client=mock_boto)

    mock_boto.execute_statement.side_effect = [{"Id": "stmt-data"}, {"Id": "stmt-pk"}]
    mock_boto.describe_statement.return_value = {"Status": "FINISHED"}

    data_page = {
        "ColumnMetadata": [{"name": "id"}, {"name": "name"}],
        "Records": [[{"longValue": 1}, {"stringValue": "Alice"}]],
    }
    # PK 2 is missing from Redshift now
    pk_page = {
        "ColumnMetadata": [{"name": "id"}],
        "Records": [[{"longValue": 1}]],
    }
    mock_boto.get_statement_result.side_effect = [data_page, pk_page]

    # State has known_pks = ["1", "2"]
    dlt_state = {"metadata_emitted": True, "known_pks": ["1", "2"]}
    monkeypatch.setattr("dlt.current.resource_state", lambda: dlt_state)

    source = redshift_source(
        client=client,
        table_name="users",
        primary_key="id",
        schema="public",
        reconcile_deletions=True,
    )

    items = list(source)

    # 1 Data row + 1 Deletion tombstone for PK 2
    assert len(items) == 2
    assert items[0]["id"] == "public.users:1"
    assert items[0]["_deleted"] is False

    assert items[1]["id"] == "public.users:2"
    assert items[1]["_deleted"] is True

    # State updated to reflect only active known_pks ["1"]
    assert dlt_state["known_pks"] == ["1"]


def test_incomplete_read_safety(monkeypatch):
    """If an API call fails mid-stream, cursor and known_pks are not mutated."""
    mock_boto = mock.MagicMock()
    client = RedshiftClient(database="dev", workgroup_name="wg", boto3_client=mock_boto)

    mock_boto.execute_statement.return_value = {"Id": "stmt-fail"}
    mock_boto.describe_statement.return_value = {"Status": "FAILED", "Error": "Connection error"}

    dlt_state = {
        "metadata_emitted": True,
        "last_timestamp": "2026-10-08T09:00:00Z",
        "known_pks": ["10"],
    }
    monkeypatch.setattr("dlt.current.resource_state", lambda: dlt_state)

    source = redshift_source(
        client=client,
        table_name="users",
        primary_key="id",
        schema="public",
        timestamp_column="updated_at",
    )

    with pytest.raises((RedshiftAPIError, dlt.extract.exceptions.ResourceExtractionError)):
        list(source)

    # State remains untouched
    assert dlt_state["last_timestamp"] == "2026-10-08T09:00:00Z"
    assert dlt_state["known_pks"] == ["10"]
