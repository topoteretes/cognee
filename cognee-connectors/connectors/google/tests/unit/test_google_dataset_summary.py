"""Moved from cognee/tests/unit/modules/integrations/test_google_dataset_summary.py (cognee dev, 2026-10-26).

Connection summaries count persisted data and stay scoped to its owner.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee_connector_google.cognee_host.ingestion import dataset_summary

# The three service packages are separate distributions; without them (a run of this
# package's tests alone) the cross-provider cases below are skipped, not failed.
gmail_adapter = pytest.importorskip(
    "cognee_connector_gmail.cognee_host.adapter", reason="needs cognee-connector-gmail"
)
calendar_adapter = pytest.importorskip(
    "cognee_connector_google_calendar.cognee_host.adapter",
    reason="needs cognee-connector-google-calendar",
)
drive_adapter = pytest.importorskip(
    "cognee_connector_google_drive.cognee_host.adapter",
    reason="needs cognee-connector-google-drive",
)
GoogleGmailIntegration = gmail_adapter.GoogleGmailIntegration
GoogleCalendarIntegration = calendar_adapter.GoogleCalendarIntegration
GoogleDriveIntegration = drive_adapter.GoogleDriveIntegration


@pytest.mark.asyncio
async def test_summary_counts_only_the_connected_owners_dataset():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    owner, other, dataset, other_dataset = [uuid4() for _ in range(4)]
    async with engine.begin() as db:
        await db.execute(text("CREATE TABLE datasets (id TEXT, name TEXT, owner_id TEXT)"))
        await db.execute(text("CREATE TABLE data (id TEXT, dataset_id TEXT)"))
        await db.execute(
            text("INSERT INTO datasets VALUES (:id, 'drive', :owner)"),
            [
                {"id": dataset.hex, "owner": owner.hex},
                {"id": other_dataset.hex, "owner": other.hex},
            ],
        )
        await db.execute(
            text("INSERT INTO data VALUES (:id, :dataset)"),
            [
                {"id": uuid4().hex, "dataset": dataset.hex},
                {"id": uuid4().hex, "dataset": other_dataset.hex},
                {"id": uuid4().hex, "dataset": other_dataset.hex},
            ],
        )
    sessions = async_sessionmaker(engine)
    try:
        with patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=SimpleNamespace(get_async_session=sessions),
        ):
            assert await dataset_summary(SimpleNamespace(user_id=owner), "drive") == (
                str(dataset),
                1,
            )
            assert await dataset_summary(SimpleNamespace(user_id=uuid4()), "drive") == (None, 0)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "integration,module,function",
    [
        (GoogleGmailIntegration(), "gmail", "sync_gmail"),
        (GoogleDriveIntegration(), "google_drive", "sync_drive"),
        (GoogleCalendarIntegration(), "google_calendar", "sync_google_calendar"),
    ],
)
async def test_reconnect_does_not_implicitly_start_an_import(integration, module, function):
    with patch(
        f"cognee_connector_{module}.cognee_host.sync.{function}", new_callable=AsyncMock
    ) as sync:
        await integration.on_installed(SimpleNamespace())
        sync.assert_not_awaited()
        await integration.sync_now(SimpleNamespace())
        sync.assert_awaited_once()
