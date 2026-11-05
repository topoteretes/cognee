"""Retirement compares bare resource names even when table_name is schema-qualified.

On Postgres, dlt stores schema-qualified table names ("dataset.table") in
system_metadata, while sync callers pass bare resource names as ``retained``.
retire_resources must normalize the stored name to its last path component
(the dlt resource name) before the set difference, or every table looks
deselected and the replace run lands on a wrongly named resource.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations import ingestion


def _remember_mock(monkeypatch):
    remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
    monkeypatch.setattr(import_module("cognee.api.v1.remember.remember"), "remember", remember)
    return remember


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retained, expected_retired",
    [
        ({"folder_a"}, {"folder_b"}),  # folder_b deselected, emptied by bare name
        ({"folder_a", "folder_b"}, set()),  # both still selected: nothing retired
    ],
)
async def test_qualified_table_names_are_normalized_before_retirement(
    monkeypatch, retained, expected_retired
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    owner = uuid4()
    dataset = uuid4()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Dataset.__table__.create)
            await conn.run_sync(Data.__table__.create)
        async with sessions() as db:
            db.add(Dataset(id=dataset, name="drive", owner_id=owner))
            # Postgres-style schema-qualified table names in system_metadata.
            for table in ("drive.folder_a", "drive.folder_b"):
                db.add(
                    Data(
                        dataset_id=dataset,
                        system_metadata={
                            "source": "google_drive",
                            "table_name": table,
                        },
                    )
                )
            await db.commit()
        monkeypatch.setattr(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            lambda: SimpleNamespace(get_async_session=sessions),
        )
        monkeypatch.setattr(ingestion, "require_active_credential", AsyncMock())
        monkeypatch.setattr(
            "cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")
        )
        remember = _remember_mock(monkeypatch)

        await ingestion.retire_resources(
            "google_drive", SimpleNamespace(user_id=owner), "drive", retained
        )

        assert {call.args[0].name for call in remember.await_args_list} == expected_retired
        # The replace must target the bare dlt resource, not "drive.<table>".
        for call in remember.await_args_list:
            assert "." not in call.args[0].name
            assert call.kwargs["write_disposition"] == "replace"
    finally:
        await engine.dispose()
