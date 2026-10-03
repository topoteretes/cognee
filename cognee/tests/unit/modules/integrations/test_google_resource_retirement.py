"""Retirement uses only system-owned tables in the account owner's dataset."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations.google import ingestion


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [{"folder_a"}, set()])
async def test_deselection_only_retires_the_owners_selected_provider_tables(monkeypatch, retained):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    owner, other_owner = uuid4(), uuid4()
    dataset, other_dataset = uuid4(), uuid4()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Dataset.__table__.create)
            await conn.run_sync(Data.__table__.create)
        async with sessions() as db:
            db.add_all(
                [
                    Dataset(id=dataset, name="drive", owner_id=owner),
                    Dataset(id=other_dataset, name="drive", owner_id=other_owner),
                ]
            )
            for dataset_id, provider, table in [
                (dataset, "google_drive", "folder_a"),
                (dataset, "google_drive", "folder_b"),
                (dataset, "google_drive", None),  # unknown legacy scope: not safe to remove
                (dataset, "gmail", "mail"),
                (other_dataset, "google_drive", "other_owner"),
            ]:
                db.add(
                    Data(
                        dataset_id=dataset_id,
                        system_metadata={"source": provider, "table_name": table},
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
        remember = AsyncMock(return_value=SimpleNamespace(status="completed"))
        monkeypatch.setattr(import_module("cognee.api.v1.remember.remember"), "remember", remember)
        await ingestion.retire_resources(
            "google_drive", SimpleNamespace(user_id=owner), "drive", retained
        )
        assert {call.args[0].name for call in remember.await_args_list} == (
            {"folder_a", "folder_b"} - retained
        )
        for call in remember.await_args_list:
            assert call.kwargs["write_disposition"] == "replace"
            assert call.kwargs["user"] == "owner"
    finally:
        await engine.dispose()
