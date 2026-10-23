"""update_provider_metadata against a real database: concurrent changes are all kept."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations import credentials
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


@pytest_asyncio.fixture
async def storage(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'credentials.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(IntegrationCredential.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        credentials, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    async with sessions() as db:
        db.add(
            IntegrationCredential(
                id=uuid4(),
                user_id=uuid4(),
                provider="github",
                provider_account_id="42",
                ciphertext=b"encrypted",
                nonce=b"nonce",
                status="active",
                provider_metadata={"other": 1},
            )
        )
        await db.commit()
    yield sessions
    await engine.dispose()


def _adding(name):
    def patch_for(stored):
        return {"names": [*stored.get("names", []), name]}

    return patch_for


@pytest.mark.asyncio
async def test_concurrent_changes_computed_from_the_stored_value_are_all_kept(storage):
    """SQLite takes no row lock, so each callable must run on what the last write left."""
    for _ in range(10):
        await credentials.update_provider_metadata("github", "42", {"names": []})
        await asyncio.gather(
            *(
                credentials.update_provider_metadata("github", "42", _adding(f"repo{index}"))
                for index in range(6)
            )
        )
        result = await credentials.update_provider_metadata("github", "42", {})
        assert sorted(result.provider_metadata["names"]) == [f"repo{index}" for index in range(6)]
        assert result.provider_metadata["other"] == 1


@pytest.mark.asyncio
async def test_a_patch_function_receives_the_stored_metadata(storage):
    seen = []

    def patch_for(stored):
        seen.append(dict(stored))
        return {"seen": True}

    result = await credentials.update_provider_metadata("github", "42", patch_for)

    assert seen == [{"other": 1}]
    assert result.provider_metadata == {"other": 1, "seen": True}


@pytest.mark.asyncio
async def test_a_failing_patch_function_writes_nothing_and_frees_the_credential(storage):
    def fails(stored):
        raise ValueError("bad shape")

    with pytest.raises(ValueError):
        await credentials.update_provider_metadata("github", "42", fails)

    result = await credentials.update_provider_metadata("github", "42", {"after": True})
    assert result.provider_metadata == {"other": 1, "after": True}


@pytest.mark.asyncio
async def test_an_unknown_credential_is_none(storage):
    assert await credentials.update_provider_metadata("github", "nobody", {"a": 1}) is None
