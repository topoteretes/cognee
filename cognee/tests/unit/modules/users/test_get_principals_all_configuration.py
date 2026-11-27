"""``get_principals_all_configuration`` reads several owners' records in one query."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.relational import Base
from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration

# The methods package rebinds get_principal_configuration to the function,
# shadowing the submodule; import the submodule explicitly to patch its engine.
module = importlib.import_module("cognee.modules.users.methods.get_principal_configuration")


@pytest.mark.asyncio
async def test_records_are_grouped_by_owner_and_absent_owners_are_omitted():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn, tables=[PrincipalConfiguration.__table__]
            )
        )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    first, second, third, stranger = uuid4(), uuid4(), uuid4(), uuid4()
    try:
        async with maker() as session:
            session.add_all(
                [
                    PrincipalConfiguration(owner_id=first, name="a", configuration={"n": 1}),
                    PrincipalConfiguration(owner_id=first, name="b", configuration={"n": 2}),
                    PrincipalConfiguration(owner_id=second, name="a", configuration={"n": 3}),
                    PrincipalConfiguration(owner_id=stranger, name="a", configuration={"n": 4}),
                ]
            )
            await session.commit()

        fake_engine = SimpleNamespace(get_async_session=maker)
        with patch.object(module, "get_relational_engine", return_value=fake_engine):
            grouped = await module.get_principals_all_configuration([first, second, third])
            single = await module.get_principal_all_configuration(second)
            none = await module.get_principals_all_configuration([])
    finally:
        await engine.dispose()

    assert set(grouped) == {first, second}
    assert sorted(record["name"] for record in grouped[first]) == ["a", "b"]
    assert [record["configuration"] for record in grouped[second]] == [{"n": 3}]
    assert single == grouped[second]
    assert none == {}
