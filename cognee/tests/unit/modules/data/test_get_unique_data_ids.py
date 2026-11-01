"""``get_unique_data_ids`` derives the id ``get_unique_data_id`` derives, for each identifier.

The id is a UUID5 of the identifier, the user and the tenant (and the dataset, when given),
except that data created before tenant-aware derivation keeps its tenant-less id. The batch
variant probes for those rows once per batch of ids, so it has to keep the same answer on both
sides of every batch boundary (SDK-1169).
"""

import importlib
from types import SimpleNamespace
from uuid import NAMESPACE_OID, UUID, uuid5

import pytest
import pytest_asyncio
from sqlalchemy import insert

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.methods.get_existing_data_ids import (
    DATA_ID_BATCH_SIZE,
    get_existing_data_ids,
)
from cognee.modules.data.models import Data

unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_data_id")

USER = SimpleNamespace(id=UUID(int=1), tenant_id=UUID(int=2))
DATASET = UUID(int=3)
OTHER_DATASET = UUID(int=4)

# Two full batches and one id over, so a probe row can sit in each batch.
IDENTIFIER_COUNT = 2 * DATA_ID_BATCH_SIZE + 1
# Every fifth identifier has a row under its tenant-less id.
PRE_TENANT_EVERY = 5


def _modern_id(identifier):
    return uuid5(NAMESPACE_OID, f"{identifier}{USER.id}{USER.tenant_id}")


def _pre_tenant_id(identifier):
    return uuid5(NAMESPACE_OID, f"{identifier}{USER.id}")


async def _store(engine, ids, dataset_id):
    values = [
        {
            "id": data_id,
            "dataset_id": dataset_id,
            "name": "row",
            "content_hash": str(data_id),
            "raw_data_location": "file:///row.txt",
            "owner_id": USER.id,
            "pipeline_status": {},
            "token_count": -1,
        }
        for data_id in ids
    ]
    async with engine.get_async_session() as session:
        await session.execute(insert(Data.__table__), values)
        await session.commit()


@pytest_asyncio.fixture
async def engine(monkeypatch, tmp_path):
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'relational.db'}")
    await engine.create_database()
    monkeypatch.setattr(unique_id_module, "get_relational_engine", lambda: engine)
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset_id", [None, DATASET])
# An iterator is read once, so it must give the same ids with and without a dataset.
@pytest.mark.parametrize("as_input", [list, iter])
async def test_each_identifier_gets_its_modern_or_its_tenantless_id(engine, dataset_id, as_input):
    scope = "" if dataset_id is None else f"{dataset_id}:"
    identifiers = [f"row-{index}" for index in range(IDENTIFIER_COUNT)]
    tenantless = identifiers[::PRE_TENANT_EVERY]
    # The probe is not filtered by dataset: a tenant-less row anywhere keeps the id.
    await _store(engine, [_pre_tenant_id(f"{scope}{i}") for i in tenantless], OTHER_DATASET)

    ids = await unique_id_module.get_unique_data_ids(as_input(identifiers), USER, dataset_id)

    assert ids == [
        _pre_tenant_id(f"{scope}{i}") if i in tenantless else _modern_id(f"{scope}{i}")
        for i in identifiers
    ]


@pytest.mark.asyncio
async def test_a_single_identifier_still_gets_the_id_it_always_got(engine):
    # Pinned from the one-identifier implementation this one replaced.
    identifier = "dlt:contacts:0:hash0"
    expected = UUID("d4c00d29-2597-5877-901c-f1f25537f8df")

    assert await unique_id_module.get_unique_data_id(identifier, USER, DATASET) == expected


@pytest.mark.asyncio
async def test_existing_ids_are_found_in_every_batch_and_only_in_the_given_dataset(engine):
    ids = [UUID(int=100 + index) for index in range(IDENTIFIER_COUNT)]
    here, elsewhere = ids[::7], ids[3::7]
    await _store(engine, here, DATASET)
    await _store(engine, elsewhere, OTHER_DATASET)

    async with engine.get_async_session() as session:
        in_dataset = await get_existing_data_ids(session, ids, DATASET)
        anywhere = await get_existing_data_ids(session, ids)

    assert in_dataset == set(here)
    assert anywhere == set(here) | set(elsewhere)
