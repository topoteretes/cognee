import importlib
import os
import tempfile
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data, Dataset, DatasetData
from cognee.modules.users.models import ACL, Permission
from cognee.modules.users.models.Principal import Principal

gdifu = importlib.import_module(
    "cognee.modules.users.permissions.methods.get_document_ids_for_user"
)
gdd = importlib.import_module("cognee.modules.data.methods.get_dataset_data")
get_document_ids_for_user = gdifu.get_document_ids_for_user

OWNER_ID = uuid4()  # user B -- owns the dataset
READER_ID = uuid4()  # user A -- only has a read ACL on it
DATASET_ID = uuid4()
DATA_ID = uuid4()
DATASET_NAME = "shared_dataset"


async def _make_engine():
    """Create a throwaway SQLite-backed relational engine seeded with a dataset
    owned by user B and shared to user A with a read ACL, containing one doc."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()

    async with engine.get_async_session() as session:
        async with session.begin():
            session.add(Principal(id=OWNER_ID, type="principal"))
            session.add(Principal(id=READER_ID, type="principal"))
            session.add(Dataset(id=DATASET_ID, name=DATASET_NAME, owner_id=OWNER_ID))
            session.add(Data(id=DATA_ID, name="doc1", data_size=10))
            session.add(DatasetData(dataset_id=DATASET_ID, data_id=DATA_ID))
            perm = Permission(name="read")
            session.add(perm)
            session.add(ACL(principal_id=READER_ID, dataset_id=DATASET_ID, permission=perm))

    return engine, tmp.name


@pytest.mark.asyncio
async def test_name_filter_keeps_acl_shared_dataset_documents(monkeypatch):
    engine, db_path = await _make_engine()
    # Both the lookup and its get_dataset_data helper must use the test engine.
    monkeypatch.setattr(gdifu, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(gdd, "get_relational_engine", lambda: engine)
    try:
        unfiltered = await get_document_ids_for_user(READER_ID)
        assert [str(i) for i in unfiltered] == [str(DATA_ID)]

        # filtering by the shared dataset's name must not drop it.
        by_name = await get_document_ids_for_user(READER_ID, datasets=[DATASET_NAME])
        assert [str(i) for i in by_name] == [str(DATA_ID)], (
            "documents of an ACL-shared dataset must survive a name filter; "
            f"got {[str(i) for i in by_name]}"
        )
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
