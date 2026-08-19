"""Unit tests for cognee.modules.ingestion.identify_many.

Asserts that identify_many() and identify() agree on which Data row to return
for a given (hash, dataset, owner, tenant) scope, and that the chunking logic
works correctly for inputs that exceed SQLite's bind-parameter limit.

Contract under test:
  - same hash + dataset + owner + tenant  → same data_id (matches identify())
  - different owner                        → no result (owner-scoped)
  - different tenant                       → no result (tenant-scoped)
  - hash not in DB                         → absent from result dict
  - empty input                            → empty dict, no DB call
  - large input (> _CHUNK_SIZE hashes)     → chunked correctly, all hits returned
"""

import importlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data
from cognee.modules.ingestion.identify_many import _CHUNK_SIZE, identify_many


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Import the module objects explicitly and patch through them. Dotted-string
# targets are unreliable here: `cognee.modules.ingestion.__init__` re-exports the
# identify/identify_many *functions*, which shadow the same-named submodules on
# the package, so patch("...ingestion.identify_many.get_relational_engine") can
# resolve to the function and raise AttributeError depending on import order.
_IDENTIFY_MANY_MODULE = importlib.import_module("cognee.modules.ingestion.identify_many")
_IDENTIFY_MODULE = importlib.import_module("cognee.modules.ingestion.identify")


async def _make_engine(rows: list[dict]) -> tuple[SQLAlchemyAdapter, str]:
    """Spin up a throwaway SQLite engine and seed it with the given Data rows."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp.name}")
    await engine.create_database()

    async with engine.get_async_session() as session:
        for row in rows:
            session.add(Data(**row))
        await session.commit()

    return engine, tmp.name


def _user(tenant_id=None):
    return SimpleNamespace(id=uuid4(), tenant_id=tenant_id)


def _row(*, dataset_id, owner_id, content_hash, tenant_id=None, name="doc.txt"):
    return dict(
        id=uuid4(),
        dataset_id=dataset_id,
        owner_id=owner_id,
        tenant_id=tenant_id,
        name=name,
        content_hash=content_hash,
        raw_data_location=f"file:///tmp/{name}",
        pipeline_status={},
        token_count=-1,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_returns_empty_dict():
    """identify_many([]) must return {} without touching the DB."""
    # No engine needed — the early-return guard fires before any DB call.
    result = await identify_many([], _user(), uuid4())
    assert result == {}


@pytest.mark.asyncio
async def test_hit_returns_correct_data_id():
    """A hash present in the DB for the right owner/dataset must be returned."""
    user = _user()
    dataset_id = uuid4()
    data_id = uuid4()
    content_hash = "hash-abc"

    row = _row(dataset_id=dataset_id, owner_id=user.id, content_hash=content_hash)
    row["id"] = data_id  # pin the id so we can assert on it

    engine, db_path = await _make_engine([row])
    try:
        with patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine):
            result = await identify_many([content_hash], user, dataset_id)

        assert result == {content_hash: data_id}, (
            "identify_many must return the seeded data_id for the matching hash"
        )
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_miss_for_unknown_hash():
    """A hash not in the DB must be absent from the result (not mapped to None)."""
    user = _user()
    dataset_id = uuid4()

    engine, db_path = await _make_engine([])
    try:
        with patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine):
            result = await identify_many(["no-such-hash"], user, dataset_id)

        assert result == {}, "unknown hash must not appear in the result dict"
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_different_owner_gets_no_result():
    """A row owned by user A must not be returned when queried as user B.

    This is the same owner-scoping contract enforced by identify().
    """
    owner = _user()
    stranger = _user()
    dataset_id = uuid4()
    content_hash = "hash-owner-scoped"

    engine, db_path = await _make_engine(
        [_row(dataset_id=dataset_id, owner_id=owner.id, content_hash=content_hash)]
    )
    try:
        with patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine):
            result = await identify_many([content_hash], stranger, dataset_id)

        assert result == {}, "another owner's row must not be returned"
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_different_tenant_gets_no_result():
    """A row with tenant_id=T must not be returned when the user has tenant_id=None.

    Same owner_id, different tenant — must miss.
    """
    user_with_tenant = _user(tenant_id=uuid4())
    dataset_id = uuid4()
    content_hash = "hash-tenant-scoped"

    engine, db_path = await _make_engine(
        [
            _row(
                dataset_id=dataset_id,
                owner_id=user_with_tenant.id,
                content_hash=content_hash,
                tenant_id=user_with_tenant.tenant_id,
            )
        ]
    )
    try:
        with patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine):
            user_no_tenant = SimpleNamespace(id=user_with_tenant.id, tenant_id=None)
            result = await identify_many([content_hash], user_no_tenant, dataset_id)

        assert result == {}, "row with a tenant must not be returned for a tenant-less user"
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_identify_many_agrees_with_identify():
    """identify_many and identify must return the same data_id for the same inputs.

    This is the core contract: the two functions share the same four filter
    predicates, so they must always agree on which row wins.
    """
    from cognee.modules.ingestion.identify import identify

    user = _user()
    dataset_id = uuid4()
    data_id = uuid4()
    content_hash = "hash-contract-check"

    row = _row(dataset_id=dataset_id, owner_id=user.id, content_hash=content_hash)
    row["id"] = data_id

    engine, db_path = await _make_engine([row])
    try:
        # Build a minimal IngestionData whose get_identifier() returns our hash.
        classified = SimpleNamespace(get_identifier=lambda: content_hash)

        with (
            patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine),
            patch.object(_IDENTIFY_MODULE, "get_relational_engine", return_value=engine),
        ):
            many_result = await identify_many([content_hash], user, dataset_id)
            single_result = await identify(classified, user, dataset_id)

        assert many_result.get(content_hash) == single_result, (
            "identify_many and identify must return the same data_id for the same inputs"
        )
        assert single_result == data_id
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_chunking_handles_large_input():
    """A list of hashes larger than _CHUNK_SIZE must be chunked and all hits returned.

    Seeds two rows with known hashes and queries with 2 * _CHUNK_SIZE + 1 hashes
    (spanning three chunks). Both seeded hashes must appear in the result; the
    rest must be absent.
    """
    user = _user()
    dataset_id = uuid4()

    hash_a = "hash-chunk-a"
    hash_b = "hash-chunk-b"
    id_a = uuid4()
    id_b = uuid4()

    row_a = _row(dataset_id=dataset_id, owner_id=user.id, content_hash=hash_a, name="a.txt")
    row_a["id"] = id_a
    row_b = _row(dataset_id=dataset_id, owner_id=user.id, content_hash=hash_b, name="b.txt")
    row_b["id"] = id_b

    engine, db_path = await _make_engine([row_a, row_b])
    try:
        # Build a list that spans three chunks: place the two real hashes at
        # positions that land in different chunks so both chunking paths are hit.
        filler = [f"no-such-hash-{i}" for i in range(2 * _CHUNK_SIZE + 1)]
        filler[0] = hash_a
        filler[_CHUNK_SIZE] = hash_b  # guaranteed to be in the second chunk

        with patch.object(_IDENTIFY_MANY_MODULE, "get_relational_engine", return_value=engine):
            result = await identify_many(filler, user, dataset_id)

        assert result.get(hash_a) == id_a, "hash_a must be found in the first chunk"
        assert result.get(hash_b) == id_b, "hash_b must be found in the second chunk"
        # All filler hashes must be absent (not mapped to None or anything else).
        filler_hits = {h: v for h, v in result.items() if h not in (hash_a, hash_b)}
        assert filler_hits == {}, f"unexpected hits for filler hashes: {filler_hits}"
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)
