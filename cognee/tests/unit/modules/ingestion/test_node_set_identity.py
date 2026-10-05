"""The node set is part of a data item's dedup identity.

Same content under the same node set is one row; the same content under
another node set (another end user of one cognee user, another project) is a
new data item. Callers that do not know the node set keep the content-only
match. Covers the pure helpers and both scoped lookups
(``identify_data_by_hash`` and ``identify_many``) against a throwaway SQLite
engine.
"""

import importlib
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.data.models import Data
from cognee.modules.ingestion.identify import identify_data_by_hash
from cognee.modules.ingestion.identify_many import identify_many
from cognee.modules.ingestion.node_set_identity import (
    UNSCOPED,
    encode_node_set,
    node_set_matches,
    normalize_node_set,
)

_identify_many_module = importlib.import_module("cognee.modules.ingestion.identify_many")
_identify_module = importlib.import_module("cognee.modules.ingestion.identify")


# --- pure helpers -----------------------------------------------------------


def test_normalize_sorts_dedupes_and_drops_empties():
    assert normalize_node_set(["bob", "alice", "bob", "", None]) == ["alice", "bob"]
    assert normalize_node_set(("x",)) == ["x"]
    assert normalize_node_set({"b", "a"}) == ["a", "b"]


def test_normalize_treats_absent_and_empty_as_no_node_set():
    assert normalize_node_set(None) is None
    assert normalize_node_set([]) is None
    assert normalize_node_set([""]) is None
    assert normalize_node_set(UNSCOPED) is None


def test_normalize_reads_the_stored_json_encoding():
    assert normalize_node_set(json.dumps(["b", "a"])) == ["a", "b"]
    assert normalize_node_set("null") is None
    assert normalize_node_set("not json") is None
    assert normalize_node_set(json.dumps({"not": "a list"})) is None


def test_encode_is_canonical_json_or_none():
    assert encode_node_set(["b", "a", "a"]) == '["a", "b"]'
    assert encode_node_set(None) is None
    assert encode_node_set([]) is None


def test_matches_compares_as_sets_and_unscoped_matches_everything():
    assert node_set_matches('["a", "b"]', ["b", "a"])
    assert node_set_matches(None, [])
    assert not node_set_matches('["a"]', ["a", "b"])
    assert not node_set_matches(None, ["a"])
    assert not node_set_matches('["a"]', None)
    assert node_set_matches('["a"]', UNSCOPED)
    assert node_set_matches(None, UNSCOPED)


# --- scoped lookups -----------------------------------------------------------


async def _make_engine(rows):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{db_path}")
    await engine.create_database()
    async with engine.get_async_session() as session:
        for row in rows:
            session.add(Data(**row))
        await session.commit()
    return engine, db_path


def _row(*, dataset_id, owner_id, content_hash, node_set):
    return {
        "id": uuid4(),
        "dataset_id": dataset_id,
        "owner_id": owner_id,
        "tenant_id": None,
        "name": "doc.txt",
        "content_hash": content_hash,
        "raw_data_location": "file:///tmp/doc.txt",
        "pipeline_status": {},
        "token_count": -1,
        "node_set": encode_node_set(node_set),
    }


@pytest_asyncio.fixture
async def scoped_rows():
    """One content hash stored twice: once for alice, once for bob — plus an
    untagged row — all in one dataset for one cognee user."""
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    dataset_id = uuid4()
    alice = _row(dataset_id=dataset_id, owner_id=user.id, content_hash="h", node_set=["user:alice"])
    bob = _row(dataset_id=dataset_id, owner_id=user.id, content_hash="h", node_set=["user:bob"])
    untagged = _row(dataset_id=dataset_id, owner_id=user.id, content_hash="h", node_set=None)
    engine, db_path = await _make_engine([alice, bob, untagged])
    try:
        with (
            patch.object(_identify_module, "get_relational_engine", return_value=engine),
            patch.object(_identify_many_module, "get_relational_engine", return_value=engine),
        ):
            yield SimpleNamespace(
                user=user,
                dataset_id=dataset_id,
                alice=alice["id"],
                bob=bob["id"],
                untagged=untagged["id"],
            )
    finally:
        await engine.engine.dispose()
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_identify_by_hash_returns_the_row_of_the_requested_scope(scoped_rows):
    s = scoped_rows
    assert (
        await identify_data_by_hash("h", s.user, s.dataset_id, node_set=["user:alice"])
    ).id == s.alice
    assert (
        await identify_data_by_hash("h", s.user, s.dataset_id, node_set=["user:bob"])
    ).id == s.bob
    assert (await identify_data_by_hash("h", s.user, s.dataset_id, node_set=None)).id == s.untagged
    assert (await identify_data_by_hash("h", s.user, s.dataset_id, node_set=[])).id == s.untagged


@pytest.mark.asyncio
async def test_identify_by_hash_misses_for_a_scope_that_has_no_row(scoped_rows):
    """Same content, new node set: a miss, so ingestion stores it for that scope."""
    s = scoped_rows
    assert await identify_data_by_hash("h", s.user, s.dataset_id, node_set=["user:carol"]) is None
    # A superset of an existing scope is a different scope too.
    assert (
        await identify_data_by_hash("h", s.user, s.dataset_id, node_set=["user:alice", "team"])
        is None
    )


@pytest.mark.asyncio
async def test_identify_by_hash_without_a_node_set_keeps_the_content_only_match(scoped_rows):
    s = scoped_rows
    hit = await identify_data_by_hash("h", s.user, s.dataset_id)
    assert hit is not None
    assert hit.id in {s.alice, s.bob, s.untagged}


@pytest.mark.asyncio
async def test_identify_many_scopes_each_hit_by_node_set(scoped_rows):
    s = scoped_rows
    assert await identify_many(["h"], s.user, s.dataset_id, node_set=["user:alice"]) == {
        "h": s.alice
    }
    assert await identify_many(["h"], s.user, s.dataset_id, node_set=["user:bob"]) == {"h": s.bob}
    assert await identify_many(["h"], s.user, s.dataset_id, node_set=None) == {"h": s.untagged}
    assert await identify_many(["h"], s.user, s.dataset_id, node_set=["user:carol"]) == {}
    unscoped = await identify_many(["h"], s.user, s.dataset_id)
    assert set(unscoped) == {"h"} and unscoped["h"] in {s.alice, s.bob, s.untagged}
