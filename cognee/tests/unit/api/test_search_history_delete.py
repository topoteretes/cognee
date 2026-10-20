"""DELETE /api/v1/search/history: a user deletes their own search history, and only theirs.

Runs the real route and the real delete over a real SQLite database holding
the ``queries`` and ``results`` tables; only the authenticated user is
overridden. The GET history route reads the same database, so the test checks
the result the way a client sees it.
"""

import importlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
    SQLAlchemyAdapter,
)
from cognee.modules.search.models.Query import Query
from cognee.modules.search.models.Result import Result

# The module object: the routers package re-exports the function under the
# module's own name, so a plain import would hand back the function.
router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
delete_history_module = importlib.import_module("cognee.modules.search.operations.delete_history")
get_history_module = importlib.import_module("cognee.modules.search.operations.get_history")


@pytest.fixture
def relational():
    """A real SQLite DB with the history tables, wired into the history read and delete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmpdir}/relational.db")

        async def create_tables():
            async with adapter.engine.begin() as connection:
                await connection.run_sync(
                    Query.metadata.create_all, tables=[Query.__table__, Result.__table__]
                )

        with (
            patch.object(delete_history_module, "get_relational_engine", return_value=adapter),
            patch.object(get_history_module, "get_relational_engine", return_value=adapter),
            patch.object(router_module, "send_telemetry"),
        ):
            client_holder = SimpleNamespace(adapter=adapter, create_tables=create_tables)
            yield client_holder


def _client(user_id) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_search_router(), prefix="/api/v1/search")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )
    return TestClient(app)


async def _seed(adapter, user_id, n):
    async with adapter.get_async_session() as session:
        for i in range(n):
            query = Query(text=f"question {i}", query_type="CHUNKS", user_id=user_id)
            session.add(query)
            await session.flush()
            session.add(Result(value=f"answer {i}", query_id=query.id, user_id=user_id))
        await session.commit()


def test_user_deletes_only_their_own_history(relational):
    alice, bob = uuid4(), uuid4()
    alice_client, bob_client = _client(alice), _client(bob)

    async def seed():
        await relational.create_tables()
        await _seed(relational.adapter, alice, 2)
        await _seed(relational.adapter, bob, 3)

    with alice_client as client:
        client.portal.call(seed)

        assert len(client.get("/api/v1/search").json()) == 4

        response = client.delete("/api/v1/search/history")
        assert response.status_code == 200
        assert response.json() == {"queriesDeleted": 2, "resultsDeleted": 2}
        assert client.get("/api/v1/search").json() == []

        # Deleting an empty history is not an error.
        assert client.delete("/api/v1/search/history").json() == {
            "queriesDeleted": 0,
            "resultsDeleted": 0,
        }

    # Bob's history is untouched by Alice's delete.
    with bob_client as client:
        assert len(client.get("/api/v1/search").json()) == 6


def test_route_takes_no_user_parameter():
    """Nobody can name another user: the route has no path, query or body input at all."""
    route = next(
        r
        for r in router_module.get_search_router().routes
        if r.path == "/history" and "DELETE" in r.methods
    )
    assert route.dependant.path_params == []
    assert route.dependant.query_params == []
    assert route.dependant.body_params == []
