from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.api.client import lifespan


@pytest.mark.asyncio
async def test_api_drains_maintenance_before_engine_cleanup():
    events = []
    drain = AsyncMock(side_effect=lambda: events.append("drain"))
    graph_clear = MagicMock(side_effect=lambda: events.append("graph"))
    vector_clear = MagicMock(side_effect=lambda: events.append("vector"))

    with (
        patch("cognee.run_migrations.run_migrations", new_callable=AsyncMock),
        patch("cognee.modules.users.methods.get_default_user", new_callable=AsyncMock),
        patch(
            "cognee.modules.cognify.recovery.recover_stale_cognify_runs_on_startup",
            new_callable=AsyncMock,
        ),
        patch(
            "cognee.infrastructure.session.session_maintenance_worker.drain_session_maintenance",
            drain,
        ),
        patch(
            "cognee.infrastructure.databases.graph.get_graph_engine._create_graph_engine.cache_clear",
            graph_clear,
        ),
        patch(
            "cognee.infrastructure.databases.vector.create_vector_engine._create_vector_engine.cache_clear",
            vector_clear,
        ),
    ):
        async with lifespan(MagicMock()):
            pass

    drain.assert_awaited_once()
    assert events == ["drain", "graph", "vector"]
