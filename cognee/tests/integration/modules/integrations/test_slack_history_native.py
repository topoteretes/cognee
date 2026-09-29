"""Slack -> native add/update/cognify/delete with real isolated local stores.

Slack, LLM and embeddings are deterministic doubles; SQLite, Ladybug,
LanceDB, dataset authorization, document identity and graph cleanup are real.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup
from cognee.modules.integrations.slack import history
from cognee.modules.integrations.slack.history_client import SlackAPIError
from cognee.modules.integrations.slack.history_models import SlackHistoryRequest
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent


@pytest.mark.asyncio
async def test_slack_history_native_lifecycle(tmp_path, monkeypatch):
    pytest.importorskip("ladybug")
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.setenv("LLM_API_KEY", "mocked-offline")
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")
    monkeypatch.setenv("GRAPH_DATABASE_SUBPROCESS_ENABLED", "false")
    monkeypatch.setenv("VECTOR_DB_SUBPROCESS_ENABLED", "false")

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)
    cognee.config.set_graph_db_config(
        {"graph_database_provider": "ladybug", "graph_dataset_database_handler": "ladybug"}
    )
    cognee.config.set_vector_db_config(
        {"vector_db_provider": "lancedb", "vector_dataset_database_handler": "lancedb"}
    )
    cognee.config.set_relational_db_config(
        {"db_provider": "sqlite", "db_path": str(tmp_path / "db")}
    )
    cognee.config.set_migration_db_config(
        {"migration_db_provider": "sqlite", "migration_db_path": str(tmp_path / "db")}
    )
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.set_vector_db_url(str(tmp_path / "system" / "databases" / "cognee.lancedb"))
    assert str(tmp_path) in str(get_relational_engine().engine.url)
    await setup()
    user = await get_default_user()
    dataset = await create_authorized_dataset("slack_history_integration", user)

    async def llm_output(text_input, system_prompt, response_model, **kwargs):
        if response_model == SummarizedContent:
            return SummarizedContent(summary="Slack decision", description="")
        if response_model == KnowledgeGraph:
            name = "Tuesday" if "Tuesday" in text_input else "Monday"
            return KnowledgeGraph(
                nodes=[Node(id=name, name=name, type="Decision", description=name)], edges=[]
            )
        return response_model()

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", AsyncMock(side_effect=llm_output))
    credential = SimpleNamespace(
        provider_account_id="T1",
        id=uuid4(),
        status="active",
        user_id=user.id,
        provider_metadata={"installed_by_slack_user_id": "U1"},
    )
    monkeypatch.setattr(history, "get_by_team", AsyncMock(return_value=credential))
    monkeypatch.setattr(
        history, "decrypt_token_payload", lambda credential: {"access_token": "fake"}
    )
    ts = "1701000000.000001"
    source = SimpleNamespace(
        call=AsyncMock(return_value={"team_id": "T1", "url": "https://acme.slack.com/"}),
        authorize_channel=AsyncMock(return_value={"id": "C1", "name": "general"}),
        history=AsyncMock(return_value=[{"ts": ts}]),
        thread=AsyncMock(return_value=[{"ts": ts, "user": "U1", "text": "Ship Monday"}]),
    )

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return source

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(history, "SlackHistoryClient", Client)
    selection = SlackHistoryRequest(
        dataset_id=dataset.id,
        channel_ids=["C1"],
        oldest=datetime.fromtimestamp(1700000000, timezone.utc),
    )

    async def graph_text():
        async with set_database_global_context_variables(dataset.id, user.id):
            nodes, _ = await (await get_graph_engine()).get_graph_data()
            return " ".join(str(props.get("text", "")) for _, props in nodes)

    first = await history.import_slack_history("T1", selection, user=user)
    assert first.added == 1
    rows = await history._stored_documents(dataset.id, "T1")
    first_id = rows[("C1", ts)].id
    assert "Ship Monday" in await graph_text()
    second = await history.import_slack_history("T1", selection, user=user)
    assert second.unchanged == 1
    source.thread.return_value[0]["text"] = "Ship Tuesday"
    edited = await history.import_slack_history("T1", selection, user=user)
    assert edited.updated == 1
    rows = await history._stored_documents(dataset.id, "T1")
    assert len(rows) == 1 and rows[("C1", ts)].id == first_id
    text = await graph_text()
    assert "Ship Tuesday" in text and "Ship Monday" not in text
    source.history.return_value = []
    await history.import_slack_history("T1", selection, user=user)
    assert ("C1", ts) in await history._stored_documents(dataset.id, "T1")
    source.thread.side_effect = SlackAPIError("replies", "thread_not_found")
    deleted = await history.import_slack_history("T1", selection, user=user, reconcile=True)
    assert deleted.deleted == 1
    assert await history._stored_documents(dataset.id, "T1") == {}
    assert "Ship Tuesday" not in await graph_text()
