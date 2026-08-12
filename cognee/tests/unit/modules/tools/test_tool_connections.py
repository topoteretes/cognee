import base64
import os
from uuid import uuid4

import pytest

import cognee.modules.tools.connections as connections_mod
from cognee.modules.tools.config import ToolsConfig
from cognee.modules.tools.errors import ToolConnectionNotFoundError, ToolError
from cognee.modules.tools.connections import (
    delete_tool_connection,
    get_tool_connection,
    infer_provider,
    list_tool_connections,
    register_tool_connection,
)


@pytest.fixture
def crypto_key(monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(os.urandom(32)).decode())
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEYS", raising=False)


@pytest.fixture
def relational_engine(monkeypatch, tmp_path):
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    adapter = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path}/relational.db")
    monkeypatch.setattr(connections_mod, "get_relational_engine", lambda: adapter)
    monkeypatch.setattr(connections_mod, "_table_ready", False)
    return adapter


@pytest.fixture
def no_env_connections(monkeypatch):
    monkeypatch.setattr(connections_mod, "get_tools_config", lambda: ToolsConfig(_env_file=None))


def test_infer_provider():
    assert infer_provider("postgresql://u:p@h/db") == "postgres"
    assert infer_provider("postgres://u:p@h/db") == "postgres"
    assert infer_provider("sqlite:///a.db") == "sqlite"
    with pytest.raises(ToolError):
        infer_provider("mysql://u:p@h/db")
    with pytest.raises(ToolError):
        infer_provider("not-a-dsn")


@pytest.mark.asyncio
async def test_register_and_resolve_round_trip(crypto_key, relational_engine, no_env_connections):
    user_id = uuid4()
    public = await register_tool_connection(
        user_id,
        "analytics",
        "sqlite:///warehouse.db",
        allowed_tables=["orders"],
        max_rows=42,
        description="test db",
    )

    # Registration output never carries the DSN.
    assert "connection_string" not in public
    assert public["provider"] == "sqlite"
    assert public["max_rows"] == 42

    resolved = await get_tool_connection(user_id, "analytics")
    assert resolved["connection_string"] == "sqlite:///warehouse.db"
    assert resolved["allowed_tables"] == ["orders"]


@pytest.mark.asyncio
async def test_listing_never_returns_secrets(crypto_key, relational_engine, no_env_connections):
    user_id = uuid4()
    await register_tool_connection(user_id, "analytics", "sqlite:///secret-path.db")

    listed = await list_tool_connections(user_id)

    assert [connection["name"] for connection in listed] == ["analytics"]
    assert "secret-path" not in str(listed)


@pytest.mark.asyncio
async def test_ownership_isolation_fails_closed(crypto_key, relational_engine, no_env_connections):
    owner, other = uuid4(), uuid4()
    await register_tool_connection(owner, "analytics", "sqlite:///owner.db")

    with pytest.raises(ToolConnectionNotFoundError):
        await get_tool_connection(other, "analytics")
    assert await list_tool_connections(other) == []


@pytest.mark.asyncio
async def test_reregistration_replaces(crypto_key, relational_engine, no_env_connections):
    user_id = uuid4()
    await register_tool_connection(user_id, "analytics", "sqlite:///old.db")
    await register_tool_connection(user_id, "analytics", "sqlite:///new.db")

    resolved = await get_tool_connection(user_id, "analytics")
    assert resolved["connection_string"] == "sqlite:///new.db"
    assert len(await list_tool_connections(user_id)) == 1


@pytest.mark.asyncio
async def test_delete(crypto_key, relational_engine, no_env_connections):
    user_id = uuid4()
    await register_tool_connection(user_id, "analytics", "sqlite:///a.db")

    assert await delete_tool_connection(user_id, "analytics") is True
    assert await delete_tool_connection(user_id, "analytics") is False
    with pytest.raises(ToolConnectionNotFoundError):
        await get_tool_connection(user_id, "analytics")


@pytest.mark.asyncio
async def test_missing_crypto_key_fails_registration(
    monkeypatch, relational_engine, no_env_connections
):
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEYS", raising=False)
    monkeypatch.delenv("INTEGRATION_CREDENTIALS_KEY", raising=False)

    with pytest.raises(RuntimeError, match="INTEGRATION_CREDENTIALS"):
        await register_tool_connection(uuid4(), "analytics", "sqlite:///a.db")


@pytest.mark.asyncio
async def test_env_connections_visible_to_everyone(crypto_key, relational_engine, monkeypatch):
    config = ToolsConfig(
        _env_file=None,
        tool_sql_connections='{"shared": {"connection_string": "sqlite:///shared.db", "max_rows": 9}}',
    )
    monkeypatch.setattr(connections_mod, "get_tools_config", lambda: config)

    anyone = uuid4()
    listed = await list_tool_connections(anyone)
    assert listed[0]["name"] == "shared"
    assert listed[0]["origin"] == "env"
    assert "shared.db" not in str(listed)

    resolved = await get_tool_connection(anyone, "shared")
    assert resolved["connection_string"] == "sqlite:///shared.db"
    assert resolved["max_rows"] == 9


@pytest.mark.asyncio
async def test_user_connection_shadows_env_connection(crypto_key, relational_engine, monkeypatch):
    config = ToolsConfig(_env_file=None, tool_sql_connections='{"analytics": "sqlite:///env.db"}')
    monkeypatch.setattr(connections_mod, "get_tools_config", lambda: config)

    user_id = uuid4()
    await register_tool_connection(user_id, "analytics", "sqlite:///mine.db")

    listed = await list_tool_connections(user_id)
    assert len(listed) == 1
    assert listed[0]["origin"] == "user"

    resolved = await get_tool_connection(user_id, "analytics")
    assert resolved["connection_string"] == "sqlite:///mine.db"
