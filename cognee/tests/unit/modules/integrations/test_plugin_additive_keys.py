"""Plugin agent keys: provisioning only ever adds one, and a full agent is a 409.

``POST /plugins/{plugin_key}/provision`` must never revoke another machine's
key. An agent already holding ``MAX_USER_API_KEYS`` keys answers 409 and nothing
changes; its owner frees a slot by revoking a key. Runs
against a real database through the real router, because the claims are about
rows and races: which keys still authenticate, how many rows remain, and what
two concurrent transactions leave behind.
"""

import asyncio
import os
import pathlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

import cognee

_SYSTEM_ROOT = str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent.parent.parent.parent,
            ".cognee_system/test_plugin_additive_keys",
        )
    ).resolve()
)

PROVISION = "/api/v1/integrations/plugins/claude-code/provision"
KEYS = "/api/v1/integrations/plugins/claude-code/keys"


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    """Point cognee at a database of its own so these rows never touch dev data."""
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    cognee.config.system_root_directory(_SYSTEM_ROOT)

    # The engine is process-global (@lru_cache), so another module in the same
    # run leaves one cached against its own system root.
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


@pytest.fixture
def key_settings():
    """Override the key cap for one test."""
    from cognee.modules.users.api_key.create_api_key import apiKeySettings

    original = apiKeySettings.max_user_api_keys

    def _set(max_user_api_keys):
        apiKeySettings.max_user_api_keys = max_user_api_keys

    yield _set
    _set(original)


async def _parent():
    """A fresh human user to provision plugin agents for."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.methods import get_user
    from cognee.modules.users.models import User

    user_id = uuid4()
    async with get_relational_engine().get_async_session() as session:
        session.add(
            User(
                id=user_id,
                email=f"{user_id}@example.com",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await session.commit()
    return await get_user(user_id)


def _client(parent) -> httpx.AsyncClient:
    from cognee.api.v1.integrations.routers.get_integrations_router import (
        get_integrations_router,
    )
    from cognee.modules.users.methods import get_authenticated_user

    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: parent
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _provision(client, path=PROVISION):
    response = await client.post(path)
    assert response.status_code == 200, response.text
    return response.json()


async def _resolve(key):
    """The user a key authenticates as, or None, resolved the way a request resolves it."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.get_user_db import get_user_db_context
    from cognee.modules.users.get_user_manager import get_user_manager_context

    async with (
        get_relational_engine().get_async_session() as session,
        get_user_db_context(session) as user_db,
        get_user_manager_context(user_db) as user_manager,
    ):
        return await user_manager.get_by_token(key)


async def _key_ids(user_id):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models.UserApiKey import UserApiKey

    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(UserApiKey.id).where(UserApiKey.user_id == UUID(str(user_id)))
        )
        return {str(key_id) for key_id in rows.scalars().all()}


async def _age_key(key_id, *, created_ago=None, last_used_ago=None, never_used=False):
    """Backdate a key's timestamps, as if it were created / last used that long ago."""
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models.UserApiKey import UserApiKey

    now = datetime.now(timezone.utc)
    values = {}
    if created_ago is not None:
        values["created_at"] = now - created_ago
    if never_used:
        values["last_used_at"] = None
    elif last_used_ago is not None:
        values["last_used_at"] = now - last_used_ago
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(UserApiKey).where(UserApiKey.id == UUID(str(key_id))).values(**values)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# add-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provisioning_twice_leaves_both_keys_valid():
    async with _client(await _parent()) as client:
        first = await _provision(client)
        second = await _provision(client)

    assert first["created"] is True and second["created"] is False
    assert first["agentId"] == second["agentId"]
    assert first["keyId"] != second["keyId"]
    assert (await _resolve(first["apiKey"])).id == (await _resolve(second["apiKey"])).id
    assert await _key_ids(first["agentId"]) == {first["keyId"], second["keyId"]}


@pytest.mark.asyncio
async def test_concurrent_provisions_leave_every_key_valid():
    async with _client(await _parent()) as client:
        await _provision(client)
        bodies = await asyncio.gather(*(_provision(client) for _ in range(3)))

    assert len({body["agentId"] for body in bodies}) == 1
    for body in bodies:
        assert await _resolve(body["apiKey"]) is not None


@pytest.mark.asyncio
async def test_concurrent_first_create_recovery_adds_a_key_and_never_rotates():
    # No agent yet: all calls race create_agent, the losers take the
    # duplicate-email branch and must add, not rotate.
    async with _client(await _parent()) as client:
        bodies = await asyncio.gather(*(_provision(client) for _ in range(4)))

    assert len({body["agentId"] for body in bodies}) == 1
    assert sum(body["created"] for body in bodies) == 1
    assert await _key_ids(bodies[0]["agentId"]) == {body["keyId"] for body in bodies}
    for body in bodies:
        assert await _resolve(body["apiKey"]) is not None


# ---------------------------------------------------------------------------
# the cap: a 409, never a deletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provisioning_never_deletes_a_key_however_old():
    async with _client(await _parent()) as client:
        idle = await _provision(client)
        orphan = await _provision(client)
        await _age_key(
            idle["keyId"], created_ago=timedelta(days=400), last_used_ago=timedelta(days=365)
        )
        await _age_key(orphan["keyId"], created_ago=timedelta(days=30), never_used=True)
        new = await _provision(client)

    assert await _key_ids(new["agentId"]) == {idle["keyId"], orphan["keyId"], new["keyId"]}
    assert await _resolve(idle["apiKey"]) is not None


@pytest.mark.asyncio
async def test_a_full_agent_answers_409_and_nothing_changes(key_settings):
    key_settings(max_user_api_keys=2)
    async with _client(await _parent()) as client:
        stale = await _provision(client)
        fresh = await _provision(client)
        await _age_key(
            stale["keyId"], created_ago=timedelta(days=400), last_used_ago=timedelta(days=365)
        )
        full = await client.post(PROVISION)

    assert full.status_code == 409
    assert "Revoke an old machine's key" in full.json()["detail"]
    assert await _key_ids(stale["agentId"]) == {stale["keyId"], fresh["keyId"]}
    assert await _resolve(stale["apiKey"]) is not None
    assert await _resolve(fresh["apiKey"]) is not None


@pytest.mark.asyncio
async def test_re_provisioning_past_the_cap_answers_409_and_revokes_nothing(key_settings):
    """Reinstalls, devcontainers and CI runners each mint a key; past the cap they
    get a 409 rather than logging out a machine that holds one of the first keys."""
    key_settings(max_user_api_keys=3)
    async with _client(await _parent()) as client:
        responses = [await client.post(PROVISION) for _ in range(10)]

    assert [response.status_code for response in responses] == [200] * 3 + [409] * 7
    kept = [response.json() for response in responses[:3]]
    assert await _key_ids(kept[0]["agentId"]) == {body["keyId"] for body in kept}
    for body in kept:
        assert await _resolve(body["apiKey"]) is not None


@pytest.mark.asyncio
async def test_revoking_a_key_frees_a_slot(key_settings):
    key_settings(max_user_api_keys=1)
    async with _client(await _parent()) as client:
        old = await _provision(client)
        assert (await client.post(PROVISION)).status_code == 409
        assert (await client.delete(f"{KEYS}/{old['keyId']}")).status_code == 200
        new = await _provision(client)

    assert await _key_ids(new["agentId"]) == {new["keyId"]}


@pytest.mark.asyncio
async def test_the_cap_counts_only_that_agents_keys(key_settings):
    from cognee.modules.users.api_key.create_api_key import create_api_key

    parent = await _parent()
    parent_key = await create_api_key(parent, "laptop")

    key_settings(max_user_api_keys=1)
    async with _client(parent) as client:
        claude = await _provision(client)
        assert (await client.post(PROVISION)).status_code == 409
        codex = await _provision(client, "/api/v1/integrations/plugins/codex/provision")

    assert str(parent_key.id) in await _key_ids(parent.id)
    assert await _key_ids(claude["agentId"]) == {claude["keyId"]}
    assert await _key_ids(codex["agentId"]) == {codex["keyId"]}


@pytest.mark.asyncio
async def test_provisions_racing_at_the_cap_never_exceed_it(key_settings):
    key_settings(max_user_api_keys=3)
    async with _client(await _parent()) as client:
        first = await _provision(client)
        await _provision(client)
        responses = await asyncio.gather(*(client.post(PROVISION) for _ in range(4)))

    assert sorted(response.status_code for response in responses) == [200, 409, 409, 409]
    assert len(await _key_ids(first["agentId"])) == 3


@pytest.mark.asyncio
async def test_the_key_is_built_and_hashed_before_the_lock_is_taken():
    """With HASH_API_KEY on, hashing is a slow PBKDF2 run; it must not hold the lock."""
    from cognee.modules.integrations import plugin_keys
    from cognee.modules.users.api_key.create_api_key import build_api_key_row

    async with _client(await _parent()) as client:
        agent_id = UUID((await _provision(client))["agentId"])

    calls = []
    real_lock = plugin_keys._lock_agent_row

    def _build(user_id, name):
        calls.append("build")
        return build_api_key_row(user_id, name)

    async def _lock(session, user_id):
        calls.append("lock")
        await real_lock(session, user_id)

    with (
        patch.object(plugin_keys, "build_api_key_row", side_effect=_build),
        patch.object(plugin_keys, "_lock_agent_row", side_effect=_lock),
    ):
        await plugin_keys.add_plugin_agent_key(agent_id, "claude-code")

    assert calls == ["build", "lock"]


# ---------------------------------------------------------------------------
# capability marker and the key routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_provision_operation_carries_the_additive_keys_marker():
    async with _client(await _parent()) as client:
        response = await client.get("/openapi.json")
    operation = response.json()["paths"]["/api/v1/integrations/plugins/{plugin_key}/provision"]
    assert operation["post"]["x-cognee-additive-plugin-keys"] is True


@pytest.mark.asyncio
async def test_list_shows_key_metadata_never_the_key():
    async with _client(await _parent()) as client:
        a = await _provision(client)
        await _provision(client)
        response = await client.get(KEYS)

    assert response.status_code == 200
    keys = response.json()["keys"]
    assert len(keys) == 2
    assert set(keys[0]) == {"id", "name", "label", "createdAt", "lastUsedAt"}
    assert a["apiKey"] not in response.text


@pytest.mark.asyncio
async def test_list_and_revoke_are_scoped_to_the_callers_own_agent():
    owner, stranger = await _parent(), await _parent()
    async with _client(owner) as owner_client, _client(stranger) as stranger_client:
        a = await _provision(owner_client)
        b = await _provision(owner_client)
        await _provision(stranger_client)

        stranger_keys = (await stranger_client.get(KEYS)).json()["keys"]
        assert a["keyId"] not in {key["id"] for key in stranger_keys}

        # Another user's key, and this user's key under another plugin, are both 404s.
        assert (await stranger_client.delete(f"{KEYS}/{a['keyId']}")).status_code == 404
        codex_keys = "/api/v1/integrations/plugins/codex/keys"
        assert (await owner_client.delete(f"{codex_keys}/{a['keyId']}")).status_code == 404
        assert await _resolve(a["apiKey"]) is not None

        revoked = await owner_client.delete(f"{KEYS}/{a['keyId']}")
        assert revoked.status_code == 200 and revoked.json() == {"revoked": True}
        assert await _resolve(a["apiKey"]) is None
        assert await _resolve(b["apiKey"]) is not None
        assert (await owner_client.delete(f"{KEYS}/{a['keyId']}")).status_code == 404


@pytest.mark.asyncio
async def test_an_agent_cannot_list_or_revoke_plugin_keys():
    from cognee.modules.users.methods import get_user

    async with _client(await _parent()) as owner_client:
        a = await _provision(owner_client)
    agent = await get_user(UUID(a["agentId"]))

    async with _client(agent) as agent_client:
        assert (await agent_client.get(KEYS)).status_code == 403
        assert (await agent_client.delete(f"{KEYS}/{a['keyId']}")).status_code == 403
    assert await _resolve(a["apiKey"]) is not None


@pytest.mark.asyncio
async def test_disconnect_still_revokes_every_key():
    async with _client(await _parent()) as client:
        a = await _provision(client)
        b = await _provision(client)
        response = await client.delete("/api/v1/integrations/plugins/claude-code")

    assert response.status_code == 200 and response.json() == {"disconnected": True}
    assert await _resolve(a["apiKey"]) is None
    assert await _resolve(b["apiKey"]) is None
    assert await _key_ids(a["agentId"]) == set()


@pytest.mark.asyncio
async def test_disconnect_racing_a_provision_never_fails_halfway():
    """Disconnect and provision lock the same agent row, so disconnect deletes one
    consistent set of keys and leaves no old key behind."""
    async with _client(await _parent()) as client:
        a = await _provision(client)
        b = await _provision(client)
        disconnect, provision = await asyncio.gather(
            client.delete("/api/v1/integrations/plugins/claude-code"), client.post(PROVISION)
        )

    assert disconnect.status_code == 200 and disconnect.json() == {"disconnected": True}
    assert provision.status_code == 200
    # Either order is valid; the provision's key survives only if it committed last.
    assert await _key_ids(a["agentId"]) <= {provision.json()["keyId"]}
    assert await _resolve(a["apiKey"]) is None
    assert await _resolve(b["apiKey"]) is None


@pytest.mark.asyncio
async def test_delete_api_key_tells_not_found_apart_from_failures():
    """Revoke maps only not-found to 404; a real failure must not look like one."""
    from cognee.modules.users.api_key import delete_api_key as delete_module
    from cognee.modules.users.api_key.exceptions import ApiKeyDeletionError, ApiKeyNotFoundError

    parent = await _parent()
    with pytest.raises(ApiKeyNotFoundError):
        await delete_module.delete_api_key(parent, uuid4())

    with (
        patch.object(delete_module, "select", side_effect=RuntimeError("db down")),
        pytest.raises(ApiKeyDeletionError) as failure,
    ):
        await delete_module.delete_api_key(parent, uuid4())
    assert not isinstance(failure.value, ApiKeyNotFoundError)
