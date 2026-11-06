"""The sync-policy routes act only for the caller's own principals and connections (SDK-1434).

Real router, real credential lookup and real store on one SQLite file (the
``database`` fixture in ``conftest.py``); only the provider is fake.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient

from cognee.api.client import exception_handler as render_cognee_error
from cognee.api.client import request_validation_exception_handler as render_validation_error
from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.exceptions import CogneeApiError
from cognee.modules.integrations.base import OAuthIntegration
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import Tenant, User

DEFAULT = "/api/v1/integrations/sync/policy"
CONNECTION = "/api/v1/integrations/fake/sync/policy"
HOURLY = {"intervalMinutes": 60}


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None
    sync_now = AsyncMock()

    def authorize_url(self, state):
        return ""

    async def exchange_code(self, code):
        return {}

    def parse_installation(self, token_response):
        raise NotImplementedError

    def state_signing_secret(self):
        return "secret"

    def frontend_base_url(self):
        return ""


class _Api:
    """An HTTP client that calls as ``as_user``."""

    def __init__(self, client: AsyncClient, caller: dict):
        self._client, self._caller = client, caller

    def __call__(self, user):
        self._caller["user"] = user
        return self._client


async def _add(database, *rows):
    async with database.get_async_session() as session:
        session.add_all(rows)
        await session.commit()


def _user_row(parent=None):
    user_id = uuid4()
    return User(
        id=user_id,
        email=f"{user_id}@example.com",
        hashed_password="x",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        parent_user_id=parent,
    )


def _credential(user_id, account_id, workspace_id=None):
    return IntegrationCredential(
        user_id=user_id,
        workspace_id=workspace_id,
        provider="fake",
        provider_account_id=account_id,
        ciphertext=b"x",
        nonce=b"x",
        status="active",
    )


@pytest_asyncio.fixture
async def world(database):
    """An owner with an agent and a tenant, a member of that tenant, and a stranger.

    The owner and the stranger each connected one account; the member connected
    one into the tenant's workspace.
    """
    owner, stranger, member = _user_row(), _user_row(), _user_row()
    agent = _user_row(parent=owner.id)
    tenant = Tenant(id=uuid4(), name="acme", owner_id=owner.id)
    await _add(database, owner, stranger, member, agent, tenant)
    await _add(
        database,
        _credential(owner.id, "OWNER"),
        _credential(stranger.id, "STRANGER"),
        _credential(member.id, "MEMBER", workspace_id=tenant.id),
    )
    return SimpleNamespace(
        owner=SimpleNamespace(id=owner.id),
        stranger=SimpleNamespace(id=stranger.id),
        member=SimpleNamespace(id=member.id),
        agent=SimpleNamespace(id=agent.id),
        tenant_id=tenant.id,
    )


@pytest_asyncio.fixture
async def api(database):
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    caller = {}
    app.dependency_overrides[get_authenticated_user] = lambda: caller["user"]

    # The real app's renderers, so status codes and bodies are what a client gets.
    app.add_exception_handler(CogneeApiError, render_cognee_error)
    app.add_exception_handler(RequestValidationError, render_validation_error)

    before = dict(supported_integrations)
    use_integration(_FakeIntegration())
    _FakeIntegration.sync_now.reset_mock()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield _Api(client, caller)
    supported_integrations.clear()
    supported_integrations.update(before)


def _is_utc(value: str) -> bool:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset().total_seconds() == 0


@pytest.mark.asyncio
async def test_a_saved_policy_comes_back_in_utc_with_the_offered_intervals(api, world):
    response = await api(world.owner).put(
        DEFAULT, json={"intervalMinutes": 1440, "weekdays": [0], "timezone": "Europe/Belgrade"}
    )
    read = await api(world.owner).get(DEFAULT)

    assert response.status_code == 200
    # SQLite hands the slot back without its offset; the response carries UTC anyway.
    body = read.json()
    assert _is_utc(body["nextDueAt"])
    assert body["level"] == "default"
    assert body["intervalPresets"] == [15, 30, 60, 120, 240, 360, 480, 1440, 10080]
    assert body["customHours"] == list(range(1, 24))


@pytest.mark.asyncio
async def test_another_principals_default_reads_like_a_missing_one(api, world):
    await api(world.owner).put(DEFAULT, json=HOURLY)
    await api(world.owner).put(DEFAULT, params={"principal_id": str(world.tenant_id)}, json=HOURLY)
    missing = await api(world.stranger).get(DEFAULT)

    refused = [
        (world.stranger, world.owner.id),
        (world.agent, world.owner.id),
        (world.member, world.tenant_id),
    ]
    for caller, principal_id in refused:
        for method in ("get", "delete", "post"):
            url = f"{DEFAULT}/pause" if method == "post" else DEFAULT
            response = await getattr(api(caller), method)(
                url, params={"principal_id": str(principal_id)}
            )
            assert (response.status_code, response.json()) == (404, missing.json())


@pytest.mark.asyncio
async def test_every_default_route_acts_on_the_principal_it_names(api, world):
    owner, tenant = api(world.owner), {"principal_id": str(world.tenant_id)}
    await owner.put(DEFAULT, json=HOURLY)
    await owner.put(DEFAULT, params=tenant, json={"intervalMinutes": 240})

    read = (await owner.get(DEFAULT, params=tenant)).json()
    paused = (await owner.post(f"{DEFAULT}/pause", params=tenant)).json()
    resumed = (await owner.post(f"{DEFAULT}/resume", params=tenant)).json()
    deleted = await owner.delete(DEFAULT, params=tenant)

    assert read["intervalMinutes"] == 240
    assert (paused["intervalMinutes"], paused["paused"]) == (240, True)
    assert (resumed["intervalMinutes"], resumed["paused"]) == (240, False)
    assert deleted.status_code == 204
    assert (await owner.get(DEFAULT, params=tenant)).status_code == 404
    # The caller's own default was never touched.
    assert (await owner.get(DEFAULT)).json()["intervalMinutes"] == 60


@pytest.mark.asyncio
async def test_another_users_connection_reads_like_a_missing_one(api, world):
    """The connection is found among the caller's own, so its owner is never the caller's
    choice: naming someone else's account is the same 404 as naming none that exists."""
    await api(world.owner).put(DEFAULT, json=HOURLY)
    missing = await api(world.stranger).put(CONNECTION, params={"account_id": "NONE"}, json=HOURLY)

    for method in ("get", "put", "delete"):
        kwargs = {"json": HOURLY} if method == "put" else {}
        response = await getattr(api(world.stranger), method)(
            CONNECTION, params={"account_id": "OWNER"}, **kwargs
        )
        assert (response.status_code, response.json()) == (404, missing.json())
    assert (await api(world.owner).get(CONNECTION, params={"account_id": "OWNER"})).json()[
        "intervalMinutes"
    ] == 60


@pytest.mark.asyncio
async def test_a_workspace_connection_belongs_to_the_workspace_not_to_who_connected_it(api, world):
    """The member connected it, but its owner is the tenant, whose policies only the
    tenant's owner manages."""
    response = await api(world.member).put(CONNECTION, params={"account_id": "MEMBER"}, json=HOURLY)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_names_the_level_and_delete_returns_the_connection_to_the_default(api, world):
    owner = api(world.owner)
    await owner.put(DEFAULT, json=HOURLY)
    params = {"account_id": "OWNER"}

    inherited = (await owner.get(CONNECTION, params=params)).json()
    await owner.put(CONNECTION, params=params, json={"intervalMinutes": 240})
    own = (await owner.get(CONNECTION, params=params)).json()
    deleted = await owner.delete(CONNECTION, params=params)
    again = (await owner.get(CONNECTION, params=params)).json()

    assert (inherited["level"], inherited["intervalMinutes"]) == ("default", 60)
    assert (own["level"], own["intervalMinutes"]) == ("connection", 240)
    assert deleted.status_code == 204
    assert again["level"] == "default"


@pytest.mark.asyncio
async def test_a_manual_sync_of_a_paused_connection_is_refused_until_resumed(api, world):
    owner = api(world.owner)
    await owner.put(DEFAULT, json=HOURLY)
    paused = (await owner.post(f"{DEFAULT}/pause")).json()
    # Read back from SQLite, which drops the offset: the response is UTC anyway.
    read_back = (await owner.get(CONNECTION)).json()

    refused = await owner.post("/api/v1/integrations/fake/sync")
    await owner.post(f"{DEFAULT}/resume")
    accepted = await owner.post("/api/v1/integrations/fake/sync")

    assert paused["paused"] is True
    assert paused["pausedBy"] == str(world.owner.id)
    assert read_back["paused"] is True
    assert _is_utc(read_back["pausedAt"])
    assert refused.status_code == 409
    assert "default level" in refused.json()["detail"]
    assert "[SyncPausedError]" in refused.json()["detail"]
    assert accepted.json() == {"accepted": True}


@pytest.mark.asyncio
async def test_a_first_save_that_loses_a_race_is_a_409_not_a_500(api, world, database):
    database.before_commit = lambda: database.execute(
        "INSERT INTO integration_sync_policies (id, principal_id, interval_minutes, weekdays,"
        " timezone, paused, created_at, updated_at) VALUES (?, ?, 60, '[0]', 'UTC', 0,"
        " '2026-11-05 00:00:00', '2026-11-05 00:00:00')",
        (uuid4().hex, world.owner.id.hex),
    )

    response = await api(world.owner).put(DEFAULT, json=HOURLY)

    assert response.status_code == 409
    assert "[SyncPolicyConflictError]" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_policy_outside_the_presets_is_refused_with_the_allowed_values(api, world):
    response = await api(world.owner).put(DEFAULT, json={"intervalMinutes": 45})

    # The app answers every request validation error with 400 (cognee/api/client.py).
    assert response.status_code == 400
    assert "1440" in str(response.json())


@pytest.mark.asyncio
async def test_a_connection_held_by_its_owners_paused_default_says_so(api, world):
    """Its own row is not paused, but the paused default holds it: GET must not read as
    running while a manual sync is refused."""
    owner, params = api(world.owner), {"account_id": "OWNER"}
    await owner.put(DEFAULT, json=HOURLY)
    await owner.put(CONNECTION, params=params, json=HOURLY)

    running = (await owner.get(CONNECTION, params=params)).json()
    await owner.post(f"{DEFAULT}/pause")
    held = (await owner.get(CONNECTION, params=params)).json()
    await owner.post(f"{CONNECTION}/pause", params=params)
    both = (await owner.get(CONNECTION, params=params)).json()
    await owner.post(f"{DEFAULT}/resume")
    own_only = (await owner.get(CONNECTION, params=params)).json()
    released = (await owner.post(f"{CONNECTION}/resume", params=params)).json()

    assert running["heldBy"] is None
    assert (held["level"], held["paused"], held["heldBy"]) == ("connection", False, "default")
    assert both["heldBy"] == "default"
    assert (own_only["paused"], own_only["heldBy"]) == (True, "connection")
    assert released["heldBy"] is None
    # The default's own routes describe the default row itself.
    assert (await owner.get(DEFAULT)).json()["heldBy"] is None
