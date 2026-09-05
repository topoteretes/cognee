"""Real SQLite checks for local workspace permissions and invitation boundaries."""

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import delete, select, update

from cognee.api.v1.users.routers.get_workspace_router import get_workspace_router
from cognee.infrastructure.databases.relational import get_relational_config, get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user, get_user
from cognee.modules.users.models import ACL, Permission, Role, Tenant, User, UserRole, UserTenant
from cognee.modules.users.models.team_invitation import TeamInvitation
from cognee.modules.users.permissions.workspace_access import dataset_access, set_direct_access
from cognee.modules.users.tenants.invitations import accept_invitation, create_invitation


@pytest_asyncio.fixture
async def state(tmp_path, monkeypatch):
    cfg = get_relational_config()
    monkeypatch.setattr(cfg, "db_path", str(tmp_path))
    monkeypatch.setattr(cfg, "db_name", "workspace.db")
    monkeypatch.setattr(cfg, "db_provider", "sqlite")
    engine = get_relational_engine()
    await engine.create_database()
    owner_id, member_id, agent_id, stranger_id, team_id = [uuid4() for _ in range(5)]
    owner = User(id=owner_id, email="owner@example.test", hashed_password="unused", is_active=True)
    member = User(
        id=member_id, email="member@example.test", hashed_password="unused", is_active=True
    )
    stranger = User(
        id=stranger_id, email="stranger@example.test", hashed_password="unused", is_active=True
    )
    agent = User(
        id=agent_id,
        email="agent@cognee.agent",
        hashed_password="unused",
        is_active=True,
        parent_user_id=owner_id,
    )
    team = Tenant(id=team_id, name="Engineering", owner_id=owner_id)
    role = Role(id=uuid4(), name="Reviewers", tenant_id=team_id)
    dataset = Dataset(id=uuid4(), name="Shared lessons", owner_id=owner_id, tenant_id=team_id)
    permissions = [
        Permission(id=uuid4(), name=name) for name in ("read", "write", "share", "delete")
    ]
    async with engine.get_async_session() as session:
        session.add_all([owner, member, stranger])
        await session.flush()
        session.add_all([team, agent])
        await session.flush()
        for person in (owner, member, agent):
            person.tenant_id = team_id
            session.add(UserTenant(user_id=person.id, tenant_id=team_id))
        session.add_all([role, dataset, *permissions])
        await session.flush()
        session.add(UserRole(user_id=agent_id, role_id=role.id))
        session.add_all(
            [
                ACL(principal_id=owner_id, permission_id=p.id, dataset_id=dataset.id)
                for p in permissions
            ]
        )
        session.add(
            ACL(principal_id=role.id, permission_id=permissions[0].id, dataset_id=dataset.id)
        )
        await session.commit()
    owner, member, stranger, agent = [
        await get_user(ident) for ident in (owner_id, member_id, stranger_id, agent_id)
    ]
    try:
        yield SimpleNamespace(**locals())
    finally:
        await engine.engine.dispose()


@pytest.mark.asyncio
async def test_access_reloads_direct_and_inherited_permissions(state):
    await set_direct_access(state.owner, state.dataset.id, state.agent.id, "write", True)
    await set_direct_access(state.owner, state.dataset.id, state.agent.id, "write", False)
    result = await dataset_access(state.owner, state.dataset.id)
    agent = next(row for row in result["principals"] if row["id"] == state.agent.id)
    assert agent["direct"] == []
    assert agent["inherited"] == agent["effective"] == ["read"]
    assert state.stranger.id not in [row["id"] for row in result["principals"]]


@pytest.mark.asyncio
async def test_access_cannot_change_owner_or_grant_foreign_principal(state):
    with pytest.raises(ValueError, match="owner"):
        await set_direct_access(state.owner, state.dataset.id, state.owner.id, "read", False)
    with pytest.raises(PermissionError, match="Principal"):
        await set_direct_access(state.owner, state.dataset.id, state.stranger.id, "read", True)
    with pytest.raises(PermissionDeniedError):
        await set_direct_access(state.member, state.dataset.id, state.agent.id, "read", True)


@pytest.mark.asyncio
async def test_invitation_is_email_bound_and_one_use(state):
    invitation, token = await create_invitation(
        state.owner, state.team.id, state.stranger.email.upper()
    )
    async with state.engine.get_async_session() as session:
        saved = await session.get(TeamInvitation, invitation.id)
        assert saved.token_hash == hashlib.sha256(token.encode()).hexdigest()
        assert token not in saved.token_hash
    with pytest.raises(ValueError, match="another email"):
        await accept_invitation(state.member, token)
    assert await accept_invitation(state.stranger, token) == state.team.id
    with pytest.raises(ValueError):
        await accept_invitation(state.stranger, token)
    async with state.engine.get_async_session() as session:
        assert await session.get(UserTenant, (state.stranger.id, state.team.id)) is not None
        grants = (
            (await session.execute(select(ACL).where(ACL.principal_id == state.stranger.id)))
            .scalars()
            .all()
        )
        assert grants == []  # joining never grants another person's private datasets


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["expired", "revoked", "owner_changed"])
async def test_invalidated_invitation_cannot_join(state, condition):
    invitation, token = await create_invitation(state.owner, state.team.id, state.stranger.email)
    async with state.engine.get_async_session() as session:
        if condition == "owner_changed":
            await session.execute(
                update(Tenant).where(Tenant.id == state.team.id).values(owner_id=state.member.id)
            )
        else:
            value = (
                {"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
                if condition == "expired"
                else {"revoked_at": datetime.now(timezone.utc)}
            )
            await session.execute(
                update(TeamInvitation).where(TeamInvitation.id == invitation.id).values(**value)
            )
        await session.commit()
    with pytest.raises(ValueError):
        await accept_invitation(state.stranger, token)


@pytest.mark.asyncio
async def test_invitation_requires_human_team_owner(state):
    for person in (state.member, state.stranger, state.agent):
        with pytest.raises(PermissionError):
            await create_invitation(person, state.team.id, "invite@example.test")


@pytest.mark.asyncio
async def test_workspace_http_uses_real_account_and_persisted_grants(state, monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEYS", "[]")
    app = FastAPI()
    app.include_router(get_workspace_router(), prefix="/api/v1/workspace")
    app.dependency_overrides[get_authenticated_user] = lambda: state.owner
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/workspace/context")
        assert response.status_code == 200
        assert response.json()["user"]["id"] == str(state.owner.id)
        assert response.json()["teams"][0]["is_owner"] is True
        assert all(not provider["configured"] for provider in response.json()["providers"])
        response = await client.put(
            f"/api/v1/workspace/datasets/{state.dataset.id}/access",
            json={"principal_id": str(state.member.id), "permission": "write", "allowed": True},
        )
        assert response.status_code == 200
        member = next(
            row for row in response.json()["principals"] if row["id"] == str(state.member.id)
        )
        assert member["effective"] == ["write"]
        response = await client.post(
            f"/api/v1/workspace/teams/{state.team.id}/invitations", json={"email": "not-an-email"}
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_plugin_agent_without_team_membership_has_only_direct_access(state):
    async with state.engine.get_async_session() as session:
        await session.execute(delete(UserTenant).where(UserTenant.user_id == state.agent.id))
        session.add(
            ACL(
                principal_id=state.team.id,
                permission_id=state.permissions[0].id,
                dataset_id=state.dataset.id,
            )
        )
        await session.commit()
    await set_direct_access(state.owner, state.dataset.id, state.agent.id, "write", True)
    listing = await dataset_access(state.owner, state.dataset.id)
    agent = next(row for row in listing["principals"] if row["id"] == state.agent.id)
    assert agent["effective"] == ["write"]
    assert agent["inherited"] == []


@pytest.mark.asyncio
async def test_toggling_one_permission_preserves_other_changes(state):
    await set_direct_access(state.owner, state.dataset.id, state.member.id, "read", True)
    await set_direct_access(state.owner, state.dataset.id, state.member.id, "write", True)
    await set_direct_access(state.owner, state.dataset.id, state.member.id, "read", False)
    listing = await dataset_access(state.owner, state.dataset.id)
    member = next(row for row in listing["principals"] if row["id"] == state.member.id)
    assert member["effective"] == ["write"]
