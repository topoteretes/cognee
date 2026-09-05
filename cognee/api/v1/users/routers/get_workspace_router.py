"""Local and self-hosted workspace controls backed by SDK authorization."""

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations.crypto import encrypt_credentials
from cognee.modules.integrations.github.github_settings import github_settings
from cognee.modules.integrations.slack.slack_settings import slack_settings
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import Tenant, User, UserTenant
from cognee.modules.users.models.team_invitation import TeamInvitation
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.modules.users.permissions.workspace_access import dataset_access, set_direct_access
from cognee.modules.users.tenants.invitations import accept_invitation, create_invitation

CurrentUser = Annotated[User, Depends(get_authenticated_user)]


class InviteRequest(BaseModel):
    email: EmailStr


class AcceptRequest(BaseModel):
    token: str = Field(min_length=32, max_length=128)


class AccessRequest(BaseModel):
    principal_id: UUID
    permission: Literal["read", "write", "share", "delete"]
    allowed: bool


def _invite_json(row: TeamInvitation) -> dict:
    result = {
        name: getattr(row, name)
        for name in (
            "id",
            "email",
            "tenant_id",
            "created_at",
            "expires_at",
            "accepted_at",
            "revoked_at",
        )
    }
    for name, value in result.items():
        if isinstance(value, datetime) and value.tzinfo is None:
            result[name] = value.replace(tzinfo=timezone.utc)
    return result


def get_workspace_router() -> APIRouter:
    router = APIRouter()

    @router.get("/context")
    async def context(user: CurrentUser):
        async with get_relational_engine().get_async_session() as session:
            teams = (
                (
                    await session.execute(
                        select(Tenant)
                        .join(UserTenant, Tenant.id == UserTenant.tenant_id)
                        .where(UserTenant.user_id == user.id)
                    )
                )
                .scalars()
                .all()
            )
        permissions: dict[str, list[str]] = {}
        datasets = {}
        for permission in ("read", "write", "share", "delete"):
            for dataset in await get_all_user_permission_datasets(user, permission):
                key = str(dataset.id)
                permissions.setdefault(key, []).append(permission)
                datasets[key] = {"id": key, "name": dataset.name, "owner_id": str(dataset.owner_id)}
        providers = []
        try:
            encrypt_credentials({})
            credentials_ready = True
        except (RuntimeError, ValueError, TypeError, AttributeError):
            credentials_ready = False
        for provider, settings, fields in (
            (
                "slack",
                slack_settings,
                (
                    "client_id",
                    "client_secret",
                    "signing_secret",
                    "redirect_uri",
                    "frontend_base_url",
                ),
            ),
            (
                "github",
                github_settings,
                (
                    "client_id",
                    "client_secret",
                    "app_id",
                    "app_slug",
                    "app_private_key",
                    "webhook_secret",
                    "frontend_base_url",
                ),
            ),
        ):
            missing = [
                f"{provider.upper()}_{field.upper()}"
                for field in fields
                if not getattr(settings, field)
            ]
            if not credentials_ready:
                missing.append("INTEGRATION_CREDENTIALS_KEY (valid encryption key required)")
            providers.append(
                {"provider": provider, "configured": not missing, "missing_settings": missing}
            )
        return {
            "user": {
                "id": user.id,
                "email": user.email,
                "tenant_id": user.tenant_id,
                "is_agent": user.parent_user_id is not None,
            },
            "teams": [
                {"id": team.id, "name": team.name, "is_owner": team.owner_id == user.id}
                for team in teams
            ],
            "datasets": [
                {**dataset, "permissions": permissions[key]} for key, dataset in datasets.items()
            ],
            "providers": providers,
        }

    @router.get("/datasets/{dataset_id}/access")
    async def access(dataset_id: UUID, user: CurrentUser):
        try:
            return await dataset_access(user, dataset_id)
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error

    @router.get("/promotions")
    async def promotions(user: CurrentUser, offset: int = Query(0, ge=0)):
        readable = await get_all_user_permission_datasets(user, "read")
        async with get_relational_engine().get_async_session() as session:
            rows = (
                await session.execute(
                    select(Data, Dataset.name)
                    .join(Dataset, Data.dataset_id == Dataset.id)
                    .where(
                        Data.dataset_id.in_([dataset.id for dataset in readable]),
                        Data.system_metadata["promotion"]["source_data_id"]
                        .as_string()
                        .is_not(None),
                    )
                    .order_by(Data.created_at.desc(), Data.id.desc())
                    .limit(50)
                    .offset(offset)
                )
            ).all()
        return [
            {
                "data_id": row.id,
                "name": row.name,
                "dataset_id": row.dataset_id,
                "dataset_name": dataset_name,
                "promotion": row.system_metadata["promotion"],
            }
            for row, dataset_name in rows
        ]

    @router.put("/datasets/{dataset_id}/access")
    async def save_access(dataset_id: UUID, payload: AccessRequest, user: CurrentUser):
        try:
            await set_direct_access(
                user, dataset_id, payload.principal_id, payload.permission, payload.allowed
            )
            return await dataset_access(user, dataset_id)
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/teams/{tenant_id}/invitations")
    async def invitations(tenant_id: UUID, user: CurrentUser):
        async with get_relational_engine().get_async_session() as session:
            tenant = await session.get(Tenant, tenant_id)
            if tenant is None or tenant.owner_id != user.id or user.parent_user_id:
                raise HTTPException(403, "Only the team owner can inspect invitations")
            rows = (
                (
                    await session.execute(
                        select(TeamInvitation)
                        .where(TeamInvitation.tenant_id == tenant_id)
                        .order_by(TeamInvitation.created_at.desc())
                    )
                )
                .scalars()
                .all()
            )
            return [_invite_json(row) for row in rows]

    @router.post("/teams/{tenant_id}/invitations")
    async def invite(tenant_id: UUID, payload: InviteRequest, user: CurrentUser):
        try:
            invitation, token = await create_invitation(user, tenant_id, str(payload.email))
            return {**_invite_json(invitation), "token": token}
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error

    @router.delete("/teams/{tenant_id}/invitations/{invitation_id}")
    async def revoke(tenant_id: UUID, invitation_id: UUID, user: CurrentUser):
        async with get_relational_engine().get_async_session() as session:
            tenant = await session.get(Tenant, tenant_id)
            if tenant is None or tenant.owner_id != user.id or user.parent_user_id:
                raise HTTPException(403, "Only the team owner can revoke invitations")
            await session.execute(
                update(TeamInvitation)
                .where(
                    TeamInvitation.id == invitation_id,
                    TeamInvitation.tenant_id == tenant_id,
                    TeamInvitation.accepted_at.is_(None),
                )
                .values(revoked_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return {"revoked": True}

    @router.post("/invitations/accept")
    async def accept(payload: AcceptRequest, user: CurrentUser):
        try:
            return {"tenant_id": await accept_invitation(user, payload.token)}
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    return router
