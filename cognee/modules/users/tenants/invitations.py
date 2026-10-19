"""Explicit invitation creation and acceptance without sending email."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Tenant, User, UserTenant
from cognee.modules.users.models.team_invitation import TeamInvitation


def _human(user: User) -> None:
    if user.parent_user_id is not None or not user.is_active:
        raise PermissionError("Sign in as a person to manage team invitations")


async def create_invitation(user: User, tenant_id: UUID, email: str) -> tuple[TeamInvitation, str]:
    _human(user)
    token = secrets.token_urlsafe(32)
    async with get_relational_engine().get_async_session() as session:
        tenant = await session.get(Tenant, tenant_id)
        if tenant is None or tenant.owner_id != user.id:
            raise PermissionError("Only the team owner can invite members")
        invitation = TeamInvitation(
            tenant_id=tenant_id,
            created_by=user.id,
            email=email.strip().casefold(),
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        )
        session.add(invitation)
        await session.commit()
        await session.refresh(invitation)
        return invitation, token


async def accept_invitation(user: User, token: str) -> UUID:
    _human(user)
    now = datetime.now(timezone.utc)
    async with get_relational_engine().get_async_session() as session:
        invitation = (
            await session.execute(
                select(TeamInvitation).where(
                    TeamInvitation.token_hash == hashlib.sha256(token.encode()).hexdigest(),
                    TeamInvitation.email == user.email.strip().casefold(),
                    TeamInvitation.expires_at > now,
                    TeamInvitation.accepted_at.is_(None),
                    TeamInvitation.revoked_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if invitation is None:
            raise ValueError("Invitation is invalid, expired, or belongs to another email")
        tenant = await session.get(Tenant, invitation.tenant_id)
        if tenant is None or tenant.owner_id != invitation.created_by:
            raise ValueError("The invitation's owner can no longer invite members")
        claimed = await session.execute(
            update(TeamInvitation)
            .where(
                TeamInvitation.id == invitation.id,
                TeamInvitation.accepted_at.is_(None),
                TeamInvitation.revoked_at.is_(None),
            )
            .values(accepted_at=now)
        )
        if claimed.rowcount != 1:
            raise ValueError("Invitation was already used or revoked")
        membership = await session.get(UserTenant, (user.id, tenant.id))
        if membership is None:
            session.add(UserTenant(user_id=user.id, tenant_id=tenant.id))
        await session.commit()
        return tenant.id
