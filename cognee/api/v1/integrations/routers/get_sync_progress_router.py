"""Authenticated progress for a user's current provider connection."""

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from cognee.modules.integrations.credentials import (
    CredentialInactiveError,
    get_active_credential_for_user,
)
from cognee.modules.integrations.registry import get_integration
from cognee.modules.integrations.sync_progress import SyncProgressResult, get_sync_progress
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def get_sync_progress_router() -> APIRouter:
    router = APIRouter()

    @router.get("/{provider}/sync/progress", response_model=SyncProgressResult)
    async def sync_progress(
        provider: str,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> SyncProgressResult:
        try:
            get_integration(provider)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown integration") from None
        credential = await get_active_credential_for_user(cast(UUID, user.id), provider, account_id)
        if credential is None:
            raise HTTPException(status_code=404, detail="Active connection not found")
        try:
            return await get_sync_progress(credential)
        except CredentialInactiveError:
            raise HTTPException(status_code=404, detail="Active connection not found") from None

    return router
