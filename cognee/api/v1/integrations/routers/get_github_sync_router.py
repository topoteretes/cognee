"""Authenticated GitHub sync policy and read-only content preview."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ConfigDict, StrictBool

from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.integrations.background import spawn_background
from cognee.modules.integrations.credentials import (
    CredentialInactiveError,
    get_active_credential_for_user,
)
from cognee.modules.integrations.github.preview import list_repositories, preview_repository
from cognee.modules.integrations.github.sync import sync_repositories
from cognee.modules.integrations.github.sync_settings import (
    NEXT_SYNC_KEY,
    GithubRepositorySettings,
    GithubSyncSettings,
    read_sync_settings,
    run_due_github_syncs,
    save_sync_settings,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


class GithubSettingsPayload(InDTO):
    model_config = ConfigDict(extra="forbid")
    settings: GithubSyncSettings
    acknowledge_removal: StrictBool = False
    sync_now: StrictBool = False


class GithubPreviewPayload(InDTO):
    model_config = ConfigDict(extra="forbid")
    repository_id: str
    settings: GithubRepositorySettings


class GithubSettingsResponse(OutDTO):
    settings: GithubSyncSettings
    next_sync_at: str | None = None
    last_synced_at: datetime | None = None
    sync_status: str | None = None


class GithubSettingsSavedResponse(GithubSettingsResponse):
    sync_requested: bool


class GithubPreviewResponse(OutDTO):
    repository_id: str
    branch: str
    tree_sha: str | None = None
    selected: list[str]
    skipped: dict[str, list[str]]


class GithubRepositoryResponse(OutDTO):
    id: str
    full_name: str
    default_branch: str | None = None


class GithubDueResponse(OutDTO):
    accepted: bool


async def _connection(user: User, account_id: str | None):
    credential = await get_active_credential_for_user(user.id, "github", account_id)
    if credential is None:
        raise HTTPException(status_code=404, detail="Active GitHub connection not found")
    return credential


def _settings_response(credential) -> dict:
    return {
        "settings": read_sync_settings(credential),
        "next_sync_at": (credential.provider_metadata or {}).get(NEXT_SYNC_KEY),
        "last_synced_at": credential.last_synced_at,
        "sync_status": credential.sync_status,
    }


def get_github_sync_router() -> APIRouter:
    router = APIRouter(prefix="/github/sync")

    @router.get("/settings", response_model=GithubSettingsResponse)
    async def get_settings(
        account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ):
        return _settings_response(await _connection(user, account_id))

    @router.put("/settings", response_model=GithubSettingsSavedResponse)
    async def put_settings(
        payload: GithubSettingsPayload,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ):
        credential = await _connection(user, account_id)
        previous = read_sync_settings(credential)
        try:
            updated = await save_sync_settings(
                credential, payload.settings, acknowledge_removal=payload.acknowledge_removal
            )
        except CredentialInactiveError:
            raise HTTPException(
                status_code=404, detail="Active GitHub connection not found"
            ) from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        resumed = not previous.enabled and payload.settings.enabled
        changed_content = previous.repositories != payload.settings.repositories
        wait_for_first_slot = (
            credential.last_synced_at is None and payload.settings.first_sync == "scheduled"
        )
        reconcile = payload.sync_now or (
            payload.settings.enabled and (resumed or (changed_content and not wait_for_first_slot))
        )
        if reconcile:
            # Reuse the SDK's tracked background-task mechanism so server shutdown
            # observes this work just like manual sync and installation hooks.
            spawn_background(
                sync_repositories(updated, trigger="manual" if payload.sync_now else "settings"),
                description="GitHub sync after settings change",
            )
        return {**_settings_response(updated), "sync_requested": reconcile}

    @router.get("/repositories", response_model=list[GithubRepositoryResponse])
    async def repositories(
        account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ):
        """Repositories the installation can read, for choosing what to sync."""
        credential = await _connection(user, account_id)
        try:
            return await list_repositories(credential)
        except CredentialInactiveError:
            raise HTTPException(
                status_code=404, detail="Active GitHub connection not found"
            ) from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except RuntimeError:
            raise HTTPException(
                status_code=502, detail="GitHub repository listing failed"
            ) from None

    @router.post("/preview", response_model=GithubPreviewResponse)
    async def preview(
        payload: GithubPreviewPayload,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ):
        credential = await _connection(user, account_id)
        try:
            return await preview_repository(credential, payload.repository_id, payload.settings)
        except CredentialInactiveError:
            raise HTTPException(
                status_code=404, detail="Active GitHub connection not found"
            ) from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except RuntimeError:
            raise HTTPException(
                status_code=502, detail="GitHub repository preview failed"
            ) from None

    @router.post("/due", response_model=GithubDueResponse)
    async def run_due(user: User = Depends(get_authenticated_user)):
        """Start this user's due schedules in the background; no scheduler is installed.

        A due sync clones and builds every selected repository, far longer than a request lives.
        Nothing starts, and ``accepted`` is false, when the user has no GitHub connection.
        """
        if await get_active_credential_for_user(user.id, "github", None) is None:
            return {"accepted": False}
        spawn_background(run_due_github_syncs(user_id=user.id), description="Due GitHub syncs")
        return {"accepted": True}

    return router
