"""Authenticated branch controls for the caller's GitHub installation."""

from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.integrations.credentials import get_active_credential_for_user
from cognee.modules.integrations.github.branches import (
    list_repository_branches,
    select_repository_branch,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


class BranchSelection(InDTO):
    repository: str
    branch: str | None = None


class BranchList(OutDTO):
    branches: list[str]


class SelectedBranch(OutDTO):
    branch: str | None = None


async def _connection(
    account_id: str | None = None,
    user: User = Depends(get_authenticated_user),
) -> IntegrationCredential:
    credential = await get_active_credential_for_user(cast(UUID, user.id), "github", account_id)
    if credential is None:
        raise HTTPException(status_code=404, detail="Active GitHub connection not found")
    return credential


def get_github_branches_router() -> APIRouter:
    router = APIRouter()

    @router.get("/github/repositories/branches", response_model=BranchList)
    async def branches(repository: str, credential=Depends(_connection)) -> BranchList:
        try:
            return BranchList(branches=await list_repository_branches(credential, repository))
        except PermissionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        except RuntimeError:
            raise HTTPException(status_code=502, detail="Could not list GitHub branches") from None

    @router.put("/github/repositories/branches", response_model=SelectedBranch)
    async def select_branch(
        payload: BranchSelection, credential=Depends(_connection)
    ) -> SelectedBranch:
        try:
            await select_repository_branch(credential, payload.repository, payload.branch)
        except PermissionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except RuntimeError:
            raise HTTPException(
                status_code=502, detail="Could not update GitHub branch selection"
            ) from None
        return SelectedBranch(branch=payload.branch)

    return router
