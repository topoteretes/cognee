"""HTTP router for listing, fetching and toggling dataset-scoped skills.

Exposes the skills SDK helpers over HTTP with explicit response schemas and
caller-scoped authorization, mirroring the schema-inventory router.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()


class SkillListItem(BaseModel):
    """One skill row returned by GET /skills."""

    id: str = Field(description="Stable skill identifier.")
    name: str = Field(description="Skill name.")
    description: str = Field(default="", description="Short summary for routing.")
    maintainer: str = Field(
        default="", description="Publishing company / team that maintains the skill."
    )
    maintainer_url: str = Field(default="", description="Maintainer homepage or repo URL.")
    version: str = Field(default="", description="Skill version string.")
    tags: list[str] = Field(default_factory=list, description="Free-form category tags.")
    license: str = Field(default="", description="License identifier.")
    declared_tools: list[str] = Field(
        default_factory=list, description="Tools the skill is allowed to use."
    )
    dataset_scope: list[str] = Field(
        default_factory=list, description="Dataset UUIDs this skill is scoped to."
    )
    is_active: bool = Field(default=True, description="Whether the skill is active for routing.")
    source_repo_url: str = Field(default="", description="Source repository URL, when known.")
    source_dir: str = Field(default="", description="On-disk source directory.")


class SkillDetail(SkillListItem):
    """A single skill including its full procedure body."""

    procedure: str = Field(default="", description="The full skill instruction body.")


class ErrorResponse(BaseModel):
    """Generic API error response."""

    error: str


def get_skills_router() -> APIRouter:
    router = APIRouter()

    async def _authorized_dataset(dataset_id: UUID, user: User, permission: str):
        """Return the authorized dataset or raise PermissionDeniedError."""
        datasets = await get_authorized_existing_datasets([dataset_id], permission, user)
        if not datasets:
            raise PermissionDeniedError(message="Not authorized for this dataset")
        return datasets[0]

    @router.get(
        "/",
        response_model=list[SkillListItem],
        responses={403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def list_dataset_skills(
        dataset_id: UUID = Query(
            ...,
            description=(
                "Dataset UUID to scope the skills to. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        ),
        include_inactive: bool = Query(
            default=False, description="Include skills whose is_active flag is false."
        ),
        limit: int = Query(default=200, ge=1, le=1000, description="Max skills to return."),
        offset: int = Query(default=0, ge=0, description="Number of skills to skip."),
        user: User = Depends(get_authenticated_user),
    ) -> list[dict]:
        """Return the skills available in an authorized dataset, with publisher metadata."""
        send_telemetry(
            "Skills List API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/skills",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.skills.list_skills import list_skills

        try:
            dataset = await _authorized_dataset(dataset_id, user, "read")
            return await list_skills(
                dataset=dataset.id,
                include_inactive=include_inactive,
                limit=limit,
                offset=offset,
            )
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except Exception as exc:
            logger.error("list skills failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=409, content={"error": "Failed to list skills"})

    @router.get(
        "/{skill_id}",
        response_model=SkillDetail,
        responses={
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def get_dataset_skill(
        skill_id: str,
        dataset_id: UUID = Query(..., description="Dataset UUID the skill belongs to."),
        user: User = Depends(get_authenticated_user),
    ):
        """Return one skill, including its full procedure body."""
        from cognee.api.v1.skills.list_skills import get_skill

        try:
            dataset = await _authorized_dataset(dataset_id, user, "read")
            skill = await get_skill(skill_id, dataset.id)
            if skill is None:
                return JSONResponse(status_code=404, content={"error": "Skill not found"})
            return skill
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except Exception as exc:
            logger.error("get skill failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=409, content={"error": "Failed to fetch skill"})

    return router
