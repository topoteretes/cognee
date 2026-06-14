"""HTTP router for listing dataset-scoped skills.

Exposes the ``list_skills`` SDK helper over HTTP with an explicit response
schema and caller-scoped authorization, mirroring the schema-inventory router.
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


class ErrorResponse(BaseModel):
    """Generic API error response."""

    error: str


def get_skills_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/",
        response_model=list[SkillListItem],
        responses={
            403: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
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
            default=False,
            description="Include skills whose is_active flag is false.",
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[dict]:
        """Return the skills available in an authorized dataset.

        Each item carries the skill's publisher metadata (maintainer, version,
        tags, license) so the UI can show who maintains it. Wraps the
        ``list_skills`` SDK helper.
        """
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
            datasets = await get_authorized_existing_datasets([dataset_id], "read", user)
            if not datasets:
                raise PermissionDeniedError(message="Not authorized to read this dataset")

            return await list_skills(
                dataset=datasets[0].id,
                include_inactive=include_inactive,
            )
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403,
                content={"error": "Not authorized to read this dataset"},
            )
        except Exception as exc:
            logger.error("list skills failed: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to list skills"},
            )

    return router
