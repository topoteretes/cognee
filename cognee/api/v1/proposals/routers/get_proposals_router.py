"""HTTP router for reviewing skill-improvement proposals.

Exposes a read-only view of a stored ``SkillImprovementProposal`` over HTTP so a
caller can inspect the before/after procedure, rationale and confidence *before*
deciding whether to apply it (apply still happens via
``POST /api/v1/remember/entry`` with ``skill_improvement``). Mirrors the
skills router: explicit response schema + caller-scoped dataset authorization.
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


class ProposalDetail(BaseModel):
    """A single skill-improvement proposal, including before/after procedures."""

    proposal_id: str = Field(description="Stable proposal identifier.")
    skill_id: str = Field(default="", description="Identifier of the skill being improved.")
    skill_name: str = Field(default="", description="Name of the skill being improved.")
    status: str = Field(
        default="proposed", description="Lifecycle status: 'proposed' or 'applied'."
    )
    confidence: float = Field(default=0.0, description="Model confidence in the proposed change.")
    rationale: str = Field(default="", description="Why the change was proposed.")
    model_name: str = Field(default="", description="LLM that generated the proposal.")
    old_procedure: str = Field(default="", description="Current skill procedure (before).")
    proposed_procedure: str = Field(default="", description="Proposed skill procedure (after).")
    runs_used: list[str] = Field(
        default_factory=list, description="SkillRun ids whose failures motivated the proposal."
    )
    dataset_scope: list[str] = Field(
        default_factory=list, description="Dataset UUIDs this proposal is scoped to."
    )


class ErrorResponse(BaseModel):
    """Generic API error response."""

    error: str


def get_proposals_router() -> APIRouter:
    router = APIRouter()

    async def _authorized_dataset(dataset_id: UUID, user: User, permission: str):
        """Return the authorized dataset or raise PermissionDeniedError."""
        datasets = await get_authorized_existing_datasets([dataset_id], permission, user)
        if not datasets:
            raise PermissionDeniedError(message="Not authorized for this dataset")
        return datasets[0]

    @router.get(
        "/{proposal_id}",
        response_model=ProposalDetail,
        responses={
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def get_skill_proposal(
        proposal_id: str,
        dataset_id: UUID = Query(
            ...,
            description=(
                "Dataset UUID the proposal is scoped to. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Return one skill-improvement proposal with its before/after procedures."""
        send_telemetry(
            "Skill Proposal Get API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/proposals/{proposal_id}",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.memify.skill_improvement import get_proposal

        try:
            dataset = await _authorized_dataset(dataset_id, user, "read")
            proposal = await get_proposal(proposal_id, dataset=dataset, user=user)
            if proposal is None:
                return JSONResponse(status_code=404, content={"error": "Proposal not found"})
            return ProposalDetail(
                proposal_id=proposal.proposal_id,
                skill_id=proposal.skill_id,
                skill_name=proposal.skill_name,
                status=proposal.status,
                confidence=proposal.confidence,
                rationale=proposal.rationale,
                model_name=proposal.model_name,
                old_procedure=proposal.old_procedure,
                proposed_procedure=proposal.proposed_procedure,
                runs_used=list(proposal.runs_used or []),
                dataset_scope=list(proposal.dataset_scope or []),
            )
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except Exception as exc:
            logger.error("get proposal failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=409, content={"error": "Failed to fetch proposal"})

    return router
