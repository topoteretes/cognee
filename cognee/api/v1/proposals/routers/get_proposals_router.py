"""HTTP router for reviewing proposals.

Two families share the prefix:

* ``/ontology`` — governed-model proposals (``OntologyProposal``: mapping,
  definition_conflict, ontology_extension) drafted by the ``ontology_proposals``
  improve stage. List, inspect, and ratify (apply / reject) them here; applying
  writes the change with ``ontology_valid=True`` and the caller as ``ratified_by``.
* ``/{proposal_id}`` — a read-only view of a stored ``SkillImprovementProposal`` so a
  caller can inspect the before/after procedure, rationale and confidence *before*
  deciding whether to apply it (apply still happens via
  ``POST /api/v1/remember/entry`` with ``skill_improvement``).

Mirrors the skills router: explicit response schema + caller-scoped dataset
authorization.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.exceptions import CogneeApiError
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


class OntologyProposalDetail(BaseModel):
    """One governed-model proposal awaiting (or past) ratification."""

    proposal_id: str = Field(description="Stable proposal identifier.")
    kind: str = Field(description="mapping | definition_conflict | ontology_extension.")
    status: str = Field(default="proposed", description="proposed | applied | rejected.")
    subject_id: str = Field(default="", description="Graph node the proposal is about.")
    subject_name: str = Field(default="")
    subject_kind: str = Field(default="", description="Node type of the subject.")
    concept_name: str = Field(default="", description="Ontology concept (mapping / extension).")
    concept_uri: str | None = Field(default=None)
    concept_category: str = Field(default="", description="classes | properties.")
    counterpart_id: str = Field(default="", description="Other node (definition_conflict).")
    counterpart_name: str = Field(default="")
    evidence: list[str] = Field(default_factory=list)
    rationale: str = Field(default="")
    confidence: float = Field(default=0.0)
    occurrences: int = Field(default=0)
    proposed_by: str = Field(default="heuristic", description="llm | heuristic.")
    model_name: str = Field(default="")
    ratified_by: str | None = Field(default=None)
    ratified_at: str | None = Field(default=None)
    resolution: str = Field(default="")
    dataset_scope: list[str] = Field(default_factory=list)

    @classmethod
    def from_proposal(cls, proposal) -> "OntologyProposalDetail":
        return cls(
            proposal_id=proposal.proposal_id,
            kind=proposal.kind,
            status=proposal.status,
            subject_id=proposal.subject_id,
            subject_name=proposal.subject_name,
            subject_kind=proposal.subject_kind,
            concept_name=proposal.concept_name,
            concept_uri=proposal.concept_uri,
            concept_category=proposal.concept_category,
            counterpart_id=proposal.counterpart_id,
            counterpart_name=proposal.counterpart_name,
            evidence=list(proposal.evidence or []),
            rationale=proposal.rationale,
            confidence=proposal.confidence,
            occurrences=proposal.occurrences,
            proposed_by=proposal.proposed_by,
            model_name=proposal.model_name,
            ratified_by=proposal.ratified_by,
            ratified_at=proposal.ratified_at,
            resolution=proposal.resolution,
            dataset_scope=list(proposal.dataset_scope or []),
        )


class RatifyProposalRequest(BaseModel):
    """Body of ``POST /ontology/{proposal_id}/ratify``."""

    decision: str = Field(description="'apply' writes the change; 'reject' only records it.")
    resolution: str = Field(
        default="",
        description="Free-text note; for definition_conflict, which definition stands.",
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

    _dataset_query = Query(
        ...,
        description=(
            "Dataset UUID the proposals are scoped to. "
            "List your datasets via GET /api/v1/datasets to find it."
        ),
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )

    @router.get(
        "/ontology",
        response_model=list[OntologyProposalDetail],
        responses={403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    async def list_ontology_proposals_endpoint(
        dataset_id: UUID = _dataset_query,
        kind: str | None = Query(
            default=None, description="mapping | definition_conflict | ontology_extension."
        ),
        status: str | None = Query(
            default="proposed", description="proposed (default) | applied | rejected | all."
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """List governed-model proposals for a dataset, highest confidence first.

        ## Query Parameters
        - **dataset_id** (UUID): Dataset the proposals are scoped to.
        - **kind** (str, optional): Filter by proposal kind.
        - **status** (str, optional): ``proposed`` by default; ``all`` lifts the filter.
        """
        send_telemetry(
            "Ontology Proposals List API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/proposals/ontology",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )
        from cognee.modules.proposals import list_ontology_proposals

        try:
            dataset = await _authorized_dataset(dataset_id, user, "read")
            proposals = await list_ontology_proposals(
                dataset=dataset,
                user=user,
                kind=kind,
                status=None if status in (None, "all") else status,
            )
            return [OntologyProposalDetail.from_proposal(proposal) for proposal in proposals]
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("list ontology proposals failed")
            return JSONResponse(status_code=409, content={"error": "Failed to list proposals"})

    @router.get(
        "/ontology/{proposal_id}",
        response_model=OntologyProposalDetail,
        responses={
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def get_ontology_proposal_endpoint(
        proposal_id: str,
        dataset_id: UUID = _dataset_query,
        user: User = Depends(get_authenticated_user),
    ):
        """Return one governed-model proposal with its evidence and rationale."""
        send_telemetry(
            "Ontology Proposal Get API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/proposals/ontology/{proposal_id}",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )
        from cognee.modules.proposals import get_ontology_proposal

        try:
            dataset = await _authorized_dataset(dataset_id, user, "read")
            proposal = await get_ontology_proposal(proposal_id, dataset=dataset, user=user)
            if proposal is None:
                return JSONResponse(status_code=404, content={"error": "Proposal not found"})
            return OntologyProposalDetail.from_proposal(proposal)
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("get ontology proposal failed")
            return JSONResponse(status_code=409, content={"error": "Failed to fetch proposal"})

    @router.post(
        "/ontology/{proposal_id}/ratify",
        response_model=OntologyProposalDetail,
        responses={
            400: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def ratify_ontology_proposal_endpoint(
        proposal_id: str,
        payload: RatifyProposalRequest,
        dataset_id: UUID = _dataset_query,
        user: User = Depends(get_authenticated_user),
    ):
        """Apply or reject a governed-model proposal.

        ``apply`` writes the change the proposal describes — a ``realizes`` edge, a
        resolved ``contradicts`` edge, or an ``EntityType`` marked ``ontology_valid`` —
        stamped with the caller as ``ratified_by``. ``reject`` records the decision so
        the same finding is not proposed again. Requires write permission.
        """
        send_telemetry(
            "Ontology Proposal Ratify API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/proposals/ontology/{proposal_id}/ratify",
                "dataset_id": str(dataset_id),
                "decision": payload.decision,
                "cognee_version": cognee_version,
            },
        )
        from cognee.modules.proposals import ProposalNotApplicableError, ratify_ontology_proposal

        if payload.decision not in ("apply", "reject"):
            return JSONResponse(
                status_code=400, content={"error": "decision must be 'apply' or 'reject'"}
            )
        try:
            dataset = await _authorized_dataset(dataset_id, user, "write")
            proposal = await ratify_ontology_proposal(
                proposal_id,
                dataset=dataset,
                user=user,
                decision=payload.decision,
                resolution=payload.resolution,
            )
            if proposal is None:
                return JSONResponse(status_code=404, content={"error": "Proposal not found"})
            return OntologyProposalDetail.from_proposal(proposal)
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403, content={"error": "Not authorized for this dataset"}
            )
        except ProposalNotApplicableError as error:
            return JSONResponse(status_code=409, content={"error": str(error)})
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("ratify ontology proposal failed")
            return JSONResponse(status_code=409, content={"error": "Failed to ratify proposal"})

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
        """Return one skill-improvement proposal with its before/after procedures.

        ## Path Parameters
        - **proposal_id** (str): ID of the skill-improvement proposal.

        ## Query Parameters
        - **dataset_id** (UUID): Dataset UUID the proposal is scoped to. List your datasets via GET
          /api/v1/datasets to find it.
        """
        send_telemetry(
            "Skill Proposal Get API Endpoint Invoked",
            user,
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
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("get proposal failed")
            return JSONResponse(status_code=409, content={"error": "Failed to fetch proposal"})

    return router
