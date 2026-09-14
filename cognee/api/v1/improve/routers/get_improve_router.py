from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO
from cognee.exceptions import CogneeApiError
from cognee.modules.improve import ImproveResult
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry

logger = get_logger()


class ImprovePayloadDTO(InDTO):
    extraction_tasks: list[str] | None = Field(default=None, examples=[[]])
    enrichment_tasks: list[str] | None = Field(default=None, examples=[[]])
    data: str | None = Field(default=None)
    dataset_name: str | None = Field(default=None)
    dataset_id: UUID | Literal[""] | None = Field(default=None, examples=[""])
    node_name: list[str] | None = Field(default=None, examples=[[]])
    run_in_background: bool | None = Field(default=False)
    build_global_context_index: bool | None = Field(default=False)
    build_truth_subspace: bool | None = Field(default=False)
    # Learning rate for the feedback-weight stage. Omitted (None) means the
    # server's IMPROVE_FEEDBACK_ALPHA applies; the stage's formula is fixed,
    # only the rate is tunable.
    feedback_alpha: float | None = Field(default=None, gt=0, le=1)
    # Session IDs to bridge into the permanent graph. Without them the
    # session-kind stages (feedback weights, Q&A / trace persistence,
    # distillation, preferences, truth subspace) are skipped with
    # ``no_session_ids`` and only the graph-kind stages can run.
    session_ids: list[str] | None = Field(default=None, examples=[[]])


def get_improve_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=ImproveResult)
    @log_usage(function_name="POST /v1/improve", log_type="api_endpoint")
    async def improve(payload: ImprovePayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Run the self-improvement loop over a dataset and report what each stage did.

        The nine stages run in a fixed order; each first *gates* (declines work it
        cannot do under the current settings, with no LLM calls) and only then runs:
        `feedback_weights`, `persist_session_qa`, `persist_agent_traces`,
        `extract_agent_context`, `distill_sessions`, `update_user_preferences`,
        `build_truth_subspace`, `triplet_enrichment`, `global_context_index`.
        Stages 1-7 need `sessionIds`; stages 7 and 9 are opt-in via the `build*` flags.

        ## Request Parameters
        - **extraction_tasks** (Optional[List[str]]): Tasks for graph/data extraction.
        - **enrichment_tasks** (Optional[List[str]]): Tasks for graph enrichment.
        - **data** (Optional[str]): Custom input data. Uses existing graph when omitted.
        - **dataset_name** (Optional[str]): Dataset name.
        - **dataset_id** (Optional[UUID]): Dataset UUID.
        - **node_name** (Optional[List[str]]): Filter to specific named entities.
        - **run_in_background** (Optional[bool]): Run all stages as one background
          task and return immediately with `status == "running"` (default: False).
        - **build_global_context_index** (Optional[bool]): Build the global context index
          after enrichment (default: False).
        - **build_truth_subspace** (Optional[bool]): Build the truth subspace from the
          sessions' distilled learnings (default: False; needs `sessionIds` and a
          backend with truth state).
        - **feedback_alpha** (Optional[float]): Learning rate in (0, 1] for the
          feedback-weight stage. Omitted means the server's `IMPROVE_FEEDBACK_ALPHA`.
        - **sessionIds** (Optional[List[str]]): Session identifiers whose cached memory
          is bridged into the permanent graph.

        Either dataset_name or dataset_id must be provided.

        ## Response
        An `ImproveResult`: `status` (`completed`, `errored`, `skipped`, `running`) and
        one `stages[]` entry per stage, in order, each with `status`
        (`completed` / `already_completed` / `skipped` / `errored`), a `reason` when
        skipped, `counts`, `duration_ms` and the pipeline `run` when the stage is a
        pipeline. A non-fatal stage error is reported inside the body with a 200;
        inspect `status`.

        ## Error Codes
        - **400 Bad Request**: Neither dataset_id nor dataset_name provided
        - **409 Conflict**: The fatal `persist_session_qa` stage failed — the body
          carries the abort reason and the partial `improve_result` (what ran
          before the abort) — or a non-Cognee error aborted the run (body
          carries the reason). Other Cognee errors return their own status
          codes.
        """
        send_telemetry(
            "Improve API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/improve",
                "cognee_version": cognee_version,
            },
        )

        if not payload.dataset_id and not payload.dataset_name:
            raise HTTPException(
                status_code=400,
                detail="Either datasetId or datasetName must be provided.",
            )

        try:
            from cognee.api.v1.improve import improve as cognee_improve

            improve_kwargs = {}
            if payload.feedback_alpha is not None:
                improve_kwargs["feedback_alpha"] = payload.feedback_alpha

            improve_run = await cognee_improve(
                extraction_tasks=payload.extraction_tasks,
                enrichment_tasks=payload.enrichment_tasks,
                data=payload.data,
                dataset=payload.dataset_id if payload.dataset_id else payload.dataset_name,
                node_name=payload.node_name,
                session_ids=payload.session_ids,
                build_global_context_index=bool(payload.build_global_context_index),
                build_truth_subspace=bool(payload.build_truth_subspace),
                user=user,
                run_in_background=bool(payload.run_in_background),
                **improve_kwargs,
            )

            return improve_run
        except CogneeApiError as error:
            partial = getattr(error, "improve_result", None)
            if partial is not None:
                # The fatal-stage abort (_abort_run attaches the partial result):
                # the documented 409, with what ran before the abort in the body
                # — the exception alone would reach clients as a bare 500.
                logger.exception("Improve run aborted by its fatal stage")
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": str(getattr(error, "message", None) or error),
                        "improve_result": partial.model_dump(mode="json"),
                    },
                )
            # Other Cognee errors carry their own status code and actionable
            # message; the global handler in cognee/api/client.py returns them.
            raise
        except Exception:
            # Generic body on purpose: an unexpected exception's text can leak
            # internals; the details go to the server log. Config mistakes
            # (e.g. an IMPROVE_STAGES_DISABLED typo) fail loudly at startup
            # instead of reaching this handler per call.
            logger.exception("Improve endpoint error")
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during graph improvement."},
            )

    return router
