from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Depends
from pydantic import Field
from typing import List, Optional, Union, Literal

from cognee.api.DTO import InDTO
from cognee.modules.improve import ImproveResult
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.exceptions import CogneeApiError

logger = get_logger()


class ImprovePayloadDTO(InDTO):
    extraction_tasks: Optional[List[str]] = Field(default=None, examples=[[]])
    enrichment_tasks: Optional[List[str]] = Field(default=None, examples=[[]])
    data: Optional[str] = Field(default=None)
    dataset_name: Optional[str] = Field(default=None)
    dataset_id: Union[UUID, Literal[""], None] = Field(default=None, examples=[""])
    node_name: Optional[List[str]] = Field(default=None, examples=[[]])
    run_in_background: Optional[bool] = Field(default=False)
    build_global_context_index: Optional[bool] = Field(default=False)
    build_truth_subspace: Optional[bool] = Field(default=False)
    # Learning rate for the feedback-weight stage. Omitted (None) means the
    # server's IMPROVE_FEEDBACK_ALPHA applies; the stage's formula is fixed,
    # only the rate is tunable.
    feedback_alpha: Optional[float] = Field(default=None, gt=0, le=1)
    # Session IDs to bridge into the permanent graph. Without them the
    # session-kind stages (feedback weights, Q&A / trace persistence,
    # distillation, preferences, truth subspace) are skipped with
    # ``no_session_ids`` and only the graph-kind stages can run.
    session_ids: Optional[List[str]] = Field(default=None, examples=[[]])


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
        - **run_in_background** (Optional[bool]): Run the whole chain as one background
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
        - **409 Conflict**: The fatal `persist_session_qa` stage failed, or another
          error aborted the run
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
        except CogneeApiError:
            # Cognee errors carry their own status code and actionable message;
            # the global handler in cognee/api/client.py returns them.
            raise
        except Exception as error:
            logger.error("Improve endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during graph improvement."},
            )

    return router
