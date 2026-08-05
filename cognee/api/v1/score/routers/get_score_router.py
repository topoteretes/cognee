"""HTTP router for the tenant memory accuracy score.

Starts a score run and serves its result document. The document carries RAW
SIGNALS only — ``below_data_floor``, ``floor_reason``, ``schema_defined``, the
per-topic accuracies with their question counts, and the real questions whose
answers were not grounded. There is deliberately no call-to-action field: this
API never decides "upload more data" vs "define a schema", and it never averages
the synthetic correctness score together with the real-question groundedness
boolean. Thresholds and copy belong to the UI.
"""

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry

logger = get_logger()

# Strong references to in-flight background runs. asyncio keeps only a weak
# reference to a bare task, so without this the run can be garbage-collected
# mid-flight — same guard as cognee/api/v1/sync/sync.py.
_BACKGROUND_SCORE_TASKS: set[asyncio.Task] = set()

_NO_TENANT_ERROR = (
    "The authenticated user has no tenant. The memory accuracy score is a "
    "tenant-wide measurement and cannot be scoped without one."
)


class StartScoreRunPayload(BaseModel):
    """Request body for starting a memory accuracy score run."""

    dataset_id: UUID = Field(
        description=(
            "The dataset to score. Required: with per-dataset graph isolation there "
            "is no single tenant graph, and the dataset to measure cannot be "
            'inferred. Cloud passes its "default_dataset"; OSS typically '
            '"main_dataset".'
        ),
    )
    run_in_background: bool = Field(
        default=True,
        description=(
            "When true (default) the run is started in the background and only its "
            "run_id is returned. When false the request blocks until the run finishes "
            "and returns the full run document."
        ),
    )
    synthetic_target: int = Field(
        default=100,
        ge=0,
        description=(
            "Total synthetic questions to aim for, split across topics by how much "
            "real traffic each topic receives."
        ),
        examples=[100],
    )
    real_question_limit: int = Field(
        default=20,
        ge=0,
        description="How many of the tenant's most recent real questions to replay.",
        examples=[20],
    )


class ScoreRunTopic(BaseModel):
    """Per-topic aggregate of one score run."""

    topic: str | None = Field(default=None, description="Topic label from graph clustering.")
    accuracy: float | None = Field(
        default=None,
        description=(
            "Mean correctness score of this topic's SYNTHETIC questions. Null when none "
            "of them could be judged — null means unmeasured, not wrong."
        ),
    )
    synthetic_count: int = Field(default=0, ge=0, description="Synthetic questions in this topic.")
    real_count: int = Field(
        default=0,
        ge=0,
        description="Real questions from tenant traffic that landed in this topic.",
    )
    from_real_traffic: bool = Field(
        default=False,
        description="True when at least one real question landed in this topic.",
    )


class ScoreRunQuestion(BaseModel):
    """One answered and judged question of a score run."""

    text: str | None = Field(default=None, description="The question that was asked.")
    topic: str | None = Field(
        default=None,
        description="Topic label, for synthetic questions. Null for real questions.",
    )
    source: str | None = Field(
        default=None, description='Either "synthetic" or "real".', examples=["synthetic"]
    )
    answer: str | None = Field(default=None, description="Answer produced by cognee recall.")
    expected_answer: str | None = Field(
        default=None,
        description="Golden answer. Synthetic questions only; null for real questions.",
    )
    score: float | None = Field(
        default=None,
        description="Correctness score. Synthetic questions only; null for real questions.",
    )
    grounded: bool | None = Field(
        default=None,
        description=(
            "Whether the answer was supported by the retrieved context. Real questions "
            "only; null for synthetic questions."
        ),
    )
    reason: str | None = Field(default=None, description="The judge's explanation.")


class ScoreRunDocument(BaseModel):
    """Full result document of one memory accuracy score run."""

    run_id: str = Field(description="Identifier of the run.")
    status: str | None = Field(
        default=None,
        description=("INITIATED, RUNNING, COMPLETED, ERRORED or SKIPPED_INSUFFICIENT_DATA."),
        examples=["COMPLETED"],
    )
    dataset_id: str | None = Field(
        default=None,
        description=(
            "The dataset this score describes. Replayed real questions are "
            "tenant-wide, so on a multi-dataset tenant some of them may not belong "
            "to this dataset."
        ),
    )
    below_data_floor: bool = Field(
        default=False,
        description=(
            "True when the graph was too small to score. Raw signal: the reason is in "
            "floor_reason and what to do about it is the caller's decision."
        ),
    )
    floor_reason: str | None = Field(
        default=None,
        description="What the data looked like when the floor gate failed. Null otherwise.",
    )
    schema_defined: bool = Field(
        default=False,
        description="True when the graph carries a semantic type beyond the ingestion defaults.",
    )
    overall_accuracy: float | None = Field(
        default=None,
        description=(
            "Mean correctness over SYNTHETIC questions only. Real questions have no "
            "golden answer and are never folded into this number."
        ),
    )
    synthetic_question_count: int = Field(default=0, ge=0)
    real_question_count: int = Field(default=0, ge=0)
    created_at: str | None = Field(default=None, description="When the run was registered.")
    completed_at: str | None = Field(default=None, description="When the run stopped.")
    topics: list[ScoreRunTopic] = Field(default_factory=list)
    questions: list[ScoreRunQuestion] = Field(default_factory=list)
    ungrounded_real_questions: list[str] = Field(
        default_factory=list,
        description="Texts of the real questions whose answers were judged not grounded.",
    )


class ErrorResponse(BaseModel):
    """Generic API error response."""

    error: str


def get_score_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        response_model=None,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    @log_usage(function_name="POST /v1/score", log_type="api_endpoint")
    async def start_score_run(
        payload: StartScoreRunPayload,
        user: User = Depends(get_authenticated_user),
    ):
        """Start a memory accuracy score run over one of the tenant's datasets.

        Request body:
            dataset_id: REQUIRED, the dataset to score. With per-dataset graph
                isolation there is no single tenant graph, so this cannot be
                inferred — Cloud passes "default_dataset", OSS "main_dataset".
            run_in_background: true (default) returns ``{"run_id": ...}`` as soon
                as the run is registered; false blocks and returns the full run
                document.
            synthetic_target: total synthetic questions to aim for (default 100).
            real_question_limit: real questions to replay (default 20).

        Errors:
            400: the authenticated user has no tenant.
            404: the dataset does not exist or is not this tenant's.
            409: a run is already active for this tenant.
        """
        send_telemetry(
            "Memory Score API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/score",
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.memory_score.methods import (
            MemoryScoreDatasetNotFoundError,
            MemoryScoreRunInProgressError,
            build_memory_score_document,
            create_memory_score_run,
            find_active_memory_score_run,
            get_memory_score_questions,
            get_memory_score_run,
            resolve_memory_score_dataset,
            run_memory_score,
        )

        tenant_id = getattr(user, "tenant_id", None)
        if tenant_id is None:
            return JSONResponse(status_code=400, content={"error": _NO_TENANT_ERROR})

        request = payload

        try:
            # Validated before the run row is registered: in background mode the
            # row is written first, and a bad dataset id must not leave an
            # INITIATED run behind blocking the tenant's next attempt.
            await resolve_memory_score_dataset(tenant_id, request.dataset_id)

            active_run_id = await find_active_memory_score_run(tenant_id)
            if active_run_id is not None:
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": (
                            "A memory score run is already in progress for this tenant "
                            f"(run {active_run_id})."
                        )
                    },
                )

            if request.run_in_background:
                # Register the run first so the caller gets an id it can poll
                # immediately; run_memory_score claims this row rather than
                # creating a second one.
                run_id = await create_memory_score_run(tenant_id, request.dataset_id, user.id)
                task = asyncio.create_task(
                    run_memory_score(
                        tenant_id=tenant_id,
                        dataset_id=request.dataset_id,
                        triggered_by_user_id=user.id,
                        synthetic_target=request.synthetic_target,
                        real_question_limit=request.real_question_limit,
                        user=user,
                    )
                )
                _BACKGROUND_SCORE_TASKS.add(task)
                task.add_done_callback(_BACKGROUND_SCORE_TASKS.discard)
                return {"run_id": str(run_id)}

            run_id = await run_memory_score(
                tenant_id=tenant_id,
                dataset_id=request.dataset_id,
                triggered_by_user_id=user.id,
                synthetic_target=request.synthetic_target,
                real_question_limit=request.real_question_limit,
                user=user,
            )
            run = await get_memory_score_run(run_id)
            if run is None:
                return JSONResponse(
                    status_code=500, content={"error": "Memory score run could not be read back"}
                )
            questions = await get_memory_score_questions(run_id)
            return build_memory_score_document(run, questions)
        except MemoryScoreDatasetNotFoundError as error:
            return JSONResponse(status_code=404, content={"error": error.message})
        except MemoryScoreRunInProgressError as error:
            return JSONResponse(status_code=409, content={"error": error.message})
        except CogneeApiError:
            raise
        except Exception as error:
            logger.error("memory score run failed to start: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500, content={"error": "Failed to start memory score run"}
            )

    # Registered BEFORE /{run_id}: FastAPI matches in declaration order, and a
    # literal path declared after the parameterised one would be swallowed by it.
    @router.get(
        "/latest",
        response_model=ScoreRunDocument,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def get_latest_score_run(user: User = Depends(get_authenticated_user)):
        """Return the most recent memory accuracy score run for the caller's tenant.

        Errors:
            400: the authenticated user has no tenant.
            404: the tenant has never been scored.
        """
        send_telemetry(
            "Memory Score API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/score/latest",
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.memory_score.methods import (
            build_memory_score_document,
            get_latest_memory_score_run,
            get_memory_score_questions,
        )

        tenant_id = getattr(user, "tenant_id", None)
        if tenant_id is None:
            return JSONResponse(status_code=400, content={"error": _NO_TENANT_ERROR})

        try:
            run = await get_latest_memory_score_run(tenant_id)
            if run is None:
                return JSONResponse(
                    status_code=404, content={"error": "No memory score run for this tenant"}
                )
            questions = await get_memory_score_questions(run.id)
            return build_memory_score_document(run, questions)
        except CogneeApiError:
            raise
        except Exception as error:
            logger.error("latest memory score run failed: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500, content={"error": "Failed to read the memory score run"}
            )

    @router.get(
        "/{run_id}",
        response_model=ScoreRunDocument,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def get_score_run(run_id: UUID, user: User = Depends(get_authenticated_user)):
        """Return one memory accuracy score run document.

        A run belonging to another tenant is reported as 404 rather than 403, so
        the endpoint does not confirm that the id exists.

        Errors:
            400: the authenticated user has no tenant.
            404: no such run for this tenant.
        """
        send_telemetry(
            "Memory Score API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/score/{run_id}",
                "run_id": str(run_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.memory_score.methods import (
            build_memory_score_document,
            get_memory_score_questions,
            get_memory_score_run,
        )

        tenant_id = getattr(user, "tenant_id", None)
        if tenant_id is None:
            return JSONResponse(status_code=400, content={"error": _NO_TENANT_ERROR})

        try:
            run = await get_memory_score_run(run_id)
            # str() on both sides: a UUID column comes back as UUID or str
            # depending on the configured driver.
            if run is None or str(run.tenant_id) != str(tenant_id):
                return JSONResponse(
                    status_code=404, content={"error": "Memory score run not found"}
                )
            questions = await get_memory_score_questions(run_id)
            return build_memory_score_document(run, questions)
        except CogneeApiError:
            raise
        except Exception as error:
            logger.error("memory score run read failed: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500, content={"error": "Failed to read the memory score run"}
            )

    return router
