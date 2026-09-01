from typing import List

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from typing_extensions import Annotated

from cognee.api.v1.validate.validate import ValidationReport, ValidationStatus, validate
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_validate_router() -> APIRouter:
    validate_router = APIRouter()

    @validate_router.get("", response_model=ValidationReport)
    async def run_validate(
        dataset: Annotated[
            List[str],
            Query(
                description=("Dataset name(s) to validate. Omit to validate the default dataset."),
            ),
        ] = [DEFAULT_DATASET_NAME],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Cross-check the graph and vector stores of a dataset for consistency.

        Detects orphaned edges (referencing deleted nodes), nodes whose id no
        longer matches cognee's own dedup contract, and nodes present in the
        graph but missing from the vector index (unreachable by semantic
        search). Read-only — never modifies any store.

        ## Query Parameters
        - **dataset** (List[str]): Dataset name(s) to validate. Omit to validate the default
          dataset.
        """
        try:
            report = await validate(dataset=dataset, user=user)
            status_code = 503 if report.status == ValidationStatus.UNHEALTHY else 200
            return JSONResponse(status_code=status_code, content=report.model_dump(mode="json"))
        except Exception as error:
            logger.error("validate() failed: %s", error, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"status": "error", "reason": "validation failed; see server logs."},
            )

    return validate_router
