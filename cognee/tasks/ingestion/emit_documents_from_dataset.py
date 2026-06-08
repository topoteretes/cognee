from typing import Any, AsyncGenerator
from uuid import UUID

from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("emit_documents_from_dataset")


async def emit_documents_from_dataset(
    _trigger: Any,
    dataset_id: UUID,
    user: User,
    permission_type: str = "read",
) -> AsyncGenerator[Data, None]:
    """Stream Data rows from a dataset as a pipeline source.

    Mirrors the implicit ``get_dataset_data`` that ``run_pipeline_per_dataset``
    does when no ``data`` argument is supplied, but exposed as a composable
    pipeline stage so custom pipelines can iterate a dataset's documents
    without having to start a fresh ``run_tasks`` call.

    The first positional argument is a pipeline trigger (whatever the upstream
    stage yielded — typically ignored).
    """
    authorized = await get_authorized_existing_datasets(
        datasets=[dataset_id], permission_type=permission_type, user=user
    )
    if not authorized:
        raise CogneeValidationError(
            message=(
                f"User (id: {user.id}) does not have {permission_type} access to "
                f"dataset {dataset_id}"
            ),
            log=False,
        )

    rows = await get_dataset_data(dataset_id=authorized[0].id)
    logger.info(f"Emitting {len(rows)} Data row(s) from dataset {authorized[0].id}")
    for row in rows:
        yield row
