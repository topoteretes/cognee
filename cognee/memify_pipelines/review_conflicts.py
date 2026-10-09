"""Review dataset facts through the normal memify lock and database context."""

from datetime import datetime
from uuid import UUID

from cognee import memify
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.improve.result import first_run_info
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.review_conflicts.read_facts import read_entity_facts
from cognee.tasks.memify.review_conflicts.review_facts import review_entities
from cognee.tasks.memify.review_conflicts.write_conflicts import (
    ReviewWriteState,
    write_review_batch,
)

logger = get_logger("review_conflicts")


async def _writable_dataset_id(dataset: str | UUID, user: User) -> UUID:
    """Resolve the dataset the caller named, refusing one they cannot write to."""
    datasets = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )
    if not datasets:
        raise CogneeValidationError(message=f"No write access to dataset: {dataset}", log=False)
    return datasets[0].id


def _report_unreviewed_entities(result, state: ReviewWriteState) -> None:
    """Carry the entities no call could review onto the completed run's record.

    The improve stage reads them back off the payload and asks for them again on
    the next run. An errored run is retried whole, so it needs no such list.
    """
    run_info = first_run_info(result)
    if isinstance(run_info, PipelineRunCompleted) and state.unreviewed_entity_ids:
        logger.warning("Conflict review could not review entities: %s", state.unreviewed_entity_ids)
        run_info.payload = {"unreviewed_entity_ids": state.unreviewed_entity_ids}


async def review_conflicts_pipeline(
    dataset: str | UUID,
    user: User,
    *,
    entity_ids: list[str] | None = None,
    since: datetime | None = None,
):
    """Read this dataset's facts, review them in bounded LLM calls, write each verdict."""
    dataset_id = await _writable_dataset_id(dataset, user)
    state = ReviewWriteState()
    result = await memify(
        extraction_tasks=[
            Task(read_entity_facts, entity_ids=entity_ids, since=since, needs_llm=False)
        ],
        enrichment_tasks=[
            Task(review_entities),
            Task(write_review_batch, batch_size=1, state=state, needs_llm=False),
        ],
        data=[{}],
        dataset=dataset_id,
        user=user,
    )
    _report_unreviewed_entities(result, state)
    return result
