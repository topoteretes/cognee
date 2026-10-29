"""Review dataset facts through the normal memify lock and database context."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from cognee.exceptions import CogneeValidationError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.improve.result import first_run_info
from cognee.modules.memify import run_memify_tasks
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.review_conflicts.read_facts import read_entity_facts
from cognee.tasks.memify.review_conflicts.review_entities import review_entities
from cognee.tasks.memify.review_conflicts.write_conflicts import write_conflicts

logger = get_logger("review_conflicts")


@dataclass(frozen=True)
class ConflictReviewResult:
    """What one review run did."""

    # The memify run's return.
    run_result: Any
    # The entities a completed run failed to review, each once and sorted: the stage
    # stamps them so that the next run reviews them again.
    failed_entity_ids: list[str]
    # The entities a completed run skipped as too large for one call, each once and sorted.
    skipped_entity_ids: list[str] = field(default_factory=list)


async def _resolve_writable_dataset_id(dataset: str | UUID, user: User) -> UUID:
    """Resolve the dataset the caller named, refusing one they cannot write to."""
    datasets = await get_authorized_existing_datasets(
        user=user, datasets=[dataset], permission_type="write"
    )
    if not datasets:
        raise CogneeValidationError(message=f"No write access to dataset: {dataset}", log=False)
    return datasets[0].id


def _report_failed_entities(result, failed_entity_ids: list[str]) -> list[str]:
    """Return the entities a completed run could not review, each once and sorted, and log them.

    The improve stage stamps them so the next run reviews them again. An errored run
    is retried whole, so it reports none.
    """
    run_info = first_run_info(result)
    if not isinstance(run_info, PipelineRunCompleted):
        return []
    failed = sorted(set(failed_entity_ids))
    if failed:
        logger.warning("Conflict review could not review entities: %s", failed)
    return failed


async def review_conflicts_pipeline(
    dataset: str | UUID,
    user: User,
    *,
    entity_ids: list[str] | None = None,
    since: datetime | None = None,
) -> ConflictReviewResult:
    """Read this dataset's facts, review them in bounded LLM calls, write the conflicts found."""
    dataset_id = await _resolve_writable_dataset_id(dataset, user)
    # The runner returns only the run's status, so the reviewer reports the subjects
    # it could not review, and those too large for one call, by filling these lists.
    failed_entity_ids: list[str] = []
    skipped_entity_ids: list[str] = []
    result = await run_memify_tasks(
        extraction_tasks=[
            Task(read_entity_facts, entity_ids=entity_ids, since=since, needs_llm=False)
        ],
        enrichment_tasks=[
            Task(
                review_entities,
                failed_entity_ids=failed_entity_ids,
                skipped_entity_ids=skipped_entity_ids,
            ),
            Task(write_conflicts, batch_size=1, needs_llm=False),
        ],
        data=[{}],
        dataset=dataset_id,
        user=user,
    )
    # An errored run is retried whole, so it reports no skipped entity either.
    completed = isinstance(first_run_info(result), PipelineRunCompleted)
    return ConflictReviewResult(
        result,
        _report_failed_entities(result, failed_entity_ids),
        sorted(set(skipped_entity_ids)) if completed else [],
    )
