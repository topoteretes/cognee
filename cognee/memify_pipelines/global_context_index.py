from typing import Literal
from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.memify import run_memify_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User
from cognee.tasks.memify.global_context_index import (
    build_document_index,
    extract_global_context_index_input,
    load_dataset_content,
    update_global_context_index,
)
from cognee.tasks.memify.global_context_index.bucketing_strategy import BucketingStrategyName

GlobalContextMethod = Literal["from_entities", "llm"]


def get_global_context_index_memify_tasks(
    method: GlobalContextMethod = "from_entities",
    link_documents: bool = True,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    rebuild: bool = False,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
):
    """
    Build the task pair for the explicit global context index pipeline.

    ``method="from_entities"`` is the default: one deterministic bucket per
    document plus one root, built by counting entities, with no LLM call.
    ``link_documents`` also applies only to it; every clustering argument below
    applies only to ``method="llm"``.

    ``bucketing_strategy="vector"`` is the default and uses
    ``placement_distance_threshold`` for level-0 and upper-level placement.
    ``bucketing_strategy="graph"`` is experimental, uses ``min_overlap`` for
    level-0 graph placement, and falls back to vector bucketing for levels
    above 0. Switching between vector-built and graph-built indexes requires
    ``rebuild=True``.
    """
    if method == "from_entities":
        return (
            [Task(load_dataset_content, needs_llm=False)],
            [Task(build_document_index, needs_llm=False, link_documents=link_documents)],
        )
    if method != "llm":
        raise ValueError('method must be "from_entities" or "llm".')

    return (
        [Task(extract_global_context_index_input)],
        [
            Task(
                update_global_context_index,
                rebuild=rebuild,
                max_bucket_size=max_bucket_size,
                placement_distance_threshold=placement_distance_threshold,
                bucketing_strategy=bucketing_strategy,
                min_overlap=min_overlap,
            )
        ],
    )


async def global_context_index_pipeline(
    user: User,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    run_in_background: bool = False,
    method: GlobalContextMethod = "from_entities",
    link_documents: bool = True,
    max_bucket_size: int = 20,
    placement_distance_threshold: float = 0.5,
    rebuild: bool = False,
    bucketing_strategy: BucketingStrategyName = "vector",
    min_overlap: float = 0.05,
):
    """
    Build or update the global context index for a dataset.

    ``method="from_entities"`` (the default) recomputes the whole index every
    run and deletes the index nodes it did not compute, so switching to it from
    ``method="llm"`` needs nothing. ``link_documents=False`` stops it writing
    the ``summarizes`` edge from each bucket to its document, which nothing
    reads; the bucket's ``document_id`` carries the same link. Switching back needs ``rebuild=True``: the
    ``llm`` path extends the buckets it finds, and document buckets are not its
    shape.

    Under ``method="llm"``, vector bucketing is the default. Graph bucketing is
    experimental and only applies to level-0 buckets; upper levels remain
    vector-based. Use ``rebuild=True`` when switching between vector and graph
    strategies.
    """
    extraction_tasks, enrichment_tasks = get_global_context_index_memify_tasks(
        method=method,
        link_documents=link_documents,
        max_bucket_size=max_bucket_size,
        placement_distance_threshold=placement_distance_threshold,
        rebuild=rebuild,
        bucketing_strategy=bucketing_strategy,
        min_overlap=min_overlap,
    )

    return await run_memify_tasks(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        dataset=dataset,
        data=[{}],
        user=user,
        run_in_background=run_in_background,
    )
