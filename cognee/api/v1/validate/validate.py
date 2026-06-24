"""Dataset-level knowledge graph integrity validation."""

from typing import Any, Optional, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from .models import ValidationReport, ValidationStatus

logger = get_logger("validate")


async def validate(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    user: Any = None,
) -> ValidationReport:
    """Validate the integrity of a dataset's knowledge graph.

    Cross-checks the graph, vector, and relational stores to surface
    inconsistencies that would otherwise only show up as bad search
    results.

    Args:
        dataset: Dataset name or UUID. Requires read permission.
        user: User context. Resolved to default user when None.

    Returns:
        ValidationReport with status, issues list, and summary stats.

    Example::

        report = await cognee.validate("my_dataset")
        print(report.status)
        for issue in report.issues:
            print(f"[{issue.severity}] {issue.type}: {issue.detail}")
    """
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.users.methods import get_default_user

    from .checks import (
        check_dangling_edges,
        check_graph_vector_sync,
        check_isolated_nodes,
        check_uncognified_data,
    )

    await setup()

    if user is None:
        user = await get_default_user()

    dataset_id = await _resolve_dataset_id(dataset, user)

    report = ValidationReport(dataset=str(dataset))

    async with set_database_global_context_variables(dataset_id, user.id):
        graph_engine = await get_graph_engine()
        vector_engine = get_vector_engine()

        is_empty = await graph_engine.is_empty()
        if is_empty:
            report.summary["graph_nodes"] = 0
            report.summary["graph_edges"] = 0
            # Still check for uncognified data — graph being empty might be the problem
            await check_uncognified_data(report, dataset_id)
            return report

        await check_graph_vector_sync(report, graph_engine, vector_engine)
        await check_dangling_edges(report, graph_engine)
        await check_isolated_nodes(report, graph_engine)
        await check_uncognified_data(report, dataset_id)

    logger.info(
        "validate: dataset=%s status=%s issues=%d",
        dataset,
        report.status,
        len(report.issues),
    )

    return report


async def _resolve_dataset_id(dataset_ref: Union[str, UUID], user) -> UUID:
    if isinstance(dataset_ref, UUID):
        from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset

        ds = await get_authorized_dataset(user, dataset_ref, "read")
        if not ds:
            raise ValueError(f"Dataset {dataset_ref} not found or not accessible.")
        return ds.id

    from cognee.modules.data.methods import get_authorized_dataset_by_name

    ds = await get_authorized_dataset_by_name(dataset_ref, user, "read")
    return ds.id
