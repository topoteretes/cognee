"""Deprecated import path of ``memify()`` (SDK-1115), kept for one release.

Until SDK-1115 this module held the public ``cognee.memify``. It now forwards
to the deprecated shim in ``cognee.api.v1.memify``, with the same signature
and behaviour. The internal runner the improve stages use is
``cognee.modules.memify.run_memify_tasks``.
"""

import warnings
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User


async def memify(
    extraction_tasks: Sequence[Task | str] | None = None,
    enrichment_tasks: Sequence[Task | str] | None = None,
    data: Any | None = None,
    dataset: str | UUID = DEFAULT_DATASET_NAME,
    user: User = None,
    node_type: type | None = NodeSet,
    node_name: list[str] | None = None,
    vector_db_config: dict | None = None,
    graph_db_config: dict | None = None,
    run_in_background: bool = False,
):
    """Deprecated: same as ``cognee.memify()``. Removed in the next release."""
    from cognee.api.v1.memify.memify import MEMIFY_DEPRECATION_MESSAGE, run_legacy_memify

    warnings.warn(MEMIFY_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    return await run_legacy_memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
        data=data,
        dataset=dataset,
        user=user,
        node_type=node_type,
        node_name=node_name,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        run_in_background=run_in_background,
    )
