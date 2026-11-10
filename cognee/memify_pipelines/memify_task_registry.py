"""Deprecated import path (SDK-1115), kept for one release.

Memify task-name resolution moved to ``cognee.api.v1.memify.memify`` together
with the deprecated ``memify()``. Removed in the next release.
"""

import warnings
from collections.abc import Sequence

from cognee.modules.pipelines.tasks.task import Task

warnings.warn(
    "cognee.memify_pipelines.memify_task_registry is deprecated and will be removed in "
    "the next release, together with cognee.memify().",
    FutureWarning,
    stacklevel=2,
)


def supported_memify_task_names() -> list[str]:
    """Return the sorted names of memify tasks selectable by name."""
    from cognee.api.v1.memify.memify import _task_factories

    return sorted(_task_factories())


def resolve_memify_tasks(tasks: Sequence[Task | str] | None) -> list[Task] | None:
    """Resolve Task instances and task names into Task instances.

    Passes ``None``/empty input through as ``None``, as before.
    """
    from cognee.api.v1.memify.memify import resolve_memify_tasks as _resolve

    return _resolve(tasks) or None
