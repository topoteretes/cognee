"""Task resolvers another process can rebuild.

A task resolver maps one data item to the task list it runs (see
``run_tasks``). Task lists hold callables, so they cannot be sent to another
process. A *portable* resolver instead exposes ``portable``: the name of a
factory registered here and the JSON arguments to call it with. The process
that receives them calls :func:`build_task_resolver` and gets the same lists.

A rebuilt resolver need not know every kind of item the original does. One that
does not exposes ``portable_item(data_item) -> bool``, and only the items it
answers True for are sent to another process (:func:`portable_item_check`).

A pipeline registers its factory where it builds its tasks, so importing the
pipeline's module is what makes its resolver rebuildable.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.models import User

TaskResolver = Callable[[Any], list[Task]]
TaskResolverFactory = Callable[..., Awaitable[TaskResolver]]

_factories: dict[str, TaskResolverFactory] = {}


def register_task_resolver_factory(name: str, factory: TaskResolverFactory) -> None:
    """Register ``factory`` as the builder of the portable resolver ``name``.

    The factory is called with ``user`` and the resolver's JSON arguments as
    keyword arguments.
    """
    _factories[name] = factory


def portable_item_check(task_resolver: Any) -> Callable[[Any], bool]:
    """Which of ``task_resolver``'s items another process can run: its own
    ``portable_item`` when it has one, otherwise all of them."""
    return getattr(task_resolver, "portable_item", lambda _data_item: True)


async def build_task_resolver(name: str, args: dict[str, Any], user: User) -> TaskResolver:
    """Rebuild the portable resolver ``name`` from its arguments."""
    if name not in _factories:
        raise KeyError(
            f"No portable task resolver named {name!r} is registered in this process. "
            f"Registered: {sorted(_factories)}"
        )
    return await _factories[name](user=user, **args)
