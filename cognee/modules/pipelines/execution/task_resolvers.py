"""Task lists another process can rebuild, registered by name.

A task resolver maps one data item to the task list it runs (see
``run_tasks``). Task lists hold callables, so they cannot be sent to another
process. A pipeline registers a factory for its task lists under a name instead
(:func:`register_task_resolver_factory`), the way a database adapter is
registered under its provider's name. The process that owns a run names the
factory and the JSON arguments it built its tasks with (the resolver's
``portable``), and the process that runs some of the run's items calls the same
factory with them (:func:`build_task_resolver`) and gets the same lists.

Any pipeline can register one: :func:`registered_tasks` builds its tasks for the
process that owns the run, so both processes build them from one place.

A rebuilt resolver need not know every kind of item the original does. One that
does not exposes ``portable_item(data_item) -> bool``, and only the items it
answers True for are sent to another process (:func:`portable_item_check`).

A pipeline registers its factory where it builds its tasks, so importing the
pipeline's module is what makes its tasks rebuildable: a worker has to import
it before it runs a shard (cognee's own are registered by ``import cognee``).
"""

from collections.abc import Awaitable, Callable
from typing import Any

from cognee.modules.pipelines.tasks.task import Task, pipeline_needs_llm
from cognee.modules.users.models import User

TaskResolver = Callable[[Any], list[Task]]
TaskResolverFactory = Callable[..., Awaitable[TaskResolver | list[Task]]]

_factories: dict[str, TaskResolverFactory] = {}


def register_task_resolver_factory(name: str, factory: TaskResolverFactory) -> None:
    """Register ``factory`` as the builder of the task lists named ``name``.

    The factory is called with ``user`` and the JSON arguments as keyword
    arguments. It returns the task list every item runs, or a resolver that
    picks each item's list.
    """
    _factories[name] = factory


def portable_item_check(task_resolver: Any) -> Callable[[Any], bool]:
    """Which of ``task_resolver``'s items another process can run: its own
    ``portable_item`` when it has one, otherwise all of them."""
    return getattr(task_resolver, "portable_item", lambda _data_item: True)


async def _build(name: str, args: dict[str, Any], user: User) -> tuple[TaskResolver, bool]:
    """``(the task lists registered as name, whether any of their tasks calls the LLM)``.

    One list for every item says so by its tasks (``Task.needs_llm``). A resolver
    picks a list per item, so it says so itself (its ``needs_llm``); one that
    does not is taken to call it.
    """
    if name not in _factories:
        raise KeyError(
            f"No task resolver factory named {name!r} is registered in this process. "
            f"Registered: {sorted(_factories)}"
        )
    built = await _factories[name](user=user, **args)
    if callable(built):
        return built, getattr(built, "needs_llm", True)
    return (lambda _data_item: built), pipeline_needs_llm(built)


async def build_task_resolver(name: str, args: dict[str, Any], user: User) -> TaskResolver:
    """Build the task lists registered as ``name`` from their arguments."""
    resolve, _needs_llm = await _build(name, args, user)
    return resolve


class RegisteredTasks:
    """The task lists registered as ``name``, built with ``args``: a task
    resolver that says how another process rebuilds it (``portable``) and
    whether a run of it needs the LLM connection (``needs_llm``)."""

    def __init__(self, name: str, args: dict[str, Any], resolve: TaskResolver, needs_llm: bool):
        self.portable = (name, args)
        self.portable_item = portable_item_check(resolve)
        self.needs_llm = needs_llm
        self._resolve = resolve

    def __call__(self, data_item: Any) -> list[Task]:
        return self._resolve(data_item)


async def registered_tasks(name: str, user: User, **args: Any) -> RegisteredTasks:
    """The task lists registered as ``name``, built here with ``args`` (JSON
    values), to pass as the ``tasks`` of a run whose items may go to workers.

    A run given a plain task list always stays in its own process: nothing says
    how another process would get the same list.
    """
    resolve, needs_llm = await _build(name, args, user)
    return RegisteredTasks(name, args, resolve, needs_llm)
