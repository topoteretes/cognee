from enum import Enum
from typing import Callable, Awaitable, List
from cognee.api.v1.cognify.cognify_v2 import get_default_tasks
from cognee.modules.pipelines.tasks.Task import Task
from evals.eval_framework.corpus_builder.task_getters.get_cascade_graph_tasks import (
    get_cascade_graph_tasks,
)


class TaskGetters(Enum):
    """Enum mapping task getter types to their respective functions."""

    DEFAULT = ("Default", get_default_tasks)
    CASCADE_GRAPH = ("CascadeGraph", get_cascade_graph_tasks)

    def __new__(cls, getter_name: str, getter_func: Callable[..., Awaitable[List[Task]]]):
        obj = object.__new__(cls)
        obj._value_ = getter_name
        obj.getter_func = getter_func
        return obj

    def __str__(self):
        return self.value
