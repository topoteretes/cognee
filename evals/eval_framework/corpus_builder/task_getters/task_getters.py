from enum import Enum
from typing import Type
from evals.eval_framework.corpus_builder.task_getters.default_task_getter import DefaultTaskGetter


class TaskGetters(Enum):
    """Enum mapping task getter types to their respective classes."""

    DEFAULT = ("Default", DefaultTaskGetter)
    # CUSTOM = ("Custom", CustomTaskGetter)

    def __new__(cls, getter_name: str, getter_class: Type):
        obj = object.__new__(cls)
        obj._value_ = getter_name
        obj.getter_class = getter_class
        return obj

    def __str__(self):
        return self.value
