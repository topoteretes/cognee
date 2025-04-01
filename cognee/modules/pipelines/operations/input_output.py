from typing import Any
from pydantic import BaseModel

from ..tasks import Task


class MergeInputs(BaseModel):
    outputs: list[Any]


def merge_inputs(*args):
    return MergeInputs(outputs=args)


def get_input_tasks(tasks: list[Task]):
    input_tasks = []

    for task in tasks:
        if isinstance(task, MergeInputs):
            input_tasks.extend(task.outputs)
        else:
            input_tasks.append(task)

    return input_tasks


def get_input_results(results, task: Task):
    input_results = []

    for task_dependency in task.task_config.inputs:
        if isinstance(task_dependency, MergeInputs):
            inputs = []
            input_results.append(inputs)
            for output in task_dependency.outputs:
                inputs.extend(results[output])
        else:
            input_results.append(results[task_dependency])

    return input_results
