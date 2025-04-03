from typing import Any
from pydantic import BaseModel

from ..tasks import Task


class MergeNeeds(BaseModel):
    needs: list[Any]


def merge_needs(*args):
    return MergeNeeds(needs=args)


def get_task_needs(tasks: list[Task]):
    input_tasks = []

    for task in tasks:
        if isinstance(task, MergeNeeds):
            input_tasks.extend(task.needs)
        else:
            input_tasks.append(task)

    return input_tasks


def get_need_task_results(results, task: Task):
    input_results = []

    for task_dependency in task.task_config.needs:
        if isinstance(task_dependency, MergeNeeds):
            task_results = []
            max_result_length = 0

            for task_need in task_dependency.needs:
                result = results[task_need]
                task_results.append(result)

                if isinstance(result, tuple):
                    max_result_length = max(max_result_length, len(result))

            final_results = [[] for _ in range(max_result_length)]

            for result in task_results:
                if isinstance(result, tuple):
                    for i, value in enumerate(result):
                        final_results[i].extend(value)
                else:
                    final_results[0].extend(result)

            input_results.extend(final_results)
        else:
            result = results[task_dependency]

            if isinstance(result, tuple):
                input_results.extend(result)
            else:
                input_results.append(result)

    return input_results
