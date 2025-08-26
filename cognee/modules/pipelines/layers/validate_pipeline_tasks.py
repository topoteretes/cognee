from ..tasks.task import Task
from ..exceptions.tasks import WrongTaskTypeError


def validate_pipeline_tasks(tasks: list[Task]):
    """
    Validates the tasks argument to ensure it is a list of Task class instances.

    Args:
        tasks (list[Task]): The list of tasks to be validated.
    """

    if not isinstance(tasks, list):
        raise WrongTaskTypeError(f"tasks argument must be a list, got {type(tasks).__name__}.")

    for task in tasks:
        if not isinstance(task, Task):
            raise WrongTaskTypeError(
                f"tasks argument must be a list of Task class instances, got {type(task).__name__} in the list."
            )
