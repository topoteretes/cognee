import inspect
from functools import wraps

from cognee.modules.users.models.User import User
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.data.methods.check_dataset_name import check_dataset_name


def validate_pipeline_inputs(pipeline_generator):
    @wraps(pipeline_generator)
    async def wrapper(*args, **kwargs):
        sig = inspect.signature(pipeline_generator)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        if "tasks" in bound_args.arguments:
            tasks = bound_args.arguments["tasks"]
            if not isinstance(tasks, list):
                raise ValueError(f"tasks must be a list, got {type(tasks).__name__}")

            for task in tasks:
                if not isinstance(task, Task):
                    raise ValueError(
                        f"tasks must be a list of Task instances, got {type(task).__name__} in the list"
                    )

        if "user" in bound_args.arguments:
            user = bound_args.arguments["user"]
            if not isinstance(user, User):
                raise ValueError(f"user must be an instance of User, got {type(user).__name__}")

        if "dataset" in bound_args.arguments:
            dataset = bound_args.arguments["dataset"]
            if not isinstance(dataset, Dataset):
                raise ValueError(
                    f"dataset must be an instance of Dataset, got {type(dataset).__name__}"
                )
            check_dataset_name(dataset.name)

        if "datasets" in bound_args.arguments:
            datasets = bound_args.arguments["datasets"]
            if not isinstance(datasets, list):
                raise ValueError(f"datasets must be a list, got {type(datasets).__name__}")

            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise ValueError(
                        f"datasets must be a list of Dataset instances, got {type(dataset).__name__} in the list"
                    )
                check_dataset_name(dataset.name)

        async for run_info in pipeline_generator(*args, **kwargs):
            yield run_info

    return wrapper
