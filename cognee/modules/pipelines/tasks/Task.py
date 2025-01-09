from typing import Union, Callable, Any, Coroutine, Generator, AsyncGenerator


class Task:
    executable: Union[
        Callable[..., Any],
        Callable[..., Coroutine[Any, Any, Any]],
        Generator[Any, Any, Any],
        AsyncGenerator[Any, Any],
    ]
    task_config: dict[str, Any] = {
        "batch_size": 1,
    }
    default_params: dict[str, Any] = {}

    def __init__(self, executable, *args, task_config=None, **kwargs):
        self.executable = executable
        self.default_params = {"args": args, "kwargs": kwargs}

        if task_config is not None:
            self.task_config = task_config

            if "batch_size" not in task_config:
                self.task_config["batch_size"] = 1

    def run(self, *args, **kwargs):
        combined_args = args + self.default_params["args"]
        combined_kwargs = {**self.default_params["kwargs"], **kwargs}

        return self.executable(*combined_args, **combined_kwargs)
