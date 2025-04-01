import inspect
from typing import Callable, Any, Union

from pydantic import BaseModel

from ..operations.input_output import MergeInputs

# TaskExecutable = Union[Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]], AsyncGenerator[Any, Any], Generator[Any, Any, Any]]


class TaskExecutionResult(BaseModel):
    is_done: bool
    result: Any = None


class TaskConfig(BaseModel):
    output_batch_size: int = 1
    inputs: list[Union[Callable, MergeInputs]] = []


class Task:
    task_config: TaskConfig
    default_params: dict[str, Any] = {}

    def __init__(self, executable, *args, task_config: TaskConfig = None, **kwargs):
        self.executable = executable
        self.default_params = {"args": args, "kwargs": kwargs}
        self.result = None

        self.task_config = task_config or TaskConfig()

    async def run(self, *args, **kwargs):
        combined_args = args + self.default_params["args"]
        combined_kwargs = {
            **self.default_params["kwargs"],
            **kwargs,
        }

        if inspect.iscoroutinefunction(self.executable):  # Async function
            result = await self.executable(*combined_args, **combined_kwargs)
            yield TaskExecutionResult(
                is_done=True,
                result=result,
            )

        elif inspect.isgeneratorfunction(self.executable):  # Generator
            result = []

            for value in self.executable(*combined_args, **combined_kwargs):
                result.append(value)  # Store the last yielded value

                if self.task_config.output_batch_size == 1:
                    yield TaskExecutionResult(
                        is_done=False,
                        result=value,
                    )
                elif self.task_config.output_batch_size == len(result):
                    yield TaskExecutionResult(
                        is_done=False,
                        result=result,
                    )
                    result = []  # Reset for the next batch

            yield TaskExecutionResult(
                is_done=True,
                result=result,
            )

        elif inspect.isasyncgenfunction(self.executable):  # Async Generator
            result = []

            async for value in self.executable(*combined_args, **combined_kwargs):
                result.append(value)  # Store the last yielded value

                if self.task_config.output_batch_size == 1:
                    yield TaskExecutionResult(
                        is_done=False,
                        result=value,
                    )
                elif self.task_config.output_batch_size == len(result):
                    yield TaskExecutionResult(
                        is_done=False,
                        result=result,
                    )
                    result = []  # Reset for the next batch

            yield TaskExecutionResult(
                is_done=True,
                result=result,
            )

        else:  # Regular function
            result = self.executable(*combined_args, **combined_kwargs)

            yield TaskExecutionResult(
                is_done=True,
                result=result,
            )
