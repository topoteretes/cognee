import inspect
from typing import Callable, Any, Union

from pydantic import BaseModel

from ..operations.needs import MergeNeeds

# TaskExecutable = Union[Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]], AsyncGenerator[Any, Any], Generator[Any, Any, Any]]


class TaskExecutionStarted(BaseModel):
    task: Callable


class TaskExecutionCompleted(BaseModel):
    task: Callable
    result: Any = None


class TaskExecutionErrored(BaseModel):
    task: Callable
    error: Exception

    model_config = {"arbitrary_types_allowed": True}


class TaskExecutionInfo(BaseModel):
    result: Any = None
    task: Callable


class TaskConfig(BaseModel):
    output_batch_size: int = 1
    needs: list[Union[Callable, MergeNeeds]] = []


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

        yield TaskExecutionStarted(
            task=self.executable,
        )

        try:
            if inspect.iscoroutinefunction(self.executable):  # Async function
                task_result = await self.executable(*combined_args, **combined_kwargs)

            elif inspect.isgeneratorfunction(self.executable):  # Generator
                result = []
                task_result = []

                for value in self.executable(*combined_args, **combined_kwargs):
                    result.append(value)  # Store the last yielded value
                    task_result.append(value)

                    if self.task_config.output_batch_size == 1:
                        yield TaskExecutionInfo(
                            result=value,
                            task=self.executable,
                        )
                    elif self.task_config.output_batch_size == len(result):
                        yield TaskExecutionInfo(
                            result=result,
                            task=self.executable,
                        )
                        result = []  # Reset for the next batch

            elif inspect.isasyncgenfunction(self.executable):  # Async Generator
                result = []
                task_result = []

                async for value in self.executable(*combined_args, **combined_kwargs):
                    result.append(value)  # Store the last yielded value
                    task_result.append(value)

                    if self.task_config.output_batch_size == 1:
                        yield TaskExecutionInfo(
                            result=value,
                            task=self.executable,
                        )
                    elif self.task_config.output_batch_size == len(result):
                        yield TaskExecutionInfo(
                            result=result,
                            task=self.executable,
                        )
                        result = []  # Reset for the next batch

            else:  # Regular function
                task_result = self.executable(*combined_args, **combined_kwargs)

            yield TaskExecutionCompleted(
                task=self.executable,
                result=task_result,
            )
        except Exception as error:
            yield TaskExecutionErrored(
                task=self.executable,
                error=error,
            )
