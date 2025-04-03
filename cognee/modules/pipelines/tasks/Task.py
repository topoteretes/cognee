import inspect
from typing import Callable, Any, Union

from pydantic import BaseModel

from ..tasks.types import TaskExecutable
from ..operations.needs import MergeNeeds
from ..exceptions import TaskExecutionException


class TaskExecutionStarted(BaseModel):
    task: Callable


class TaskExecutionCompleted(BaseModel):
    task: Callable
    result: Any = None


class TaskExecutionErrored(BaseModel):
    task: TaskExecutable
    error: TaskExecutionException

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
                end_result = await self.executable(*combined_args, **combined_kwargs)

            elif inspect.isgeneratorfunction(self.executable):  # Generator
                task_result = []
                end_result = []

                for value in self.executable(*combined_args, **combined_kwargs):
                    task_result.append(value)  # Store the last yielded value
                    end_result.append(value)

                    if self.task_config.output_batch_size == 1:
                        yield TaskExecutionInfo(
                            result=value,
                            task=self.executable,
                        )
                    elif self.task_config.output_batch_size == len(task_result):
                        yield TaskExecutionInfo(
                            result=task_result,
                            task=self.executable,
                        )
                        task_result = []  # Reset for the next batch

                # Yield any remaining items in the final batch if it's not empty
                if task_result and self.task_config.output_batch_size > 1:
                    yield TaskExecutionInfo(
                        result=task_result,
                        task=self.executable,
                    )

            elif inspect.isasyncgenfunction(self.executable):  # Async Generator
                task_result = []
                end_result = []

                async for value in self.executable(*combined_args, **combined_kwargs):
                    task_result.append(value)  # Store the last yielded value
                    end_result.append(value)

                    if self.task_config.output_batch_size == 1:
                        yield TaskExecutionInfo(
                            result=value,
                            task=self.executable,
                        )
                    elif self.task_config.output_batch_size == len(task_result):
                        yield TaskExecutionInfo(
                            result=task_result,
                            task=self.executable,
                        )
                        task_result = []  # Reset for the next batch

                # Yield any remaining items in the final batch if it's not empty
                if task_result and self.task_config.output_batch_size > 1:
                    yield TaskExecutionInfo(
                        result=task_result,
                        task=self.executable,
                    )
            else:  # Regular function
                end_result = self.executable(*combined_args, **combined_kwargs)

            yield TaskExecutionCompleted(
                task=self.executable,
                result=end_result,
            )

        except Exception as error:
            import traceback

            error_details = TaskExecutionException(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )

            yield TaskExecutionErrored(
                task=self.executable,
                error=error_details,
            )
