from typing import Union, Callable, Any, Coroutine, Generator, AsyncGenerator
import inspect


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
    task_type: str = None

    def __init__(self, executable, *args, task_config=None, **kwargs):
        self.executable = executable
        self.default_params = {"args": args, "kwargs": kwargs}

        if inspect.isasyncgenfunction(executable):
            self.task_type = "Async Generator"
        elif inspect.isgeneratorfunction(executable):
            self.task_type = "Generator"
        elif inspect.iscoroutinefunction(executable):
            self.task_type = "Coroutine"
        elif inspect.isfunction(executable):
            self.task_type = "Function"
        else:
            raise ValueError(f"Unsupported task type: {executable}")

        if task_config is not None:
            self.task_config = task_config

            if "batch_size" not in task_config:
                self.task_config["batch_size"] = 1

    def run(self, *args, **kwargs):
        """Execute the underlying task with given arguments."""
        combined_args = args + self.default_params["args"]
        combined_kwargs = {**self.default_params["kwargs"], **kwargs}

        return self.executable(*combined_args, **combined_kwargs)

    async def execute_async_generator(self, args, batch_size):
        """Execute async generator task and collect results in batches."""
        results = []
        async_iterator = self.run(*args)

        async for partial_result in async_iterator:
            results.append(partial_result)

            if len(results) == batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_generator(self, args, batch_size):
        """Execute generator task and collect results in batches."""
        results = []

        for partial_result in self.run(*args):
            results.append(partial_result)

            if len(results) == batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_coroutine(self, args):
        """Execute coroutine task and yield the result."""
        task_result = await self.run(*args)
        yield task_result

    async def execute_function(self, args):
        """Execute function task and yield the result."""
        task_result = self.run(*args)
        yield task_result

    async def execute(self, args, next_batch_size=None):
        """Execute the task based on its type and yield results with the next task's batch size."""
        if next_batch_size is None:
            next_batch_size = 1

        if self.task_type == "Async Generator":
            async for batch_results in self.execute_async_generator(args, next_batch_size):
                yield batch_results
        elif self.task_type == "Generator":
            async for batch_results in self.execute_generator(args, next_batch_size):
                yield batch_results
        elif self.task_type == "Coroutine":
            async for result in self.execute_coroutine(args):
                yield result
        elif self.task_type == "Function":
            async for result in self.execute_function(args):
                yield result
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
