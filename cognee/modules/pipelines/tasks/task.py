from typing import Union, Callable, Any, Coroutine, Generator, AsyncGenerator
import inspect


def task_summary(template: str):
    """Decorator that attaches a human-readable summary template to a task function.

    The template should contain ``{n}`` as a placeholder for the result count.

    Example::

        @task_summary("Classified {n} document(s)")
        async def classify_documents(data_documents):
            ...
    """

    def decorator(func):
        func.__task_summary__ = template
        return func

    return decorator


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
    _execute_method: Callable = None
    _next_batch_size: int = 1

    def __init__(self, executable, *args, task_config=None, **kwargs):
        self.executable = executable
        self.default_params = {"args": args, "kwargs": kwargs}

        if inspect.isasyncgenfunction(executable):
            self.task_type = "Async Generator"
            self._execute_method = self.execute_async_generator
        elif inspect.isgeneratorfunction(executable):
            self.task_type = "Generator"
            self._execute_method = self.execute_generator
        elif inspect.iscoroutinefunction(executable):
            self.task_type = "Coroutine"
            self._execute_method = self.execute_coroutine
        elif inspect.isfunction(executable):
            self.task_type = "Function"
            self._execute_method = self.execute_function
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

    async def execute_async_generator(self, args, kwargs):
        """Execute async generator task and collect results in batches."""
        results = []
        async_iterator = self.run(*args, **kwargs)

        async for partial_result in async_iterator:
            results.append(partial_result)

            if len(results) == self._next_batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_generator(self, args, kwargs):
        """Execute generator task and collect results in batches."""
        results = []

        for partial_result in self.run(*args, **kwargs):
            results.append(partial_result)

            if len(results) == self._next_batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_coroutine(self, args, kwargs):
        """Execute coroutine task and yield the result."""
        task_result = await self.run(*args, **kwargs)
        yield task_result

    async def execute_function(self, args, kwargs):
        """Execute function task and yield the result."""
        task_result = self.run(*args, **kwargs)
        yield task_result

    async def execute(self, args, kwargs, next_batch_size=None):
        """Execute the task based on its type and yield results with the next task's batch size."""
        if next_batch_size is not None:
            self._next_batch_size = next_batch_size

        async for result in self._execute_method(args, kwargs):
            yield result
