from typing import Union, Callable, Any, Coroutine, Generator, AsyncGenerator
import inspect

from cognee.pipelines.types import _Drop


def task(fn=None, *, batch_size=None, enriches=False, **default_params):
    """Decorator that wraps a function into a Task.

    The decorated function stays directly callable — no deferred execution.
    The Task is attached as a `.task` attribute so pipeline code can use it.

    Can be used with or without arguments:

        @task
        async def classify_documents(data):
            ...

        @task(batch_size=20)
        async def extract_graph(chunks, graph_model):
            ...

    Then build pipelines with the .task attribute:

        tasks = [classify_documents.task, extract_graph.task]

    Or override config at the call site:

        tasks = [extract_graph.task.with_config(batch_size=10)]
    """

    def decorator(func):
        t = Task(func, batch_size=batch_size, enriches=enriches, **default_params)
        func.task = t
        return func

    if fn is not None:
        return decorator(fn)
    return decorator


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
    enriches: bool = False
    _execute_method: Callable = None
    _next_batch_size: int = 1

    def __init__(
        self, executable, *args, task_config=None, batch_size=None, enriches=False, **kwargs
    ):
        self.executable = executable
        self.default_params = {"args": args, "kwargs": kwargs}
        self.enriches = enriches

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
        else:
            self.task_config = {"batch_size": 1}

        if batch_size is not None:
            self.task_config["batch_size"] = batch_size

    def with_config(self, **overrides) -> "Task":
        """Return a new Task with overridden config.

        Example:
            base = Task(extract_graph, batch_size=20, graph_model=KnowledgeGraph)
            tasks = [base.with_config(batch_size=10)]
        """
        batch_size = overrides.pop("batch_size", self.task_config["batch_size"])
        enriches = overrides.pop("enriches", self.enriches)
        merged_kwargs = {**self.default_params["kwargs"], **overrides}
        return Task(
            self.executable,
            *self.default_params["args"],
            batch_size=batch_size,
            enriches=enriches,
            **merged_kwargs,
        )

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
            if isinstance(partial_result, _Drop):
                continue
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
            if isinstance(partial_result, _Drop):
                continue
            results.append(partial_result)

            if len(results) == self._next_batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_coroutine(self, args, kwargs):
        """Execute coroutine task and yield the result."""
        task_result = await self.run(*args, **kwargs)
        if isinstance(task_result, _Drop):
            return
        if self.enriches and task_result is None:
            yield args[0] if args else None
            return
        yield task_result

    async def execute_function(self, args, kwargs):
        """Execute function task and yield the result."""
        task_result = self.run(*args, **kwargs)
        if isinstance(task_result, _Drop):
            return
        if self.enriches and task_result is None:
            yield args[0] if args else None
            return
        yield task_result

    async def execute(self, args, kwargs, next_batch_size=None):
        """Execute the task based on its type and yield results with the next task's batch size."""
        if next_batch_size is not None:
            self._next_batch_size = next_batch_size

        async for result in self._execute_method(args, kwargs):
            yield result
