from typing import Union, Callable, Any, Coroutine, Generator, AsyncGenerator
import inspect

from cognee.pipelines.types import _Drop


class BoundTask:
    """A Task with pre-bound keyword arguments, ready for pipeline chaining.

    Created by calling a TaskSpec. The first positional argument (pipeline data)
    is supplied at execution time by run_pipeline; all other kwargs are captured
    at definition time.

    Example::

        extract = task(extract_graph, batch_size=20)
        bound = extract(graph_model=KnowledgeGraph)
        # bound.task has batch_size=20
        # bound.kwargs has {"graph_model": KnowledgeGraph}
        # When the pipeline runs: extract_graph(pipeline_data, graph_model=KnowledgeGraph)
    """

    def __init__(self, inner_task: "Task", **kwargs):
        self.task = inner_task
        self.kwargs = kwargs

    def __repr__(self):
        name = self.task.executable.__name__
        params = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        bs = self.task.task_config.get("batch_size", 1)
        return f"BoundTask({name}({params}), batch_size={bs})"


class TaskSpec:
    """Callable wrapper returned by @task.

    Calling a TaskSpec does NOT execute the function — it returns a BoundTask
    that captures kwargs for later execution by run_pipeline.

    Supports three usage patterns::

        # As a decorator
        @task(batch_size=20)
        async def extract_graph(chunks, graph_model=None): ...

        # As a functional wrapper
        extract_graph_task = task(extract_graph_existing, batch_size=20)

        # Both produce a TaskSpec. Calling it creates a BoundTask:
        bound = extract_graph_task(graph_model=KnowledgeGraph)
        bound = extract_graph_task(graph_model=KnowledgeGraph, batch_size=5)

        # Use in a pipeline:
        await run_pipeline([
            classify_task(),
            extract_graph_task(graph_model=KnowledgeGraph),
        ], data=raw_input, dataset="main")

    To call the underlying function directly (for testing), use .direct()::

        result = await extract_graph_task.direct(chunks, graph_model=KnowledgeGraph)
    """

    def __init__(self, fn, batch_size=None, enriches=False, **default_params):
        self._fn = fn
        self._batch_size = batch_size
        self._enriches = enriches
        self._default_params = default_params

        # Pre-build the base Task
        self._base_task = Task(fn, batch_size=batch_size, enriches=enriches, **default_params)

        # Copy function metadata for introspection
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = getattr(fn, "__module__", None)
        self.__wrapped__ = fn

    def __call__(self, **kwargs) -> BoundTask:
        """Create a BoundTask with pre-bound kwargs.

        Special kwargs:
            batch_size: Override the Task's batch_size for this pipeline step.
            enriches: Override the enriches flag for this pipeline step.

        All other kwargs are passed to the underlying function at execution time.
        """
        batch_size = kwargs.pop("batch_size", None)
        enriches = kwargs.pop("enriches", None)

        if batch_size is not None or enriches is not None:
            inner = self._base_task.with_config(
                **({"batch_size": batch_size} if batch_size is not None else {}),
                **({"enriches": enriches} if enriches is not None else {}),
            )
        else:
            inner = self._base_task

        return BoundTask(inner, **kwargs)

    @property
    def task(self) -> "Task":
        """Access the underlying Task directly (backward compat)."""
        return self._base_task

    def direct(self, *args, **kwargs):
        """Call the underlying function directly (for testing/one-off use).

        Returns the raw coroutine/generator/value — not a BoundTask.
        """
        merged = {**self._default_params, **kwargs}
        return self._fn(*args, **merged)

    def __repr__(self):
        bs = self._batch_size
        return f"TaskSpec({self.__name__}, batch_size={bs})"


def task(fn=None, *, batch_size=None, enriches=False, **default_params):
    """Create a TaskSpec from a function.

    Can be used as a decorator or as a functional wrapper::

        # Decorator (with or without arguments)
        @task
        async def classify(data): ...

        @task(batch_size=20)
        async def extract(chunks, graph_model=None): ...

        # Functional wrapper (for functions you don't own)
        extract_task = task(extract_graph_existing, batch_size=20)

    Calling the result returns a BoundTask for use in run_pipeline::

        await run_pipeline([
            classify(),                              # no extra kwargs
            extract(graph_model=KnowledgeGraph),     # bind config
            extract(graph_model=KG, batch_size=5),   # override batch_size
        ], data=input_data)

    To call the function directly (testing): extract.direct(chunks, graph_model=KG)
    """

    def decorator(func):
        return TaskSpec(func, batch_size=batch_size, enriches=enriches, **default_params)

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

        # Whether the executable accepts a ctx parameter.
        # Used by the pipeline to decide whether to pass PipelineContext.
        try:
            self.accepts_ctx = "ctx" in inspect.signature(executable).parameters
        except (ValueError, TypeError):
            self.accepts_ctx = False

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

    async def execute_async_generator(self, args, kwargs, batch_size):
        """Execute async generator task and collect results in batches."""
        results = []
        async_iterator = self.run(*args, **kwargs)

        async for partial_result in async_iterator:
            if isinstance(partial_result, _Drop):
                continue
            results.append(partial_result)

            if len(results) == batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_generator(self, args, kwargs, batch_size):
        """Execute generator task and collect results in batches."""
        results = []

        for partial_result in self.run(*args, **kwargs):
            if isinstance(partial_result, _Drop):
                continue
            results.append(partial_result)

            if len(results) == batch_size:
                yield results
                results = []

        if results:
            yield results

    async def execute_coroutine(self, args, kwargs, batch_size):
        """Execute coroutine task and yield the result."""
        task_result = await self.run(*args, **kwargs)
        if isinstance(task_result, _Drop):
            return
        if self.enriches and task_result is None:
            yield args[0] if args else None
            return
        yield task_result

    async def execute_function(self, args, kwargs, batch_size):
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
        batch_size = next_batch_size if next_batch_size is not None else 1

        async for result in self._execute_method(args, kwargs, batch_size):
            yield result
