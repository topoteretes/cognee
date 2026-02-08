"""
Simple pipeline execution via flow().

flow() is the primary entry point for the simplified pipeline API.
It accepts plain functions, @step-decorated functions, or Task objects,
auto-wraps them as needed, and returns results directly.

Example:
    async def extract_names(text: str) -> list[str]:
        return ["Alice", "Bob"]

    async def greet(names: list[str]) -> list[str]:
        return [f"Hello {name}!" for name in names]

    results = await flow(extract_names, greet, input="Alice and Bob")
    # results = ["Hello Alice!", "Hello Bob!"]
"""

import inspect
from functools import wraps
from typing import Any, Optional

from cognee.pipelines.types import (
    Drop,
    _Drop,
    _CtxMarker,
    _PipeMarker,
    get_ctx_param_name,
    get_pipe_param_name,
)
from cognee.pipelines.step import StepConfig


def _get_original(fn):
    """Unwrap a @step-decorated function to get the original."""
    return getattr(fn, "_original_fn", fn)


def _get_step_config(fn) -> Optional[StepConfig]:
    """Get StepConfig from a @step-decorated function, if present."""
    return getattr(fn, "_cognee_step_config", None)


def _to_task(fn, step_config: Optional[StepConfig] = None):
    """Convert a function (plain or @step-decorated) to a Task object."""
    from cognee.modules.pipelines.tasks.task import Task

    original = _get_original(fn)

    task_config = None
    if step_config and step_config.batch_size > 1:
        task_config = {"batch_size": step_config.batch_size}

    return Task(original, task_config=task_config)


def _wrap_with_ctx_injection(fn, context: dict):
    """Wrap a function to inject context into Ctx-annotated parameters.

    If the function has a parameter annotated with Ctx[T], the pipeline
    context dict is automatically injected as that parameter's value.
    """
    sig = inspect.signature(fn)
    ctx_param = get_ctx_param_name(sig)

    if ctx_param is None:
        return fn

    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            kwargs[ctx_param] = context
            return await fn(*args, **kwargs)

    else:

        @wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs[ctx_param] = context
            return fn(*args, **kwargs)

    return wrapper


def _wrap_with_drop_filter(fn):
    """Wrap a function to filter out Drop sentinel values from results.

    If a step returns Drop, the item is removed from the pipeline.
    If a step returns a list containing Drop values, those items are filtered out.
    """
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            return _filter_drops(result)

    elif inspect.isasyncgenfunction(fn):
        # Don't wrap generators — Drop filtering happens at the flow level
        return fn
    else:

        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return _filter_drops(result)

    return wrapper


def _filter_drops(result):
    """Remove Drop sentinels from a result."""
    if isinstance(result, _Drop):
        return Drop
    if isinstance(result, list):
        filtered = [item for item in result if not isinstance(item, _Drop)]
        return filtered if filtered else Drop
    return result


async def flow(*steps, input=None, dataset: str = None, context: dict = None, **kwargs):
    """Execute a pipeline of steps with a simple API.

    Accepts plain async functions, @step-decorated functions, or Task objects.
    Returns results directly — no async generator ceremony needed.

    Args:
        *steps: Functions to execute in sequence. Output of step N becomes
                input to step N+1.
        input: Initial data to feed into the first step.
        dataset: Optional dataset name for context.
        context: Optional context dict (user, dataset info, etc.).
        **kwargs: Additional keyword arguments passed to the pipeline.

    Returns:
        The output of the last step. If multiple items, returns a list.

    Example:
        results = await flow(step1, step2, step3, input="data")
    """
    if not steps:
        return input

    # Build context
    ctx = dict(context) if context else {}
    if dataset:
        ctx["dataset"] = dataset

    # Execute steps sequentially
    data = input

    for step_fn in steps:
        original = _get_original(step_fn)

        # Wrap with context injection if needed
        wrapped = _wrap_with_ctx_injection(original, ctx)

        # Execute the step
        if inspect.isasyncgenfunction(wrapped):
            # Async generator: collect results
            results = []
            if isinstance(data, list):
                for item in data:
                    async for result in wrapped(item):
                        if not isinstance(result, _Drop):
                            results.append(result)
            else:
                async for result in wrapped(data):
                    if not isinstance(result, _Drop):
                        results.append(result)
            data = results
        elif inspect.iscoroutinefunction(wrapped):
            result = await wrapped(data)
            if isinstance(result, _Drop):
                return []
            data = result
        elif inspect.isgeneratorfunction(wrapped):
            results = []
            if isinstance(data, list):
                for item in data:
                    for result in wrapped(item):
                        if not isinstance(result, _Drop):
                            results.append(result)
            else:
                for result in wrapped(data):
                    if not isinstance(result, _Drop):
                        results.append(result)
            data = results
        else:
            result = wrapped(data)
            if isinstance(result, _Drop):
                return []
            data = result

    return data
