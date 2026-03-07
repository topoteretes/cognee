"""
Pipeline execution via run_steps().

run_steps() is the primary entry point for the simplified pipeline API.
It accepts plain functions, @step-decorated functions, or Task objects,
auto-wraps them as needed, and returns results directly.

Supports two execution modes:
- Sequential (default): data flows step-by-step, breadth-first.
- Parallel (parallel=True): each input item flows through the full
  chain independently, concurrently — like the original run_tasks engine.

Example:
    results = await run_steps(extract, transform, load, input="text")

    # Per-item parallelism (each doc processed through full chain concurrently):
    results = await run_steps(chunk, extract, store, input=docs, parallel=True)
"""

import asyncio
import inspect
from functools import wraps

from cognee.pipelines.types import (
    _Drop,
    get_ctx_param_name,
)


def _get_original(fn):
    """Unwrap a @step-decorated function to get the original."""
    return getattr(fn, "_original_fn", fn)


def _get_step_config(fn):
    """Get StepConfig from a @step-decorated function, if present."""
    return getattr(fn, "_cognee_step_config", None)


def _get_default_params(fn) -> dict:
    """Get default params from a @step-decorated function."""
    config = _get_step_config(fn)
    return config.params if config else {}


def _wrap_with_default_params(fn, default_params: dict):
    """Wrap a function to inject default params from @step(..., key=value).

    Only injects params that the function actually accepts and that
    weren't already provided by the caller.
    """
    if not default_params:
        return fn

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    applicable = {k: v for k, v in default_params.items() if k in accepted}

    if not applicable:
        return fn

    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            merged = {**applicable, **kwargs}
            async for item in fn(*args, **merged):
                yield item

        return wrapper

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            merged = {**applicable, **kwargs}
            return await fn(*args, **merged)

        return wrapper

    if inspect.isgeneratorfunction(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            merged = {**applicable, **kwargs}
            yield from fn(*args, **merged)

        return wrapper

    @wraps(fn)
    def wrapper(*args, **kwargs):
        merged = {**applicable, **kwargs}
        return fn(*args, **merged)

    return wrapper


def _wrap_with_ctx_injection(fn, context: dict):
    """Wrap a function to inject context into Ctx-annotated or legacy 'context' parameters."""
    sig = inspect.signature(fn)
    ctx_param = get_ctx_param_name(sig)

    # Fallback: legacy tasks use a plain parameter named "context"
    if ctx_param is None and "context" in sig.parameters:
        ctx_param = "context"

    if ctx_param is None:
        return fn

    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def async_gen_wrapper(*args, **kwargs):
            kwargs[ctx_param] = context
            async for item in fn(*args, **kwargs):
                yield item

        return async_gen_wrapper

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            kwargs[ctx_param] = context
            return await fn(*args, **kwargs)

        return async_wrapper

    if inspect.isgeneratorfunction(fn):

        @wraps(fn)
        def gen_wrapper(*args, **kwargs):
            kwargs[ctx_param] = context
            yield from fn(*args, **kwargs)

        return gen_wrapper

    @wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs[ctx_param] = context
        return fn(*args, **kwargs)

    return wrapper


def _split_batches(data, batch_size: int):
    """Split list data into batches. Returns None if no splitting needed."""
    if not isinstance(data, list) or batch_size <= 1 or len(data) <= batch_size:
        return None
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def _build_context(dataset, context):
    """Build the context dict, integrating dataset() and cognee_pipeline() context."""
    ctx = dict(context) if context else {}

    # Merge pipeline context (user, dataset objects) from cognee_pipeline() if active
    from cognee.pipelines.context import get_current_dataset, get_pipeline_context

    pipeline_ctx = get_pipeline_context()
    if pipeline_ctx:
        for key, value in pipeline_ctx.items():
            if key not in ctx:
                ctx[key] = value

    if dataset:
        ctx["dataset"] = dataset
    elif "dataset" not in ctx:
        current = get_current_dataset()
        if current is not None:
            ctx["dataset"] = current
    return ctx


def _prepare_step(step_fn, ctx):
    """Prepare a step function: unwrap, inject defaults, inject context."""
    original = _get_original(step_fn)
    default_params = _get_default_params(step_fn)
    wrapped = _wrap_with_default_params(original, default_params)
    wrapped = _wrap_with_ctx_injection(wrapped, ctx)
    return wrapped


async def _execute_step(wrapped, data, batch_size, enriches):
    """Execute a single step against data, handling all function types."""
    if inspect.isasyncgenfunction(wrapped):
        results = []
        async for result in wrapped(data):
            if not isinstance(result, _Drop):
                results.append(result)
        return results

    if inspect.iscoroutinefunction(wrapped):
        batches = _split_batches(data, batch_size) if batch_size > 1 else None

        if batches is not None:
            results = []
            for batch in batches:
                result = await wrapped(batch)
                if enriches and result is None:
                    results.extend(batch)
                elif isinstance(result, _Drop):
                    continue
                elif isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            return results

        result = await wrapped(data)
        if enriches and result is None:
            return data
        if isinstance(result, _Drop):
            return []
        return result

    if inspect.isgeneratorfunction(wrapped):
        results = []
        for result in wrapped(data):
            if not isinstance(result, _Drop):
                results.append(result)
        return results

    # Plain sync function
    result = wrapped(data)
    if enriches and result is None:
        return data
    if isinstance(result, _Drop):
        return []
    return result


async def _run_chain(steps_prepared, data, configs):
    """Run a single item through the full step chain sequentially."""
    for i, wrapped in enumerate(steps_prepared):
        batch_size = configs[i].batch_size if configs[i] else 1
        enriches = configs[i].enriches if configs[i] else False
        data = await _execute_step(wrapped, data, batch_size, enriches)
    return data


async def run_steps(
    *steps, input=None, dataset: str = None, context: dict = None, parallel: bool = False, **kwargs
):
    """Execute a pipeline of steps.

    Args:
        *steps: Functions to execute in sequence.
        input: Initial data to feed into the first step.
        dataset: Optional dataset name for context.
        context: Optional context dict (user, dataset info, etc.).
        parallel: If True and input is a list, each item flows through
                  the full chain independently, concurrently. This matches
                  the original run_tasks per-data-item parallelism.

    Returns:
        The output of the last step.

    Examples:
        # Sequential (default):
        results = await run_steps(step1, step2, input="data")

        # Parallel per-item:
        results = await run_steps(step1, step2, input=[d1, d2, d3], parallel=True)
    """
    if not steps:
        return input

    ctx = _build_context(dataset, context)

    # Prepare all steps (unwrap, inject defaults + context)
    steps_prepared = [_prepare_step(fn, ctx) for fn in steps]
    configs = [_get_step_config(fn) for fn in steps]

    # --- Parallel mode: each input item through full chain concurrently ---
    if parallel and isinstance(input, list):
        tasks = [asyncio.create_task(_run_chain(steps_prepared, item, configs)) for item in input]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from errors
        output = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append((i, result))
            elif isinstance(result, list):
                output.extend(result)
            else:
                output.append(result)

        if errors:
            from cognee.shared.logging_utils import get_logger

            logger = get_logger("run_steps")
            for idx, err in errors:
                logger.error(f"Item {idx} failed: {err}", exc_info=err)

        return output

    # --- Sequential mode: step-by-step, breadth-first ---
    data = input
    for i, wrapped in enumerate(steps_prepared):
        batch_size = configs[i].batch_size if configs[i] else 1
        enriches = configs[i].enriches if configs[i] else False
        data = await _execute_step(wrapped, data, batch_size, enriches)

    return data
