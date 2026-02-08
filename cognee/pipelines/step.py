"""
@step decorator for configuring pipeline steps.

Provides a declarative way to add configuration to pipeline functions
without wrapping them in Task objects.

Example:
    @step(batch_size=10)
    async def process_batch(items: list) -> list:
        return [x * 2 for x in items]

    # Use in flow:
    results = await flow(process_batch, input=[1, 2, 3])
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


@dataclass
class StepConfig:
    """Configuration attached to a decorated step function."""

    batch_size: int = 1
    cache: bool = False
    extra: dict = field(default_factory=dict)


def step(fn=None, *, batch_size: int = 1, cache: bool = False, **extra_config):
    """Decorator to configure a pipeline step.

    Can be used with or without arguments:

        @step
        async def simple(data): ...

        @step(batch_size=10)
        async def batched(data): ...

        @step(batch_size=10, cache=True)
        async def cached_batch(data): ...

    The decorated function retains its original behavior and can be called
    directly. The configuration is stored as metadata for the pipeline runtime.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        import inspect

        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper

        # Store config as metadata on the wrapper
        wrapper._cognee_step_config = StepConfig(
            batch_size=batch_size,
            cache=cache,
            extra=extra_config,
        )
        wrapper._original_fn = func

        return wrapper

    if fn is not None:
        # Called without arguments: @step
        return decorator(fn)
    # Called with arguments: @step(batch_size=10)
    return decorator
