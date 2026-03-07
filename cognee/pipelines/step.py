"""
@step decorator for configuring pipeline steps.

Example:
    @step(batch_size=10)
    async def process_batch(items: list) -> list:
        return [x * 2 for x in items]

    # Default params — injected automatically when the step is called:
    @step(batch_size=10, graph_model=KnowledgeGraph)
    async def extract_graph(chunks, graph_model): ...

    results = await run_steps(process_batch, input=[1, 2, 3])
"""

import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable


@dataclass
class StepConfig:
    """Configuration attached to a decorated step function."""

    batch_size: int = 1
    cache: bool = False
    enriches: bool = False
    params: dict = field(default_factory=dict)


def step(fn=None, *, batch_size: int = 1, cache: bool = False, enriches: bool = False, **params):
    """Decorator to configure a pipeline step.

    Can be used with or without arguments:

        @step
        async def simple(data): ...

        @step(batch_size=10)
        async def batched(data): ...

        @step(batch_size=10, graph_model=KnowledgeGraph)
        async def extract(chunks, graph_model): ...

    Extra keyword arguments beyond batch_size/cache/enriches are stored as
    default params and auto-injected when the step is called, matching by
    parameter name. This replaces Task(fn, **kwargs).
    """

    def decorator(func: Callable) -> Callable:
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def wrapper(*args, **kwargs):
                async for item in func(*args, **kwargs):
                    yield item

        elif inspect.iscoroutinefunction(func):

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

        elif inspect.isgeneratorfunction(func):

            @wraps(func)
            def wrapper(*args, **kwargs):
                yield from func(*args, **kwargs)

        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

        wrapper._cognee_step_config = StepConfig(
            batch_size=batch_size,
            cache=cache,
            enriches=enriches,
            params=params,
        )
        wrapper._original_fn = func

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
