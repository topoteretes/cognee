"""
Dataset context manager for scoped pipeline operations.

Provides an async context manager that sets the current dataset,
so pipeline functions don't need to thread dataset names everywhere.

Example:
    async with dataset("my_data"):
        await cognee.add(text)
        await cognee.cognify()
        results = await cognee.search(query)
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

_current_dataset: ContextVar[Optional[str]] = ContextVar("current_dataset", default=None)


@asynccontextmanager
async def dataset(name: str):
    """Context manager for dataset scope.

    Sets the current dataset for all pipeline operations within the block.
    Nested contexts are supported — the outer dataset is restored on exit.

    Args:
        name: The dataset name to use within this scope.

    Example:
        async with dataset("my-data"):
            await cognee.add(text)
            await cognee.cognify()
            results = await cognee.search(query)
    """
    previous = _current_dataset.get()
    _current_dataset.set(name)
    try:
        yield name
    finally:
        _current_dataset.set(previous)


def get_current_dataset() -> Optional[str]:
    """Get the current dataset name from context, or None if not set."""
    return _current_dataset.get()
