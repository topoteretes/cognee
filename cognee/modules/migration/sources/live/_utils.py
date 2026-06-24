"""Shared helpers for live-API memory sources."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def require_extra(extra: str, import_name: str) -> Any:
    """Import *import_name* or raise with an install hint for cognee[extra]."""
    try:
        return __import__(import_name, fromlist=[""])
    except ImportError as error:
        raise ImportError(
            f"Live import requires the optional '{extra}' dependency. "
            f"Install it with: pip install cognee[{extra}]"
        ) from error


async def call_maybe_async(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke *func* whether it is sync or async."""
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking SDK call off the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def paginate_uuid_cursor(
    fetch_page: Callable[..., Any],
    *,
    page_size: int,
    cursor_kw: str = "uuid_cursor",
    limit_kw: str = "limit",
) -> list[Any]:
    """Collect all items from a uuid-cursor paginated API."""
    items: list[Any] = []
    cursor = None
    while True:
        kwargs = {limit_kw: page_size}
        if cursor is not None:
            kwargs[cursor_kw] = cursor
        batch = await call_maybe_async(fetch_page, **kwargs)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < page_size:
            break
        last = batch[-1]
        cursor = getattr(last, "uuid_", None) or getattr(last, "uuid", None)
        if cursor is None and isinstance(last, dict):
            cursor = last.get("uuid") or last.get("uuid_")
        if cursor is None:
            break
    return items


async def await_if_needed(value: Any | Awaitable[Any]) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
