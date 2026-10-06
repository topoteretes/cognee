"""Which ``after_run_completed`` hook a dataset's run gets in a multi-dataset run."""

from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any


def hook_for_dataset(
    hook: Callable[..., Awaitable[Any]] | None, index: int, total: int
) -> Callable[..., Awaitable[Any]] | None:
    """``hook`` for the ``index``-th of ``total`` datasets, in run order.

    Every dataset but the last gets ``last_in_invocation=False``; the last gets
    the hook unchanged, so the hook's own default (True) applies. Wrappers only
    ever mark a dataset as NOT last, never as last, so they compose: the
    background executor splits a multi-dataset run into one ``run_pipeline``
    call per dataset and marks all but the last, and the single-dataset
    ``run_pipeline`` inside each call leaves that mark alone.
    """
    if hook is None or index >= total - 1:
        return hook
    return partial(hook, last_in_invocation=False)
