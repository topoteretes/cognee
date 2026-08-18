"""Session-independent, operation-scoped LLM token accumulation.

A ContextVar-scoped accumulator fed from ``record_llm_call`` (the single
choke point for counted LLM calls) regardless of whether a session-usage
scope is active. Accumulators chain to their parent scope, so a top-level
operation (e.g. ``remember``) sees the total including its nested
operations' tokens — do NOT sum token columns across nesting levels.

This module must stay import-light (no cognee imports) so it can be
imported from ``usage_tracking`` without pulling in the package-init chain.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class OperationUsage:
    """Mutable token tally for one active operation scope."""

    tokens_in: int = 0
    tokens_out: int = 0
    parent: Optional["OperationUsage"] = None

    def add(self, tokens_in: int, tokens_out: int) -> None:
        """Add a call's tokens locally, then propagate up the parent chain."""
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if self.parent is not None:
            self.parent.add(tokens_in, tokens_out)


_active_operation_usage: ContextVar[Optional[OperationUsage]] = ContextVar(
    "cognee_operation_usage", default=None
)


def get_active_operation_usage() -> Optional[OperationUsage]:
    """Return the innermost active operation accumulator, if any."""
    return _active_operation_usage.get()


@contextmanager
def operation_usage_scope() -> Iterator[OperationUsage]:
    """Activate a fresh accumulator chained to any already-active one.

    Sync contextmanager on purpose — it only sets/resets a ContextVar.
    ContextVars propagate into ``asyncio.create_task`` children, so
    parallel task fan-outs attribute to the scope that spawned them.
    """
    usage = OperationUsage(parent=_active_operation_usage.get())
    token = _active_operation_usage.set(usage)
    try:
        yield usage
    finally:
        _active_operation_usage.reset(token)
