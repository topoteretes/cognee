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
from uuid import UUID


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


@dataclass
class ParentRun:
    """One link of the parent-attribution chain (operation OR pipeline run).

    ``pipeline_runs.parent_operation_id`` must mirror the token-chaining
    structure above: a cognify pipeline running inside a memify pipeline
    parents to that memify run, not to the enclosing operation. Otherwise
    the two runs appear as siblings each carrying the same (chained) token
    totals, and summing a row's children double-counts.
    """

    run_id: UUID
    parent: Optional["ParentRun"] = None


_current_parent_run: ContextVar[Optional[ParentRun]] = ContextVar(
    "cognee_current_parent_run", default=None
)


def get_parent_run_id(excluding: Optional[UUID] = None) -> Optional[UUID]:
    """The innermost enclosing run id — the value for ``parent_operation_id``.

    ``excluding`` skips the caller's own scope: a pipeline's terminal-row
    writer runs inside that pipeline's own ``parent_run_scope``, and its
    parent is the next scope up, not itself.
    """
    node = _current_parent_run.get()
    if node is not None and excluding is not None and node.run_id == excluding:
        node = node.parent
    return node.run_id if node is not None else None


@contextmanager
def parent_run_scope(run_id: UUID) -> Iterator[None]:
    """Mark *run_id* (an operation id or pipeline run id) as the current parent."""
    node = ParentRun(run_id=run_id, parent=_current_parent_run.get())
    token = _current_parent_run.set(node)
    try:
        yield
    finally:
        _current_parent_run.reset(token)
