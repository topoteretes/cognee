"""Pure in-memory tests for the operation token accumulator (SDK-399).

``operation_usage_scope()`` activates a ContextVar-scoped ``OperationUsage``
chained to any already-active one: child scopes propagate their tokens up
the parent chain (so ``remember`` sees its nested add/cognify totals), the
contextvar is reset on exit, and ``asyncio.create_task`` children inherit
the scope that spawned them.
"""

import asyncio

import pytest

from cognee.modules.operations.usage_accumulator import (
    OperationUsage,
    get_active_operation_usage,
    operation_usage_scope,
)


def test_no_scope_active_by_default():
    assert get_active_operation_usage() is None


def test_scope_activates_and_resets():
    with operation_usage_scope() as usage:
        assert get_active_operation_usage() is usage
        usage.add(3, 1)
        assert (usage.tokens_in, usage.tokens_out) == (3, 1)
    assert get_active_operation_usage() is None


def test_nested_scope_propagates_to_parent_chain():
    with operation_usage_scope() as outer:
        with operation_usage_scope() as middle:
            with operation_usage_scope() as inner:
                inner.add(10, 4)
            middle.add(5, 2)

        # Child totals include everything below; parents include children.
        assert (inner.tokens_in, inner.tokens_out) == (10, 4)
        assert (middle.tokens_in, middle.tokens_out) == (15, 6)
        assert (outer.tokens_in, outer.tokens_out) == (15, 6)

        # After the child exits, the outer scope is active again.
        assert get_active_operation_usage() is outer
        outer.add(1, 1)

    assert (outer.tokens_in, outer.tokens_out) == (16, 7)


def test_parent_chain_is_explicit():
    with operation_usage_scope() as outer:
        with operation_usage_scope() as inner:
            assert inner.parent is outer
        assert outer.parent is None


def test_add_without_parent_is_local_only():
    usage = OperationUsage()
    usage.add(7, 3)
    assert (usage.tokens_in, usage.tokens_out) == (7, 3)
    assert usage.parent is None


@pytest.mark.asyncio
async def test_create_task_children_inherit_the_active_scope():
    """run_tasks' parallel _run_item fan-out attributes to the run's scope."""

    async def _worker(tokens_in, tokens_out):
        active = get_active_operation_usage()
        assert active is not None
        active.add(tokens_in, tokens_out)

    with operation_usage_scope() as run_usage:
        await asyncio.gather(
            asyncio.create_task(_worker(4, 1)),
            asyncio.create_task(_worker(6, 2)),
        )

    assert (run_usage.tokens_in, run_usage.tokens_out) == (10, 3)


@pytest.mark.asyncio
async def test_scope_does_not_leak_across_sibling_tasks():
    """A scope opened inside one task must not be visible to a sibling task."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _scoped():
        with operation_usage_scope():
            started.set()
            await release.wait()

    async def _unscoped():
        await started.wait()
        assert get_active_operation_usage() is None
        release.set()

    await asyncio.gather(asyncio.create_task(_scoped()), asyncio.create_task(_unscoped()))
