"""
Dataset context manager for scoped pipeline operations.

Provides context managers that set up the current dataset and Cognee
infrastructure (permissions, per-dataset DB isolation, etc.).

Example:
    # Lightweight — just sets the dataset name:
    async with dataset("my_data"):
        await cognee.add(text)
        await cognee.cognify()

    # Full orchestration — sets up DB isolation, permissions, ContextVars:
    async with cognee_pipeline(dataset="my_data"):
        results = await run_steps(extract, transform, load, input=data)
"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional

_current_dataset: ContextVar[Optional[str]] = ContextVar("current_dataset", default=None)
_pipeline_context: ContextVar[Optional[dict]] = ContextVar("pipeline_context", default=None)


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


@asynccontextmanager
async def cognee_pipeline(dataset: str = None, user=None):
    """Context manager that sets up full Cognee orchestration for run_steps().

    Handles everything the original run_pipeline does before task execution:
    - Creates DB tables and tests LLM/embedding connections (first run)
    - Resolves the user and checks dataset permissions
    - Sets up per-dataset database isolation (ContextVars) when
      ENABLE_BACKEND_ACCESS_CONTROL is enabled

    After this context is active, run_steps() calls inside it will
    automatically use the correct isolated databases.

    Args:
        dataset: Dataset name. If None, uses the current dataset() scope.
        user: Optional User object. Defaults to the system default user.

    Example:
        async with cognee_pipeline(dataset="my_data"):
            results = await run_steps(chunk, extract, store, input=docs)

        # Or nested inside a dataset() scope:
        async with dataset("my_data"):
            async with cognee_pipeline():
                results = await run_steps(chunk, extract, store, input=docs)
    """
    from cognee.modules.pipelines.layers.setup_and_check_environment import (
        setup_and_check_environment,
    )
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.context_global_variables import set_database_global_context_variables

    # Resolve dataset name
    dataset_name = dataset or get_current_dataset()

    # Set up environment (DB tables, LLM connection test on first run)
    await setup_and_check_environment()

    # Resolve user and check permissions
    datasets_arg = [dataset_name] if dataset_name else []
    _user, authorized_datasets = await resolve_authorized_user_datasets(
        datasets_arg, user
    )

    # Set up per-dataset DB isolation for the first authorized dataset
    active_dataset = authorized_datasets[0] if authorized_datasets else None
    if active_dataset:
        await set_database_global_context_variables(
            active_dataset.id, active_dataset.owner_id  # type: ignore[arg-type]
        )

    # Set dataset in ContextVar so run_steps picks it up
    ds_name = str(active_dataset.name) if active_dataset else dataset_name
    previous = _current_dataset.get()
    if ds_name:
        _current_dataset.set(ds_name)

    # Store resolved objects for run_steps context building
    _pipeline_context.set({
        "user": _user,
        "dataset": active_dataset,
    })

    try:
        yield active_dataset
    finally:
        _current_dataset.set(previous)
        _pipeline_context.set(None)


def get_current_dataset() -> Optional[str]:
    """Get the current dataset name from context, or None if not set."""
    return _current_dataset.get()


def get_pipeline_context() -> Optional[dict]:
    """Get the pipeline context (user, dataset objects) set by cognee_pipeline(), or None."""
    return _pipeline_context.get()
