"""
Simplified pipeline API for Cognee.

Three tiers of abstraction:

Tier 1 - Smart Functions (zero config):
    results = await flow(extract, transform, load, input=data)

Tier 2 - Decorators (add config when needed):
    @step(batch_size=10)
    async def process(items): ...

Tier 3 - Pipeline Objects (full control):
    pipeline = Pipeline("my-pipeline").add_step(fn1).add_step(fn2)
    results = await pipeline.execute(input=data)

Legacy imports (from cognee.pipelines import Task, run_tasks, etc.) also work.
"""

# New simplified API (no circular dependency risk)
from cognee.pipelines.flow import flow
from cognee.pipelines.step import step
from cognee.pipelines.builder import Pipeline
from cognee.pipelines.context import dataset, get_current_dataset
from cognee.pipelines.types import (
    Pipe,
    Ctx,
    Cfg,
    Drop,
    Batch,
    Stream,
    inject,
    get_pipe_param_name,
    get_ctx_param_name,
    get_cfg_param_name,
)

# Legacy re-exports are lazy to avoid circular imports with cognee.modules.pipelines
_LEGACY_IMPORTS = {
    "Task": "cognee.modules.pipelines.tasks.task",
    "run_tasks": "cognee.modules.pipelines.operations.run_tasks",
    "run_tasks_parallel": "cognee.modules.pipelines.operations.run_parallel",
    "run_pipeline": "cognee.modules.pipelines.operations.pipeline",
}


def __getattr__(name):
    if name in _LEGACY_IMPORTS:
        import importlib

        module = importlib.import_module(_LEGACY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Legacy (backward compatible, lazy-loaded)
    "Task",
    "run_tasks",
    "run_tasks_parallel",
    "run_pipeline",
    # Tier 1: Simple execution
    "flow",
    # Tier 2: Decorators
    "step",
    # Tier 3: Builder
    "Pipeline",
    # Context
    "dataset",
    "get_current_dataset",
    # Type annotations
    "Pipe",
    "Ctx",
    "Cfg",
    "Drop",
    "Batch",
    "Stream",
    "inject",
    # Introspection helpers
    "get_pipe_param_name",
    "get_ctx_param_name",
    "get_cfg_param_name",
]
