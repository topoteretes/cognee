"""
Simplified pipeline API for Cognee.

Pipeline definition uses the deferred-call pattern:

    classify = task(classify_documents)
    extract = task(extract_graph, batch_size=20)

    await run_pipeline([
        classify(),
        extract(graph_model=KnowledgeGraph),
    ], data=raw_input)

Legacy imports (from cognee.pipelines import Task, run_tasks, etc.) also work.
"""

from cognee.pipelines.types import (
    Drop,
)

# Legacy re-exports are lazy to avoid circular imports with cognee.modules.pipelines
_LEGACY_IMPORTS = {
    "Task": "cognee.modules.pipelines.tasks.task",
    "task": "cognee.modules.pipelines.tasks.task",
    "run_tasks": "cognee.modules.pipelines.operations.run_tasks",
    "run_tasks_parallel": "cognee.modules.pipelines.operations.run_parallel",
    "run_pipeline": "cognee.modules.pipelines.operations.run_pipeline",
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
    "task",
    "run_tasks",
    "run_tasks_parallel",
    "run_pipeline",
    # Type annotations
    "Drop",
]
