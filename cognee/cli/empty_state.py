"""Cheap, honest emptiness checks — so search teaches instead of erroring.

Everything is answered from the relational database (two queries), which the
CLI and the pipelines always share. Deliberately NOT a graph-engine probe:
under multi-tenant access control (the default) each dataset lives in its own
graph database selected via context vars, so a bare get_graph_engine() would
probe the wrong, empty database and block search forever.

States:
- "empty":          nothing has ever been added -> teach add/cognify/search
- "not_cognified":  data was added but never successfully cognified -> teach cognify
- "ready":          there is something to search
"""

from typing import Optional, Tuple

COGNIFY_PIPELINE_NAMES = ("cognify_pipeline",)


async def check_memory_state(user) -> Tuple[str, Optional[str], int]:
    """Return (state, dataset_name, document_count).

    Any unexpected failure returns "ready" — the honest check must never
    block a search that might have worked.
    """
    try:
        from cognee.modules.data.methods import get_datasets

        datasets = await get_datasets(user.id)
        if not datasets:
            return "empty", None, 0

        if await _any_successful_cognify_run([dataset.id for dataset in datasets]):
            return "ready", None, 0

        first = datasets[0]
        count = 0
        try:
            from cognee.modules.data.methods import get_dataset_data

            count = len(await get_dataset_data(first.id))
        except Exception:
            pass
        return "not_cognified", getattr(first, "name", None), count
    except Exception:
        return "ready", None, 0


async def _any_successful_cognify_run(dataset_ids) -> bool:
    """True when any dataset's latest cognify run isn't a hard failure.

    A run that started (even if still in flight) counts as ready — wrongly
    blocking a search that could work is worse than letting it run.
    """
    try:
        from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status

        for pipeline_name in COGNIFY_PIPELINE_NAMES:
            statuses = await get_pipeline_status(dataset_ids, pipeline_name)
            for status in statuses.values():
                # str() comparison: some backends hand back enum instances,
                # others raw strings.
                if "ERRORED" not in str(status):
                    return True
        return False
    except Exception:
        # If the status table can't be read, assume ready — never block.
        return True
