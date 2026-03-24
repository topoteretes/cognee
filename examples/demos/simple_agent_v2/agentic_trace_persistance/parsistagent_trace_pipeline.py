from __future__ import annotations

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.pipelines.layers.resolve_authorized_user_dataset import (
    resolve_authorized_user_dataset,
)
from cognee.tasks.storage import add_data_points
from examples.demos.simple_agent_v2.agentic_context_trace.prompt_trace_context import (
    AgentContextTrace,
)

TRACE_DATASET_NAME = "main_dataset"


async def persist_agent_trace_default_pipeline(trace: AgentContextTrace) -> None:
    _user, dataset = await resolve_authorized_user_dataset(dataset_name=TRACE_DATASET_NAME)
    await set_database_global_context_variables(dataset.id, dataset.owner_id)
    await add_data_points([trace])
