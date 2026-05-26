from __future__ import annotations

from typing import Any

from cognee.modules.pipelines.models import PipelineContext

from .graph_input import load_context_index_input_from_graph
from .models import GlobalContextIndexInput


async def extract_global_context_index_input(
    data: Any = None,
    ctx: PipelineContext | None = None,
) -> GlobalContextIndexInput:
    if isinstance(data, GlobalContextIndexInput):
        return data
    return await load_context_index_input_from_graph(ctx)
