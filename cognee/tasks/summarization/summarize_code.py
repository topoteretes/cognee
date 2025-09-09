import asyncio
from typing import AsyncGenerator, Union
from uuid import uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.llm.extraction import extract_code_summary
from .models import CodeSummary


async def summarize_code(
    code_graph_nodes: list[DataPoint],
) -> AsyncGenerator[Union[DataPoint, CodeSummary], None]:
    if len(code_graph_nodes) == 0:
        return

    code_data_points = [file for file in code_graph_nodes if hasattr(file, "source_code")]

    file_summaries = await asyncio.gather(
        *[extract_code_summary(file.source_code) for file in code_data_points]
    )

    file_summaries_map = {
        code_data_point.id: str(file_summary)
        for code_data_point, file_summary in zip(code_data_points, file_summaries)
    }

    for node in code_graph_nodes:
        if not isinstance(node, DataPoint):
            continue
        yield node

        if not hasattr(node, "source_code"):
            continue

        yield CodeSummary(
            id=uuid5(node.id, "CodeSummary"),
            summarizes=node,
            text=file_summaries_map[node.id],
        )
