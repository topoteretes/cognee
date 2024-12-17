import asyncio
from uuid import uuid5
from typing import Type

from pydantic import BaseModel

from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.extraction.extract_summary import extract_summary
from cognee.shared.CodeGraphEntities import CodeFile
from .models import CodeSummary


async def summarize_code(
    code_graph_nodes: list[DataPoint],
    summarization_model: Type[BaseModel],
) -> list[DataPoint]:
    if len(code_graph_nodes) == 0:
        return

    code_files_data_points = [file for file in code_graph_nodes if isinstance(file, CodeFile)]

    file_summaries = await asyncio.gather(
        *[extract_summary(file.source_code, summarization_model) for file in code_files_data_points]
    )

    file_summaries_map = {
        code_file_data_point.extracted_id: file_summary.summary
        for code_file_data_point, file_summary in zip(code_files_data_points, file_summaries)
    }

    for node in code_graph_nodes:
        if not isinstance(node, DataPoint):
            continue
        yield node

        if not isinstance(node, CodeFile):
            continue

        yield CodeSummary(
            id=uuid5(node.id, "CodeSummary"),
            made_from=node,
            text=file_summaries_map[node.extracted_id],
        )
