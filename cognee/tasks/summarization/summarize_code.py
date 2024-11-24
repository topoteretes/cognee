import asyncio
from typing import Type, Union
from uuid import uuid5

from pydantic import BaseModel

from cognee.modules.data.extraction.extract_summary import extract_summary
from cognee.shared.CodeGraphEntities import CodeFile
from cognee.tasks.storage import add_data_points

from .models import CodeSummary


async def summarize_code(
    code_files: list[CodeFile], summarization_model: Type[BaseModel]
) -> list[CodeFile]:
    if len(code_files) == 0:
        return code_files

    file_summaries = await asyncio.gather(
        *[extract_summary(file.source_code, summarization_model) for file in code_files]
    )

    summaries = [
        CodeSummary(
            id=uuid5(file.id, "CodeSummary"),
            made_from=file,
            text=file_summaries[file_index].summary,
        )
        for (file_index, file) in enumerate(code_files)
    ]

    await add_data_points(summaries)

    return code_files, summaries
