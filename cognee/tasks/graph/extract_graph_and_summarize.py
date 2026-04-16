from typing import List, Type, Optional, Dict
from pydantic import BaseModel
import asyncio

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.ontology.ontology_config import Config
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text


async def extract_graph_and_summarize(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    ctx=None,
    summarization_model: Type[BaseModel] = None,
    **kwargs,
):
    import time

    start = time.perf_counter()
    # print(f"START TIME OF EXTRACTION AND SUMMARIZATION: {start}")
    result_chunks = await asyncio.gather(
        extract_graph_from_data(
            data_chunks=data_chunks,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            ctx=ctx,
            **kwargs,
        ),
        summarize_text(
            data_chunks=data_chunks,
            summarization_model=summarization_model,
        ),
    )

    print(f"TOTAL TIME OF EXTRACTION AND SUMMARIZATION: {time.perf_counter() - start}")
    # Return only TextSummary objects, keeping the same logic as sequential execution of these tasks
    return result_chunks[1]
