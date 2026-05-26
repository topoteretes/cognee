from __future__ import annotations

import asyncio
from uuid import UUID

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.tasks.summarization.models import GlobalContextSummary

from .bucket_assignment import create_root_summary_id
from .models import GlobalContextSummaryContent, SummaryNode


async def summarize_with_prompt(children: list[SummaryNode], prompt_file: str) -> str:
    child_summaries = "\n\n".join(
        f"Input {index + 1}:\n{child.text}" for index, child in enumerate(children)
    )
    system_prompt = read_query_prompt(prompt_file) or ""
    result = await LLMGateway.acreate_structured_output(
        child_summaries,
        system_prompt,
        GlobalContextSummaryContent,
    )
    return result.summary


async def generate_bucket_summary(children: list[SummaryNode]) -> str:
    return await summarize_with_prompt(children, "global_context_bucket_summary.txt")


async def generate_global_context_summary(children: list[SummaryNode]) -> str:
    return await summarize_with_prompt(children, "global_context_root_summary.txt")


async def build_bucket_summary_datapoint(
    bucket: SummaryNode,
    children_by_id: dict[str, SummaryNode],
    dataset_id: str,
) -> GlobalContextSummary:
    children = [
        children_by_id[child_id] for child_id in bucket.child_ids if child_id in children_by_id
    ]
    bucket.text = await generate_bucket_summary(children)

    return GlobalContextSummary(
        id=UUID(bucket.id),
        text=bucket.text,
        dataset_id=dataset_id,
        level=bucket.level if bucket.level is not None else 0,
        is_root=False,
    )


async def generate_bucket_summary_datapoints(
    buckets: list[SummaryNode],
    children_by_id: dict[str, SummaryNode],
    dataset_id: str,
) -> list[GlobalContextSummary]:
    return await asyncio.gather(
        *[build_bucket_summary_datapoint(bucket, children_by_id, dataset_id) for bucket in buckets]
    )


async def build_global_context_summary_datapoint(
    children: list[SummaryNode],
    dataset_id: str,
    level: int,
) -> GlobalContextSummary:
    root_text = await generate_global_context_summary(children)
    return GlobalContextSummary(
        id=create_root_summary_id(dataset_id),
        text=root_text,
        dataset_id=dataset_id,
        level=level,
        is_root=True,
    )
