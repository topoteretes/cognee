import asyncio
from pydantic import BaseModel
from typing import Union, Optional, List, Type
from uuid import UUID

from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.llm import get_max_chunk_tokens, get_llm_config

from cognee.api.v1.cognify.cognify import run_cognify_blocking
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

from cognee.modules.users.models import User
from cognee.tasks.documents import (
    check_permissions_on_dataset,
    classify_documents,
    extract_chunks_from_documents,
)
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee.temporal_poc.models.models import EventList
from cognee.temporal_poc.datapoints.datapoints import Interval, Timestamp, Event

logger = get_logger("temporal_cognify")


async def extract_event_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = """
        You are an extractor. From input text, pull out:

        Timestamps: concrete points (year, month, day, hour, minute, second).

        Intervals: spans with explicit start and end times; resolve relative durations if anchored.

        Entities: people, organizations, topics, etc., with name, short description, and with their type (person/org/location/topic/other). Always attach the type.

        Events: include name, brief description, subject (actor), object (target), time as either a point (at) or span (during), and location. Prefer during if it’s a multi-hour span; use at for a point. Omit ambiguous times rather than guessing.

        Output JSON. Reuse entity names when repeated. Use null for missing optional fields.
        ”
    """

    content_graph = await llm_client.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph


async def extract_events_and_entities(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Extracts events and entities from a chunk of documents."""
    # data_chunks = data_chunks + data_chunks

    events = await asyncio.gather(
        *[extract_event_graph(chunk.text, EventList) for chunk in data_chunks]
    )
    for data_chunk, event_list in zip(data_chunks, events):
        for event in event_list.events:
            if event.time_from and event.time_to:
                event_interval = Interval(
                    time_from=int(event.time_from), time_to=int(event.time_to)
                )
                event_datapoint = Event(
                    name=event.name,
                    description=event.description,
                    during=event_interval,
                    location=event.location,
                )
            elif event.time_from:
                event_time_at = Timestamp(time_at=int(event.time_from))
                event_datapoint = Event(
                    name=event.name,
                    description=event.description,
                    at=event_time_at,
                    location=event.location,
                )
            elif event.time_to:
                event_time_at = Timestamp(time_at=int(event.time_to))
                event_datapoint = Event(
                    name=event.name,
                    description=event.description,
                    at=event_time_at,
                    location=event.location,
                )
            else:
                event_datapoint = Event(
                    name=event.name, description=event.description, location=event.location
                )

            data_chunk.contains.append(event_datapoint)

    return data_chunks


async def get_temporal_tasks(
    user: User = None, chunker=TextChunker, chunk_size: int = None
) -> list[Task]:
    temporal_tasks = [
        Task(classify_documents),
        Task(check_permissions_on_dataset, user=user, permissions=["write"]),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size or get_max_chunk_tokens(),
            chunker=chunker,
        ),
        Task(extract_events_and_entities, task_config={"chunk_size": 10}),
        Task(add_data_points, task_config={"batch_size": 10}),
    ]

    return temporal_tasks


async def temporal_cognify(
    datasets: Union[str, list[str], list[UUID]] = None,
    user: User = None,
    chunker=TextChunker,
    chunk_size: int = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    incremental_loading: bool = True,
):
    tasks = await get_temporal_tasks(user, chunker, chunk_size)

    return await run_cognify_blocking(
        tasks=tasks,
        user=user,
        datasets=datasets,
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        incremental_loading=incremental_loading,
    )
