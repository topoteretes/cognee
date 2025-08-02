import asyncio
import uuid

from pydantic import BaseModel
from typing import Union, Optional, List, Type
from uuid import UUID, uuid5
from datetime import datetime, timezone
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


def date_to_int(ts: Timestamp) -> int:
    dt = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=timezone.utc)

    time = int(dt.timestamp() * 1000)
    return time


async def extract_event_graph(content: str, response_model: Type[BaseModel]):
    llm_client = get_llm_client()

    system_prompt = """
            For the purposes of building event-based knowledge graphs, you are tasked with extracting highly granular stream events from a text. The events are defined as follows:
            ## Event Definition
            - Anything with a date or a timestamp is an event
            - Anything that took place in time (even if the time is unknown) is an event
            - Anything that lasted over a period of time, or happened in an instant is an event: from historical milestones (wars, presidencies, olympiads) to personal milestones (birth, death, employment, etc.), to mundane actions (a walk, a conversation, etc.)
            - **ANY action or verb represents an event** - this is the most important rule
            - Every single verb in the text corresponds to an event that must be extracted
            - This includes: thinking, feeling, seeing, hearing, moving, speaking, writing, reading, eating, sleeping, working, playing, studying, traveling, meeting, calling, texting, buying, selling, creating, destroying, building, breaking, starting, stopping, beginning, ending, etc.
            - Even the most mundane or obvious actions are events: "he walked", "she sat", "they talked", "I thought", "we waited"
            ## Requirements
            - **Be extremely thorough** - extract EVERY event mentioned, no matter how small or obvious
            - **Timestamped first" - every time stamp, or date should have atleast one event
            - **Verbs/actions  = one event** - After you are done with timestamped events -- every verb that is an action should have a corresponding event.
            - We expect long streams of events from any piece of text, easily reaching a hundred events
            - Granularity and richness of the stream is key to our success and is of utmost importance
            - Not all events will have timestamps, add timestamps only to known events
            - For events that were instantaneous, just attach the time_from or time_to property don't create both
            - **Do not skip any events** - if you're unsure whether something is an event, extract it anyway
            - **Quantity over filtering** - it's better to extract too many events than to miss any
            - **Descriptions** - Always include the event description together with entities (Who did what, what happened? What is the event?). If you can include the corresponding part from the text.
            ## Output Format
            Your reply should be a JSON: list of dictionaries with the following structure:
            ```python
            class Event(BaseModel):
                name: str [concise]
                description: Optional[str] = None
                time_from: Optional[Timestamp] = None
                time_to: Optional[Timestamp] = None
                location: Optional[str] = None
            ```
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
                event_time_from = date_to_int(event.time_from)
                event_time_to = date_to_int(event.time_to)
                timestamp_time_from = Timestamp(
                    id=uuid5(uuid.NAMESPACE_OID, name=str(event_time_from)), time_at=event_time_from
                )
                timestamp_time_to = Timestamp(
                    id=uuid5(uuid.NAMESPACE_OID, name=str(event_time_to)), time_at=event_time_to
                )
                event_interval = Interval(time_from=timestamp_time_from, time_to=timestamp_time_to)
                event_datapoint = Event(
                    name=event.name,
                    description=event.description,
                    during=event_interval,
                    location=event.location,
                )
            elif event.time_from:
                event_time_from = date_to_int(event.time_from)
                event_time_at = Timestamp(
                    id=uuid5(uuid.NAMESPACE_OID, name=str(event_time_from)), time_at=event_time_from
                )
                event_datapoint = Event(
                    name=event.name,
                    description=event.description,
                    at=event_time_at,
                    location=event.location,
                )
            elif event.time_to:
                event_time_to = date_to_int(event.time_to)
                event_time_at = Timestamp(
                    id=uuid5(uuid.NAMESPACE_OID, name=str(event_time_to)), time_at=event_time_to
                )
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
