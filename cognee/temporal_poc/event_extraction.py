import asyncio

from pydantic import BaseModel
from typing import Type, List
from datetime import datetime, timezone

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.utils import generate_node_id
from cognee.temporal_poc.models.models import EventList
from cognee.temporal_poc.datapoints.datapoints import Interval, Timestamp, Event


# Global system prompt for event extraction
EVENT_EXTRACTION_SYSTEM_PROMPT = """
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


def date_to_int(ts: Timestamp) -> int:
    """Convert timestamp to integer milliseconds."""
    dt = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=timezone.utc)
    time = int(dt.timestamp() * 1000)
    return time


def create_timestamp_datapoint(ts: Timestamp) -> Timestamp:
    """Create a Timestamp datapoint from a Timestamp model."""
    time_at = date_to_int(ts)
    timestamp_str = (
        f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d} {ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
    )
    return Timestamp(
        id=generate_node_id(str(time_at)),
        time_at=time_at,
        year=ts.year,
        month=ts.month,
        day=ts.day,
        hour=ts.hour,
        minute=ts.minute,
        second=ts.second,
        timestamp_str=timestamp_str,
    )


def create_event_datapoint(event) -> Event:
    """Create an Event datapoint from an event model."""
    # Base event data
    event_data = {
        "name": event.name,
        "description": event.description,
        "location": event.location,
    }

    # Create timestamps if they exist
    time_from = create_timestamp_datapoint(event.time_from) if event.time_from else None
    time_to = create_timestamp_datapoint(event.time_to) if event.time_to else None

    # Add temporal information
    if time_from and time_to:
        event_data["during"] = Interval(time_from=time_from, time_to=time_to)
        # Enrich description with temporal info
        temporal_info = f"\n---\nTime data: {time_from.timestamp_str} to {time_to.timestamp_str}"
        event_data["description"] = (event_data["description"] or "Event") + temporal_info
    elif time_from or time_to:
        timestamp = time_from or time_to
        event_data["at"] = timestamp
        # Enrich description with temporal info
        temporal_info = f"\n---\nTime data: {timestamp.timestamp_str}"
        event_data["description"] = (event_data["description"] or "Event") + temporal_info

    return Event(**event_data)


async def extract_event_graph(
    content: str, response_model: Type[BaseModel], system_prompt: str = None
):
    """Extract event graph from content using LLM."""

    if system_prompt is None:
        system_prompt = EVENT_EXTRACTION_SYSTEM_PROMPT

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph


async def extract_events_and_entities(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Extracts events and entities from a chunk of documents."""
    events = await asyncio.gather(
        *[extract_event_graph(chunk.text, EventList) for chunk in data_chunks]
    )

    for data_chunk, event_list in zip(data_chunks, events):
        for event in event_list.events:
            event_datapoint = create_event_datapoint(event)
            data_chunk.contains.append(event_datapoint)

    return data_chunks
