from cognee.modules.engine.models import Interval, Event
from cognee.modules.engine.utils.generate_timestamp_datapoint import generate_timestamp_datapoint


def generate_event_datapoint(event) -> Event:
    """
    Generates an Event datapoint from a given event model, including temporal metadata if available.

    The function maps the basic attributes (name, description, location) from the input event
    and enriches them with temporal information. If start and end times are provided, an
    Interval is created. If only one timestamp is available, it is added directly. Temporal
    information is also appended to the event description for context.

    Args:
        event: An event model instance containing attributes such as name, description,
               location, time_from, and time_to.

    Returns:
        Event: A structured Event object with name, description, location, and enriched
               temporal details.
    """
    # Base event data
    event_data = {
        "name": event.name,
        "description": event.description,
        "location": event.location,
    }

    # Create timestamps if they exist
    time_from = generate_timestamp_datapoint(event.time_from) if event.time_from else None
    time_to = generate_timestamp_datapoint(event.time_to) if event.time_to else None

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
