"""Stored fact-conflict names and backend-independent property readers."""

import json

CONFLICT_ABOUT = "conflict_about"
CONFLICT_VALUE = "conflict_value"
CONFLICT_CITES = "conflict_cites"
CONFLICT_EDGE_PREFIX = "conflict_"


def is_conflict_edge(relationship_name: str) -> bool:
    return relationship_name.startswith(CONFLICT_EDGE_PREFIX)


def read_list_property(properties: dict, name: str) -> list:
    """Read a list or Neo4j's JSON alias, preserving an explicit empty list."""
    value = properties.get(name)
    if value is None:
        try:
            value = json.loads(properties.get(f"{name}_json") or "null")
        except (ValueError, TypeError):
            return []
    return value if isinstance(value, list) else []


def read_conflict_marks(properties: dict) -> list[dict]:
    return [
        mark for mark in read_list_property(properties, "conflict_marks") if isinstance(mark, dict)
    ]


def effective_date_display(value) -> str:
    return str(value)[:10] if value is not None else ""


# A fact can hold marks from several conflicts; the most alarming status wins.
STATUS_PRECEDENCE = ("conflicting", "superseded", "current")


def fact_status(properties: dict) -> str | None:
    statuses = {mark.get("status") for mark in read_conflict_marks(properties)}
    return next((status for status in STATUS_PRECEDENCE if status in statuses), None)


def status_label(properties: dict) -> str:
    """How a reviewed fact is annotated wherever it is shown to a reader or an LLM."""
    status = fact_status(properties)
    if status is None:
        return ""
    date = effective_date_display(properties.get("effective_date"))
    if not date:
        return f"[{status}]"
    if status == "current":
        return f"[as of {date}]"
    return f"[{status}; as of {date}]"
