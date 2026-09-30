"""The companion graph model: a small picture of one person's life that a local model can fill.

The companion asks memory the same few things at every check-in: which goals are active,
which dates are coming up, who the people around the user are, and what the user prefers.
Each of those is one node type, so every answer comes from a short walk through the graph.

The model is kept flat on purpose. A local 8B model extracts four types with a handful of
fields reliably; deep nesting is where small models start to drop fields.
"""

from typing import Annotated

from cognee.low_level import DataPoint, FromIdentity


class Person(DataPoint):
    name: str
    relationship: str | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Goal(DataPoint):
    description: str
    target_date: str | None = None
    status: str | None = None
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["description"]}


class Preference(DataPoint):
    description: str
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["description"]}


class Event(DataPoint):
    description: str
    date: str | None = None
    people: Annotated[list[Person], FromIdentity()] = []
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["description"]}


class LifeGraph(DataPoint):
    people: list[Person] = []
    goals: list[Goal] = []
    preferences: list[Preference] = []
    events: list[Event] = []


EXTRACTION_PROMPT = """
The text is a personal journal. Extract the people, goals, preferences and dated events in it.

- A person has their full name as written and their relationship to the writer, such as
  "sister" or "physio".
- A goal is something the writer is working toward, with its target date when one is given.
- A preference is how the writer likes to do things, such as "runs better in the morning".
- An event is something that happens on a date, with the date as written and the people in it.

Put every person you mention anywhere in the people list.
"""
