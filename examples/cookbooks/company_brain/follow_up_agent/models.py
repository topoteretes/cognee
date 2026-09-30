"""The follow-up graph model: what the agent needs to turn a call into the right Linear issues.

The model is shaped by the questions the agent asks after each call:

- "Who owns this next step, and on which team?" -> Person, member of a Team.
- "Which project is it part of?"                -> Project, owned by a Team.
- "Is it already tracked?"                      -> Issue, with its Linear identifier.
- "What was agreed, and when is it due?"        -> Meeting and ActionItem.

Every type declares ``identity_fields``, so the Omar Haddad in a Granola note, the assignee of
a Linear issue and the name in an email become one Person node. An Issue is identified by its
Linear identifier ("PAY-104"), which is how the agent recognizes a next step that is already
tracked.
"""

import re
from typing import Annotated

from pydantic import field_validator

from cognee.low_level import DataPoint, FromIdentity


class Team(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    # "Payments team" and "Payments" are the same team.
    @field_validator("name")
    @classmethod
    def _bare_team_name(cls, name: str) -> str:
        return re.sub(r"^(the\s+)?(.*?)(\s+team)?$", r"\2", name.strip(), flags=re.IGNORECASE)


class Person(DataPoint):
    name: str
    email: str | None = None
    role: str | None = None
    member_of: Annotated[Team, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Project(DataPoint):
    name: str
    owned_by: Annotated[Team, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Issue(DataPoint):
    identifier: str
    # Optional so an action item can point at an issue by its identifier alone.
    title: str | None = None
    status: str | None = None
    assignee: Annotated[Person, FromIdentity()] | None = None
    team: Annotated[Team, FromIdentity()] | None = None
    project: Annotated[Project, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["title"], "identity_fields": ["identifier"]}


class Meeting(DataPoint):
    title: str
    date: str | None = None
    attendees: Annotated[list[Person], FromIdentity()] = []
    about: Annotated[list[Project], FromIdentity()] = []
    metadata: dict = {"index_fields": ["title"], "identity_fields": ["title"]}


class ActionItem(DataPoint):
    description: str
    owner: Annotated[Person, FromIdentity()] | None = None
    due: str | None = None
    from_meeting: Annotated[Meeting, FromIdentity()] | None = None
    tracked_in: Annotated[Issue, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["description"]}


class CompanyGraph(DataPoint):
    people: list[Person] = []
    teams: list[Team] = []
    projects: list[Project] = []
    issues: list[Issue] = []
    meetings: list[Meeting] = []
    action_items: list[ActionItem] = []


EXTRACTION_PROMPT = """
Extract the people, teams, projects, Linear issues, meetings and action items in the text.

Write every name exactly as the text writes it and add no words to it: a team is "Payments",
never "Payments team"; a project is "Checkout v2", never "the Checkout v2 project". Use full
names for people. A Linear issue is identified by its identifier, such as "PAY-104". A
meeting is identified by its title.

An action item is something a person agreed to do. Record its owner, its due date as written,
the meeting it came from, and the Linear issue that tracks it when the text names one.

Put every person, team, project, issue and meeting you mention anywhere, including ones you
only reference, in the matching top-level list.
"""
