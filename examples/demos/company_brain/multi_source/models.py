"""The company brain graph model: five node types that every source is extracted into.

Each type declares ``identity_fields``, so its node id is derived from its name (or ticket
id). That is what links the sources: when the HR database, the ticket export and the
meeting notes each mention "Dana Kim", all three extractions produce the same ``Person``
node instead of three copies.

References to other nodes use ``FromIdentity``: the LLM answers a name ("Search") rather
than a nested object, and cognee resolves it against the nodes extracted from the same
text. That is why ``CompanyGraph`` asks for every referenced node in its top-level lists.
"""

import re
from typing import Annotated

from pydantic import field_validator

from cognee.low_level import DataPoint, Edge, FromIdentity


def _strip_words(name: str, prefix: str, suffix: str) -> str:
    """Drop a leading ``prefix`` word and a trailing ``suffix`` word, case-insensitively."""
    name = re.sub(rf"^(the\s+)?{prefix}\s+", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(rf"\s+{suffix}$", "", name, flags=re.IGNORECASE).strip()


class Team(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    # The prompt asks for bare names, but the LLM still writes "Billing team" now and
    # then. Normalizing here makes both spellings the same identity. It also covers
    # FromIdentity references, which are validated through this model.
    @field_validator("name")
    @classmethod
    def _bare_team_name(cls, name: str) -> str:
        return _strip_words(name, "the", "team")


class Customer(DataPoint):
    name: str
    industry: str | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Project(DataPoint):
    name: str
    status: str | None = None
    owned_by: Annotated[Team, FromIdentity()] | None = None
    for_customer: Annotated[Customer, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

    @field_validator("name")
    @classmethod
    def _bare_project_name(cls, name: str) -> str:
        # "Project Atlas", "Atlas project" and "Atlas (tech lead)" all mean "Atlas".
        name = re.sub(r"\s*\(.*\)$", "", name)
        return _strip_words(name, "project", "project")


class Person(DataPoint):
    name: str
    role: str | None = None
    member_of: Annotated[Team, FromIdentity()] | None = None
    works_on: Annotated[list[Project], FromIdentity()] = []
    # Same-type endpoints are spelled as strings: Person is not bound yet inside its body.
    reports_to: list[Edge["Person", "Person"]] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Ticket(DataPoint):
    ticket_id: str
    title: str
    status: str | None = None
    priority: str | None = None
    raised_by: Annotated[Customer, FromIdentity()] | None = None
    assigned_to: Annotated[Person, FromIdentity()] | None = None
    about: Annotated[Project, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["title"], "identity_fields": ["ticket_id"]}


class CompanyGraph(DataPoint):
    people: list[Person] = []
    teams: list[Team] = []
    projects: list[Project] = []
    customers: list[Customer] = []
    tickets: list[Ticket] = []


# Names are identities, so they must be spelled the same way in every source.
EXTRACTION_PROMPT = """
Extract the people, teams, projects, customers and support tickets in the text.

Write every name exactly as the text writes it and add no words to it: a team is
"Search", never "Search team" or "the Search Team"; a project is "Atlas", never
"Project Atlas" or "Atlas (tech lead)". Use full names for people. A ticket is
identified by its id, such as "T-1041".

A project is a named product or initiative. A component, service or worker inside a
project (an indexer, a cache, an email worker) is not a project: attach what the text
says about it to the project it belongs to.

Put every person, team, project, customer and ticket you mention anywhere, including
ones you only reference, in the matching top-level list.
"""
