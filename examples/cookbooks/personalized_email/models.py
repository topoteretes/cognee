"""The inbox graph model: what the draft agent needs to know before it writes a reply.

The model is shaped by the questions the agent asks, not by the shape of an email:

- "Who is this sender?"             -> Person, linked to an Organization.
- "What did we discuss and decide?" -> Meeting, with its attendees and decisions.
- "What do I still owe them?"       -> Commitment, with an owner, a recipient and a due date.
- "Which conversation is this?"     -> EmailThread, with its participants.

Every type declares ``identity_fields``, so the same person named in a Granola note and in an
email becomes one node instead of two. References use ``FromIdentity``: the LLM answers a
name and cognee resolves it against the nodes extracted from the same text.
"""

import re
from typing import Annotated

from pydantic import field_validator

from cognee.low_level import DataPoint, FromIdentity


class Organization(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Person(DataPoint):
    name: str
    email: str | None = None
    role: str | None = None
    works_at: Annotated[Organization, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Meeting(DataPoint):
    title: str
    date: str | None = None
    attendees: Annotated[list[Person], FromIdentity()] = []
    decisions: list[str] = []
    metadata: dict = {"index_fields": ["title"], "identity_fields": ["title"]}


class Commitment(DataPoint):
    description: str
    owner: Annotated[Person, FromIdentity()] | None = None
    owed_to: Annotated[Person, FromIdentity()] | None = None
    due: str | None = None
    status: str | None = None
    made_in: Annotated[Meeting, FromIdentity()] | None = None
    metadata: dict = {"index_fields": ["description"], "identity_fields": ["description"]}


class EmailThread(DataPoint):
    subject: str
    participants: Annotated[list[Person], FromIdentity()] = []
    metadata: dict = {"index_fields": ["subject"], "identity_fields": ["subject"]}

    # "Re: Recap" and "Recap" are the same thread.
    @field_validator("subject")
    @classmethod
    def _strip_reply_prefix(cls, subject: str) -> str:
        return re.sub(r"^((re|fwd?|aw):\s*)+", "", subject.strip(), flags=re.IGNORECASE)


class InboxGraph(DataPoint):
    people: list[Person] = []
    organizations: list[Organization] = []
    meetings: list[Meeting] = []
    commitments: list[Commitment] = []
    threads: list[EmailThread] = []


EXTRACTION_PROMPT = """
Extract the people, organizations, meetings, commitments and email threads in the text.

Use full names for people, exactly as the text writes them, and add their email address and
role when the text gives them. A meeting is identified by its title. An email thread is
identified by its subject.

A commitment is a promise someone made to do something: sending a document, answering a
question, setting something up. Record who owes it (owner), who it is for (owed_to), the due
date as written, and status "open" unless the text says it was done.

Put every person, organization, meeting and commitment you mention anywhere, including ones
you only reference, in the matching top-level list.
"""
