"""Memories about code, kept verbatim and joined to the code graph by id.

A memory entry that names the code it is about (``QAEntry.code_refs`` /
``TraceEntry.code_refs``, see ``cognee/memory/code_refs.py``) is persisted as
one ``CodeFinding`` -- the entry's text as written -- citing one
``ObservedDeclaration`` per reference: the declaration at the version the
author saw it. Both are deterministic in their ids, so a retried persist
writes the same nodes, and identical references from different findings
share one declaration node (its findings are then the memories about that
version of the code).

An ``ObservedDeclaration`` is joined to the ``CodeSymbol`` its code lives in by
a ``resolves_to`` edge that ``link_observed_declarations`` writes and keeps
current: by the declaration's enola id, or by its body hash when the id is
gone and exactly one declaration of the repository has that body. Neither
node is ever changed after it is written; what the code looks like now is
read from the symbol at query time.
"""

import json
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.memory.code_refs import CodeReference

# Edge names. ``cites`` is written by add_data_points from CodeFinding.cites;
# ``resolves_to`` by link_observed_declarations, by id through the graph engine.
CITES_RELATIONSHIP = "cites"
RESOLVES_TO_RELATIONSHIP = "resolves_to"


class ObservedDeclaration(DataPoint):
    """One declaration at the version a memory's author saw it: a ``CodeReference`` as a node.

    The fields are the reference's, unchanged. ``name`` is the declaration's
    qualified name, as enola names it.
    """

    name: str
    repo: str
    enola_id: str
    file: str
    line: int
    body_hash: str
    decl_hash: str | None = None
    observed_commit: str
    extractor_version: str
    resolution: str
    metadata: dict = {"index_fields": []}


class CodeFinding(DataPoint):
    """A memory entry about code, verbatim, with the declarations it cites.

    ``entry_kind`` is ``qa`` (a ``QAEntry``; the text is its answer) or
    ``trace`` (a ``TraceEntry``; the text is the step's feedback as the trace
    window carries it). ``entry_id`` is the entry's id in the session cache.
    """

    name: str
    text: str
    entry_kind: str
    entry_id: str
    session_id: str
    user_id: str
    time: str | None = None
    cites: list[ObservedDeclaration]
    metadata: dict = {"index_fields": []}


def observed_declaration_id(reference: CodeReference) -> UUID:
    """The node id of a reference: identical references share one node."""
    key = json.dumps(reference.model_dump(), sort_keys=True, separators=(",", ":"))
    return uuid5(NAMESPACE_OID, "cognee:observed_declaration:" + key)


def code_finding_id(user_id: str, session_id: str, entry_id: str) -> UUID:
    """The node id of a session entry's finding, stable across retries."""
    key = ":".join(f"{len(part)}:{part}" for part in (user_id, session_id, entry_id))
    return uuid5(NAMESPACE_OID, "cognee:code_finding:" + key)


def observed_declaration(reference: dict | CodeReference) -> ObservedDeclaration:
    """The node of one reference, validated as a ``CodeReference`` first."""
    validated = CodeReference.model_validate(reference)
    return ObservedDeclaration(id=observed_declaration_id(validated), **validated.model_dump())


def code_finding(
    *,
    user_id: str,
    session_id: str,
    entry_id: str,
    entry_kind: str,
    text: str,
    code_refs: list[dict],
    time: str | None = None,
) -> CodeFinding:
    """The finding of one session entry and the declarations it cites."""
    return CodeFinding(
        id=code_finding_id(user_id, session_id, entry_id),
        name=entry_id,
        text=text,
        entry_kind=entry_kind,
        entry_id=entry_id,
        session_id=session_id,
        user_id=user_id,
        time=time,
        cites=[observed_declaration(reference) for reference in code_refs],
    )
