"""A window's entries with code references become findings; persisted verbatim and joined by id."""

import importlib
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry, SessionQAEntry
from cognee.tasks.code_memory import (
    ObservedDeclaration,
    code_finding,
    code_findings_from_qa_entries,
    code_findings_from_trace_entries,
    persist_code_findings,
)

persist_module = importlib.import_module("cognee.tasks.code_memory.persist_code_findings")
storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)

REF = {
    "repo": "httpx",
    "enola_id": "3dd612bf822d62516cc184abebddf52b",
    "name": "httpx/_client.Client.send",
    "file": "httpx/_client.py",
    "line": 646,
    "body_hash": "2d5be8556574e8c3",
    "observed_commit": "a4463d044f370aef186f3c64bd79e236e4df9e2a",
    "extractor_version": "0.4.38",
}


def test_only_entries_with_references_become_findings_and_keep_their_text():
    entries = [
        SessionQAEntry(time="t1", question="q", context="", answer="plain", qa_id="qa-1"),
        SessionQAEntry(
            time="t2",
            question="q",
            context="",
            answer="send retries",
            qa_id="qa-2",
            code_refs=[REF],
        ),
    ]
    (finding,) = code_findings_from_qa_entries(entries, user_id="u", session_id="s")
    assert (finding.entry_kind, finding.entry_id, finding.text, finding.time) == (
        "qa",
        "qa-2",
        "send retries",
        "t2",
    )
    (declaration,) = finding.cites
    assert isinstance(declaration, ObservedDeclaration)
    assert declaration.model_dump(include=set(REF)) == REF
    assert declaration.resolution == "exact"


def test_trace_findings_take_the_steps_text_as_the_window_carries_it():
    entries = [
        SessionAgentTraceEntry(trace_id="t-1", origin_function="read", status="success"),
        SessionAgentTraceEntry(
            trace_id="t-2", origin_function="read", status="success", code_refs=[REF]
        ),
    ]
    (finding,) = code_findings_from_trace_entries(
        entries, ["read succeeded.", None], user_id="u", session_id="s"
    )
    assert (finding.entry_kind, finding.entry_id, finding.text) == ("trace", "t-2", "")


def test_ids_are_deterministic_and_identical_references_share_a_declaration():
    first = code_finding(
        user_id="u", session_id="s", entry_id="qa-1", entry_kind="qa", text="a", code_refs=[REF]
    )
    second = code_finding(
        user_id="u", session_id="s", entry_id="qa-2", entry_kind="qa", text="b", code_refs=[REF]
    )
    assert first.id != second.id
    assert first.cites[0].id == second.cites[0].id
    assert (
        code_finding(
            user_id="u", session_id="s", entry_id="qa-1", entry_kind="qa", text="x", code_refs=[]
        ).id
        == first.id
    )
    other_commit = {**REF, "observed_commit": "b" * 40}
    assert (
        code_finding(
            user_id="u",
            session_id="s",
            entry_id="qa-3",
            entry_kind="qa",
            text="c",
            code_refs=[other_commit],
        )
        .cites[0]
        .id
        != first.cites[0].id
    )


def test_a_malformed_reference_is_refused():
    with pytest.raises(ValidationError):
        code_finding(
            user_id="u",
            session_id="s",
            entry_id="qa-1",
            entry_kind="qa",
            text="a",
            code_refs=[{**REF, "enola_id": "not-a-fact-id"}],
        )


@pytest.mark.asyncio
async def test_persist_writes_the_findings_under_the_node_set_then_joins_them(monkeypatch):
    finding = code_finding(
        user_id="u", session_id="s", entry_id="qa-1", entry_kind="qa", text="a", code_refs=[REF]
    )
    add_data_points = AsyncMock()
    link = AsyncMock(return_value={})
    engine = AsyncMock()
    monkeypatch.setattr(storage_module, "add_data_points", add_data_points)
    monkeypatch.setattr(persist_module, "link_observed_declarations", link)
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=engine))

    await persist_code_findings(
        [finding],
        node_set=["user_sessions", "team-a"],
        user=None,
        dataset_id=uuid4(),
        data_id=uuid4(),
        pipeline_run_id=uuid4(),
    )

    (written,), kwargs = add_data_points.call_args
    assert written == [finding] and kwargs["graph_only"] is True
    assert kwargs["ctx"] is None  # no user: stored without provenance
    assert [node_set.name for node_set in finding.belongs_to_set] == ["user_sessions", "team-a"]
    assert [node_set.name for node_set in finding.cites[0].belongs_to_set] == [
        "user_sessions",
        "team-a",
    ]
    link.assert_awaited_once_with(engine, {"httpx"}, ctx=None)


@pytest.mark.asyncio
async def test_persist_without_findings_touches_nothing(monkeypatch):
    add_data_points = AsyncMock()
    monkeypatch.setattr(storage_module, "add_data_points", add_data_points)
    await persist_code_findings(
        [], node_set=None, user=None, dataset_id=None, data_id=None, pipeline_run_id=None
    )
    add_data_points.assert_not_called()
