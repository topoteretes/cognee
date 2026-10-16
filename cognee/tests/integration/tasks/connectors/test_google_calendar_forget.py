# Adapted from topoteretes/cognee-community (Apache-2.0), PR #370, commit 5e78950.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Forget-on-delete must remove an event's cognified graph content, not just its
dlt row. Syncs three events, cognifies (LLM + embeddings mocked, no live creds),
then deletes one event and cancels one occurrence of a series. The deleted
event's extracted entity must leave the graph, the series entity (a cancelled
occurrence never forgets its series) and the untouched event's entity stay.
"""

import importlib

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.tasks.ingestion.connectors import google_calendar_source

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "gcal_forget_test"
PRIMARY = "owner@example.com"
# Distinctive, unique tokens so each event maps to exactly one graph entity.
ALPHA = "Alphacorp"
BRAVO = "Bravocorp"
CHARLIE = "Charliecorp"
TOKENS = (ALPHA, BRAVO, CHARLIE)


class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self, num_retries=0):
        return self._result


class _Events:
    def __init__(self, calendar):
        self.calendar = calendar

    def list(self, calendarId, maxResults, singleEvents, syncToken=None, pageToken=None):
        if syncToken:
            items = self.calendar.changes[int(syncToken.removeprefix("t")) :]
        else:
            items = list(self.calendar.live.values())
        return _Req({"items": items, "nextSyncToken": f"t{len(self.calendar.changes)}"})


class _CalendarList:
    def list(self, pageToken=None, maxResults=None, showHidden=None):
        return _Req({"items": [{"id": PRIMARY, "summary": "Owner", "primary": True}]})


class FakeCalendar:
    """One calendar whose sync token ``tN`` replays every change after the Nth."""

    def __init__(self):
        self.live = {}
        self.changes = []

    def events(self):
        return _Events(self)

    def calendarList(self):
        return _CalendarList()

    def put(self, event_id, summary, description, **fields):
        event = {
            "id": event_id,
            "status": "confirmed",
            "summary": summary,
            "description": description,
            "start": {"dateTime": "2026-10-12T10:00:00+00:00"},
            "end": {"dateTime": "2026-10-12T11:00:00+00:00"},
            **fields,
        }
        self.live[event_id] = event
        self.changes.append(event)

    def delete(self, event_id):
        del self.live[event_id]
        self.changes.append({"id": event_id, "status": "cancelled"})

    def cancel_occurrence(self, series_id, occurrence_id):
        self.changes.append(
            {"id": occurrence_id, "status": "cancelled", "recurringEventId": series_id}
        )


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity named after whichever token appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        name = next((t for t in TOKENS if text_input and t in text_input), None)
        nodes = (
            [KGNode(id=name, name=name, type="Company", description=f"{name} entity")]
            if name
            else []
        )
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    token = token.lower()
    for _nid, props in nodes:
        if any(token in str(v).lower() for v in (props or {}).values()):
            return True
    return False


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_a, **_k):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return [[0.0] * self.get_vector_size() for _ in text]

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _sync(calendar):
    await cognee.add(
        google_calendar_source(service=calendar),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


@pytest.mark.asyncio
async def test_deleted_event_is_forgotten_and_a_cancelled_occurrence_keeps_the_series(
    clean_environment,
):
    calendar = FakeCalendar()
    calendar.put(
        "series1",
        "Daily sync",
        f"{ALPHA} builds the delivery drone.",
        recurrence=["RRULE:FREQ=DAILY;COUNT=3"],
    )
    calendar.put("finance1", "Finance review", f"{BRAVO} pays for it.")
    calendar.put("rocket1", "Rocket demo", f"{CHARLIE} rockets.")

    await _sync(calendar)
    for token in TOKENS:
        assert await _graph_has(token), f"{token} should be in the graph after the first sync"

    calendar.cancel_occurrence("series1", "series1_20261013T100000Z")
    calendar.delete("rocket1")
    await _sync(calendar)

    assert await _graph_has(ALPHA), "cancelling one occurrence must keep the series"
    assert await _graph_has(BRAVO), "an untouched event must stay"
    assert not await _graph_has(CHARLIE), "a deleted event must be forgotten from the graph"
