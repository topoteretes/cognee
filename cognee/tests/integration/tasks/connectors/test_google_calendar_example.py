"""CI-safe smoke test for ``examples/guides/google_calendar.py``.

Runs the guide's ``main()`` end-to-end with the Calendar API faked and the LLM and
embeddings mocked, the same way ``test_google_drive_example.py`` does for the Drive guide.
No live Google or LLM credentials are used. The guide syncs twice; the second sync must
fetch through the sync token and report no events, which is what makes re-runs cheap.
"""

import ast
import importlib
import importlib.util
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")
calendar_module = importlib.import_module("cognee.tasks.ingestion.connectors.google_calendar")

EXAMPLE_PATH = pathlib.Path(__file__).parents[5] / "examples" / "guides" / "google_calendar.py"
PRIMARY = "owner@example.com"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("google_calendar_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self, num_retries=0):
        return self._result


class _Events:
    def __init__(self, calendar):
        self.calendar = calendar

    def list(self, **params):
        self.calendar.list_calls.append(params)
        # Nothing changes between the two syncs, so a sync token replays no events.
        items = [] if params.get("syncToken") else list(self.calendar.events_by_id.values())
        return _Req({"items": items, "nextSyncToken": "t1"})


class _CalendarList:
    def list(self, pageToken=None, maxResults=None, showHidden=None):
        return _Req({"items": [{"id": PRIMARY, "summary": PRIMARY, "primary": True}]})


class _FakeCalendarService:
    def __init__(self):
        self.list_calls = []
        self.events_by_id = {
            event_id: {
                "id": event_id,
                "status": "confirmed",
                "summary": summary,
                "description": "Acme reviews the delivery plan.",
                "start": {"dateTime": "2026-10-21T10:00:00+02:00", "timeZone": "Europe/Berlin"},
                "end": {"dateTime": "2026-10-21T11:00:00+02:00", "timeZone": "Europe/Berlin"},
                "organizer": {"email": PRIMARY},
            }
            for event_id, summary in (("evt1", "Planning with Acme"), ("evt2", "Acme review"))
        }

    def events(self):
        return _Events(self)

    def calendarList(self):
        return _CalendarList()


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        return KnowledgeGraph(
            nodes=[KGNode(id="Acme", name="Acme", type="Company", description="A company")],
            edges=[],
        )
    return response_model()


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    # The guide's documented setup: the installed-app flow from two paths.
    monkeypatch.setenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "unused.json")
    monkeypatch.setenv("GOOGLE_CALENDAR_TOKEN_PATH", str(tmp_path / "token.json"))
    for name in ("ACCESS_TOKEN", "REFRESH_TOKEN", "IDS"):
        monkeypatch.delenv(f"GOOGLE_CALENDAR_{name}", raising=False)

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_args, **_kwargs):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return EmbeddingResult(vectors=[[0.0] * self.get_vector_size() for _ in text], tokens=0)

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    service = _FakeCalendarService()
    monkeypatch.setattr(calendar_module, "build_calendar_service", lambda *args, **kwargs: service)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield service

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_example_syncs_then_resyncs_nothing(clean_environment, capsys):
    pytest.importorskip("ladybug")
    service = clean_environment

    await _load_example_module().main()

    stats = [
        ast.literal_eval(line.removeprefix("Sync stats: "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Sync stats: ")
    ]
    assert [run["scanned"] for run in stats] == [2, 0]
    assert all(run["failed"] == 0 for run in stats)
    # The re-sync went through Google's sync token rather than reading everything again.
    assert [call.get("syncToken") for call in service.list_calls] == [None, "t1"]
