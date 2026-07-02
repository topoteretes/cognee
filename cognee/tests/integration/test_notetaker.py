"""Integration tests for the meeting/notetaker bot (#3614).

Uses the mocked-LLM approach (#3601): every test is deterministic and needs no
real API keys. The tests assert the two make-or-break preconditions —
date-anchoring into the transcript text, and temporal recall being *scoped to
the series dataset* — plus the ingest/recall/forget wiring.
"""

from unittest.mock import patch, AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.tasks.notetaker.normalize import normalize_transcript
from cognee.api.v1.search import SearchType


# --- Normalization / date-anchoring (pure, no cognee runtime) -----------------


def test_date_anchoring_positive():
    """The absolute meeting datetime is injected into the chunk text itself."""
    turns = [("Alice", "Let's release v1 today.", "2026-06-10 10:00")]
    norm = normalize_transcript(turns, meeting_id="m1")
    assert norm.startswith("[2026-06-10 10:00] Alice:")
    assert "Let's release v1 today." in norm


def test_date_anchoring_negative_falls_back():
    """A missing timestamp degrades to a deterministic default, never crashes."""
    turns = [("Alice", "Let's release v1 today.", None)]
    norm = normalize_transcript(turns, meeting_id="m1")
    assert "[1970-01-01 00:00] Alice: (meeting_id=m1)" in norm


def test_citation_prefix_round_trips_into_text():
    """Speaker + permalink land in the text so citations can ground to them."""
    turns = [("Alice", "Deploy blocked on staging creds", "2026-06-24 14:32")]
    norm = normalize_transcript(turns, meeting_id="m1", permalink="https://example.com/m1")
    assert "[2026-06-24 14:32] Alice: (meeting_id=m1, permalink=https://example.com/m1)" in norm
    assert "Deploy blocked on staging creds" in norm


# --- cognify: temporal vs custom graph_model exclusivity ----------------------


class _DummyGraphModel(BaseModel):
    title: str = "x"


@pytest.mark.asyncio
async def test_temporal_cognify_takes_temporal_path():
    """temporal_cognify=True routes to get_temporal_tasks; the custom graph_model
    branch (get_default_tasks) is not taken — documents the exclusivity."""
    import cognee

    with (
        patch("cognee.api.v1.cognify.cognify.get_default_tasks") as mock_default,
        patch("cognee.api.v1.cognify.cognify.get_temporal_tasks") as mock_temporal,
    ):
        mock_temporal.return_value = []  # empty pipeline; we only assert dispatch
        try:
            await cognee.cognify(
                datasets=["exclusivity_series"],
                temporal_cognify=True,
                graph_model=_DummyGraphModel,
            )
        except Exception:
            # An empty task list may raise downstream — irrelevant to the dispatch assertion.
            pass

        mock_temporal.assert_called_once()
        mock_default.assert_not_called()


# --- recall: scoped to the series via SearchType.TEMPORAL ---------------------


@pytest.mark.asyncio
async def test_recall_scopes_temporal_search_to_series():
    """The recall endpoint must route through cognee.search with TEMPORAL, the
    series as the dataset, and the focused action-items prompt."""
    from cognee.api.v1 import notetaker

    with patch("cognee.search", new=AsyncMock(return_value=["stub"])) as mock_search:
        resp = await notetaker.notetaker_recall(
            series_id="eng_standup",
            query="What are the action items?",
            query_type="action_items",
            user=None,
        )

    assert resp["series_id"] == "eng_standup"
    mock_search.assert_awaited_once()
    kwargs = mock_search.await_args.kwargs
    assert kwargs["query_type"] == SearchType.TEMPORAL
    assert kwargs["datasets"] == ["eng_standup"]
    assert kwargs["system_prompt_path"].endswith("notetaker_action_items.txt")
    assert kwargs["include_references"] is True


@pytest.mark.asyncio
async def test_recall_rejects_unknown_query_type():
    from fastapi import HTTPException
    from cognee.api.v1 import notetaker

    with pytest.raises(HTTPException) as exc:
        await notetaker.notetaker_recall(series_id="s", query="q", query_type="bogus", user=None)
    assert exc.value.status_code == 400


# --- forget: series vs single occurrence -------------------------------------


@pytest.mark.asyncio
async def test_forget_series_uses_dataset():
    from cognee.api.v1 import notetaker

    with patch("cognee.forget", new=AsyncMock(return_value={})) as mock_forget:
        await notetaker.notetaker_forget(notetaker.ForgetPayload(series_id="series_a"), user=None)
    mock_forget.assert_awaited_once()
    kwargs = mock_forget.await_args.kwargs
    assert kwargs["dataset"] == "series_a"
    assert "data_id" not in kwargs  # whole-series wipe, not a single occurrence


@pytest.mark.asyncio
async def test_forget_meeting_uses_data_id_within_series():
    from cognee.api.v1 import notetaker

    data_id = str(uuid4())
    with patch("cognee.forget", new=AsyncMock(return_value={})) as mock_forget:
        await notetaker.notetaker_forget(
            notetaker.ForgetPayload(series_id="series_a", data_id=data_id), user=None
        )
    kwargs = mock_forget.await_args.kwargs
    assert str(kwargs["data_id"]) == data_id
    assert kwargs["dataset"] == "series_a"


@pytest.mark.asyncio
async def test_forget_meeting_rejects_bad_uuid():
    from fastapi import HTTPException
    from cognee.api.v1 import notetaker

    with pytest.raises(HTTPException) as exc:
        await notetaker.notetaker_forget(
            notetaker.ForgetPayload(series_id="s", data_id="not-a-uuid"), user=None
        )
    assert exc.value.status_code == 400
