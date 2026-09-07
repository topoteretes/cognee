import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.tasks.memify.extract_feedback_qas import (
    build_implicit_rating_map,
    extract_feedback_qas,
    resolve_feedback,
)
from cognee.tasks.memify.feedback_weights_constants import (
    FEEDBACK_SOURCE_EXPLICIT,
    FEEDBACK_SOURCE_IMPLICIT,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)

extract_feedback_qas_module = sys.modules["cognee.tasks.memify.extract_feedback_qas"]


def _make_entry(**kwargs) -> SessionQAEntry:
    defaults = {
        "time": "2026-01-01T10:00:00",
        "question": "Test question",
        "context": "Test context",
        "answer": "Test answer",
    }
    return SessionQAEntry(**{**defaults, **kwargs})


def _feedback_row(qa_ids, rating, raw_text="that was wrong", row_id="f1"):
    return {
        "id": row_id,
        "kind": "feedback",
        "created_at": "2026-01-01T10:05:00",
        "raw_text": raw_text,
        "referenced_qa_ids": qa_ids,
        "referenced_qa_rating": rating,
        "influencing_context_ids": [],
        "candidate_context_entries": [],
    }


def _session_manager(entries=None, context_rows=None, **overrides):
    manager = MagicMock()
    manager.is_available = True
    manager.get_session = AsyncMock(return_value=entries if entries is not None else [])
    manager.get_session_context_entries = AsyncMock(
        return_value=context_rows if context_rows is not None else []
    )
    for key, value in overrides.items():
        setattr(manager, key, value)
    return manager


async def _collect(session_manager, user, session_ids):
    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=session_manager,
        ),
    ):
        mock_session_user.get.return_value = user

        extracted = []
        async for item in extract_feedback_qas([{}], session_ids=session_ids):
            extracted.append(item)
    return extracted


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = "u1"
    return user


@pytest.mark.asyncio
async def test_extract_feedback_qas_filters_eligible_entries(mock_user):
    entries = [
        _make_entry(
            qa_id="q1",
            feedback_score=5,
            feedback_text="great",
            used_graph_element_ids={"node_ids": ["n1"], "edge_ids": ["e1"]},
            memify_metadata=None,
        ),
        _make_entry(
            qa_id="q2",
            time="2026-01-01T10:01:00",
            feedback_score=3,
            used_graph_element_ids={"node_ids": ["n2"]},
            memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: True},
        ),
        _make_entry(
            qa_id="q3",
            time="2026-01-01T10:02:00",
            feedback_score=None,
            used_graph_element_ids={"node_ids": ["n3"]},
        ),
    ]

    extracted = await _collect(_session_manager(entries=entries), mock_user, ["s1"])

    assert len(extracted) == 1
    assert extracted[0]["qa_id"] == "q1"
    assert extracted[0]["session_id"] == "s1"
    assert extracted[0]["feedback_score"] == 5
    assert extracted[0]["feedback_source"] == FEEDBACK_SOURCE_EXPLICIT
    assert extracted[0]["feedback_text"] == "great"


@pytest.mark.asyncio
async def test_extract_feedback_qas_yields_rated_rows_without_ids_so_they_get_marked(mock_user):
    entries = [_make_entry(qa_id="q1", feedback_score=4, used_graph_element_ids=None)]

    extracted = await _collect(_session_manager(entries=entries), mock_user, ["s1"])

    assert [item["qa_id"] for item in extracted] == ["q1"]
    assert extracted[0]["used_graph_element_ids"] is None


@pytest.mark.asyncio
async def test_extract_feedback_qas_uses_implicit_rating_when_no_explicit_score(mock_user):
    entries = [
        _make_entry(qa_id="q1", feedback_score=None, used_graph_element_ids={"node_ids": ["n1"]}),
    ]
    context_rows = [_feedback_row(["q1"], 2, raw_text="no, that is not what I asked")]

    extracted = await _collect(
        _session_manager(entries=entries, context_rows=context_rows), mock_user, ["s1"]
    )

    assert len(extracted) == 1
    assert extracted[0]["feedback_score"] == 2
    assert extracted[0]["feedback_source"] == FEEDBACK_SOURCE_IMPLICIT
    assert extracted[0]["feedback_text"] == "no, that is not what I asked"


@pytest.mark.asyncio
async def test_extract_feedback_qas_explicit_score_wins_over_implicit(mock_user):
    entries = [
        _make_entry(
            qa_id="q1",
            feedback_score=5,
            feedback_text="perfect",
            used_graph_element_ids={"node_ids": ["n1"]},
        ),
    ]
    context_rows = [_feedback_row(["q1"], 1)]

    extracted = await _collect(
        _session_manager(entries=entries, context_rows=context_rows), mock_user, ["s1"]
    )

    assert extracted[0]["feedback_score"] == 5
    assert extracted[0]["feedback_source"] == FEEDBACK_SOURCE_EXPLICIT
    assert extracted[0]["feedback_text"] == "perfect"


@pytest.mark.asyncio
async def test_extract_feedback_qas_implicit_rating_does_not_revive_applied_rows(mock_user):
    entries = [
        _make_entry(
            qa_id="q1",
            feedback_score=None,
            used_graph_element_ids={"node_ids": ["n1"]},
            memify_metadata={MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY: True},
        ),
    ]
    context_rows = [_feedback_row(["q1"], 4)]

    extracted = await _collect(
        _session_manager(entries=entries, context_rows=context_rows), mock_user, ["s1"]
    )

    assert extracted == []


def test_build_implicit_rating_map_latest_row_wins_and_ignores_garbage():
    rows = [
        _feedback_row(["q1"], 2, raw_text="first", row_id="f1"),
        _feedback_row(["q1", "q2"], 5, raw_text="second", row_id="f2"),
        _feedback_row(["q3"], None, row_id="f3"),
        _feedback_row(["q4"], 9, row_id="f4"),
        {"kind": "context", "id": "c1", "referenced_qa_ids": ["q5"], "referenced_qa_rating": 3},
        "not-a-row",
    ]

    ratings = build_implicit_rating_map(rows)

    assert ratings == {"q1": (5, "second"), "q2": (5, "second")}


def test_resolve_feedback_returns_none_without_any_rating():
    entry = _make_entry(qa_id="q1", feedback_score=None)

    assert resolve_feedback(entry, {}) is None
    assert resolve_feedback(entry, {"q1": (3, "")}) == (3, FEEDBACK_SOURCE_IMPLICIT, None)


@pytest.mark.asyncio
async def test_extract_feedback_qas_respects_session_ids(mock_user):
    session_manager = _session_manager(
        get_session=AsyncMock(
            side_effect=[
                [
                    _make_entry(
                        qa_id="qa-a",
                        feedback_score=4,
                        used_graph_element_ids={"node_ids": ["n1"]},
                    )
                ],
                [
                    _make_entry(
                        qa_id="qa-b",
                        time="2026-01-01T10:01:00",
                        feedback_score=2,
                        used_graph_element_ids={"edge_ids": ["e1"]},
                    )
                ],
            ]
        )
    )

    extracted = await _collect(session_manager, mock_user, ["sA", "sB"])

    assert [item["session_id"] for item in extracted] == ["sA", "sB"]
    assert session_manager.get_session.call_count == 2
    assert session_manager.get_session_context_entries.call_count == 2


@pytest.mark.asyncio
async def test_extract_feedback_qas_preserves_session_entry_order(mock_user):
    entries = [
        _make_entry(
            qa_id="q2",
            time="2026-01-01T11:00:00",
            feedback_score=4,
            used_graph_element_ids={"node_ids": ["n2"]},
        ),
        _make_entry(
            qa_id="q1",
            time="2026-01-01T10:00:00",
            feedback_score=5,
            used_graph_element_ids={"node_ids": ["n1"]},
        ),
    ]

    extracted = await _collect(_session_manager(entries=entries), mock_user, ["s1"])

    assert [item["qa_id"] for item in extracted] == ["q2", "q1"]


@pytest.mark.asyncio
async def test_extract_feedback_qas_unavailable_session_manager_yields_nothing(mock_user):
    unavailable_session_manager = SessionManager(cache_engine=None)

    extracted = await _collect(unavailable_session_manager, mock_user, ["s1"])

    assert extracted == []


@pytest.mark.asyncio
async def test_extract_feedback_qas_rejects_non_list_session_ids(mock_user):
    with pytest.raises(CogneeValidationError, match="session_ids must be provided"):
        await _collect(_session_manager(), mock_user, "session_1")
