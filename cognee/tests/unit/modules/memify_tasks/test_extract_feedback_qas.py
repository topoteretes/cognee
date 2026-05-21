import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.tasks.memify.extract_feedback_qas import extract_feedback_qas
from cognee.tasks.memify.feedback_weights_constants import (
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

    mock_session_manager = MagicMock()
    mock_session_manager.is_available = True
    mock_session_manager.get_session = AsyncMock(return_value=entries)

    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            extracted.append(item)

    assert len(extracted) == 1
    assert extracted[0]["qa_id"] == "q1"
    assert extracted[0]["session_id"] == "s1"
    assert extracted[0]["feedback_score"] == 5


@pytest.mark.asyncio
async def test_extract_feedback_qas_respects_session_ids(mock_user):
    mock_session_manager = MagicMock()
    mock_session_manager.is_available = True
    mock_session_manager.get_session = AsyncMock(
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

    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted = []
        async for item in extract_feedback_qas([{}], session_ids=["sA", "sB"]):
            extracted.append(item)

    assert [item["session_id"] for item in extracted] == ["sA", "sB"]
    assert mock_session_manager.get_session.call_count == 2


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

    mock_session_manager = MagicMock()
    mock_session_manager.is_available = True
    mock_session_manager.get_session = AsyncMock(return_value=entries)

    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            extracted.append(item)

    assert [item["qa_id"] for item in extracted] == ["q2", "q1"]


@pytest.mark.asyncio
async def test_extract_feedback_qas_unavailable_session_manager_yields_nothing(mock_user):
    unavailable_session_manager = SessionManager(cache_engine=None)

    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=unavailable_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        extracted = []
        async for item in extract_feedback_qas([{}], session_ids=["s1"]):
            extracted.append(item)

    assert extracted == []


@pytest.mark.asyncio
async def test_extract_feedback_qas_rejects_non_list_session_ids(mock_user):
    mock_session_manager = MagicMock()
    mock_session_manager.get_session = AsyncMock(return_value=[])

    with (
        patch.object(extract_feedback_qas_module, "session_user") as mock_session_user,
        patch.object(
            extract_feedback_qas_module,
            "get_session_manager",
            return_value=mock_session_manager,
        ),
    ):
        mock_session_user.get.return_value = mock_user

        with pytest.raises(CogneeValidationError, match="session_ids must be provided"):
            async for _ in extract_feedback_qas([{}], session_ids="session_1"):
                pass
