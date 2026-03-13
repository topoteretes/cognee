"""Unit tests for the /api/v1/skills REST router.

Mocks the Skills client so we test HTTP plumbing without needing
a real graph/vector database or LLM calls.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_SKILL_ID = "test-skill"
MOCK_AMENDMENT_ID = "amend-abc123"


@pytest.fixture(scope="session")
def test_client():
    from cognee.api.v1.skills.routers import get_skills_router

    app = FastAPI()
    app.include_router(get_skills_router(), prefix="/api/v1/skills")

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    test_client.app.dependency_overrides[get_authenticated_user] = (
        override_get_authenticated_user
    )
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _mock_skills_client():
    """Return a mock Skills instance with all methods pre-configured."""
    mock = AsyncMock()

    mock.ingest = AsyncMock(return_value=None)

    mock.upsert = AsyncMock(
        return_value={"unchanged": 1, "updated": 0, "added": 0, "removed": 0}
    )

    mock.remove = AsyncMock(return_value=True)

    mock.list = AsyncMock(
        return_value=[
            {
                "skill_id": MOCK_SKILL_ID,
                "name": "Test Skill",
                "instruction_summary": "A test skill",
                "tags": ["test"],
                "complexity": "simple",
            }
        ]
    )

    mock.load = AsyncMock(
        return_value={
            "skill_id": MOCK_SKILL_ID,
            "name": "Test Skill",
            "instructions": "Do the thing.",
            "instruction_summary": "A test skill",
            "description": "Test description",
            "tags": ["test"],
            "complexity": "simple",
            "source_path": "/tmp/skills/test-skill",
            "task_patterns": [],
        }
    )

    mock.execute = AsyncMock(
        return_value={
            "output": "Done.",
            "skill_id": MOCK_SKILL_ID,
            "model": "gpt-4o-mini",
            "latency_ms": 150,
            "success": True,
            "error": None,
            "quality_score": 0.9,
            "quality_reason": "Good output",
        }
    )

    mock.observe = AsyncMock(
        return_value={
            "selected_skill_id": MOCK_SKILL_ID,
            "success_score": 0.9,
        }
    )

    mock.inspect = AsyncMock(
        return_value={
            "inspection_id": "insp-001",
            "skill_id": MOCK_SKILL_ID,
            "skill_name": "Test Skill",
            "failure_category": "instruction_gap",
            "root_cause": "Missing guard clause",
            "severity": "high",
            "improvement_hypothesis": "Add guard for empty input",
            "analyzed_run_count": 5,
            "avg_success_score": 0.2,
            "inspection_confidence": 0.88,
        }
    )

    mock.preview_amendify = AsyncMock(
        return_value={
            "amendment_id": MOCK_AMENDMENT_ID,
            "skill_id": MOCK_SKILL_ID,
            "skill_name": "Test Skill",
            "inspection_id": "insp-001",
            "original_instructions": "Do the thing.",
            "amended_instructions": "Do the thing. Handle empty input.",
            "change_explanation": "Added guard clause",
            "expected_improvement": "No more failures on empty input",
            "status": "proposed",
            "amendment_confidence": 0.82,
            "pre_amendment_avg_score": 0.2,
        }
    )

    mock.amendify = AsyncMock(
        return_value={
            "success": True,
            "status": "applied",
            "skill_id": MOCK_SKILL_ID,
        }
    )

    mock.rollback_amendify = AsyncMock(return_value=True)

    mock.evaluate_amendify = AsyncMock(
        return_value={
            "pre_avg": 0.2,
            "post_avg": 0.91,
            "improvement": 0.71,
            "run_count": 8,
            "recommendation": "keep",
        }
    )

    mock.auto_amendify = AsyncMock(
        return_value={
            "inspection": {"inspection_id": "insp-001"},
            "amendment": {"amendment_id": MOCK_AMENDMENT_ID},
            "applied": {"success": True},
        }
    )

    return mock


# ---------------------------------------------------------------------------
# Tests — management
# ---------------------------------------------------------------------------

SKILLS_CLIENT_PATH = "cognee.api.v1.skills.routers.get_skills_router.skills"


class TestListSkills:
    def test_list_returns_skills(self, client):
        mock = _mock_skills_client()
        with patch(
            "cognee.cognee_skills.client.skills", mock
        ):
            response = client.get("/api/v1/skills?node_set=skills")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["skill_id"] == MOCK_SKILL_ID

    def test_list_empty(self, client):
        mock = _mock_skills_client()
        mock.list = AsyncMock(return_value=[])
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.get("/api/v1/skills")

        assert response.status_code == 200
        assert response.json() == []


class TestLoadSkill:
    def test_load_existing(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.get(f"/api/v1/skills/{MOCK_SKILL_ID}")

        assert response.status_code == 200
        data = response.json()
        assert data["skill_id"] == MOCK_SKILL_ID
        assert "instructions" in data

    def test_load_not_found(self, client):
        mock = _mock_skills_client()
        mock.load = AsyncMock(return_value=None)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.get("/api/v1/skills/nonexistent")

        assert response.status_code == 404


class TestRemoveSkill:
    def test_remove_existing(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.delete(f"/api/v1/skills/{MOCK_SKILL_ID}")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_remove_not_found(self, client):
        mock = _mock_skills_client()
        mock.remove = AsyncMock(return_value=False)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.delete("/api/v1/skills/nonexistent")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests — ingestion
# ---------------------------------------------------------------------------


class TestIngest:
    def test_ingest_missing_folder(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/ingest",
                json={"skills_folder": "/nonexistent/path"},
            )

        assert response.status_code == 400
        assert "not found" in response.json()["error"].lower()

    def test_ingest_valid_folder(self, client, tmp_path):
        mock = _mock_skills_client()
        folder = str(tmp_path)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/ingest",
                json={"skills_folder": folder},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock.ingest.assert_called_once()


class TestUpsert:
    def test_upsert_valid_folder(self, client, tmp_path):
        mock = _mock_skills_client()
        folder = str(tmp_path)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/upsert",
                json={"skills_folder": folder},
            )

        assert response.status_code == 200
        data = response.json()
        assert "unchanged" in data
        mock.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — execution & observation
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_success(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/execute",
                json={
                    "skill_id": MOCK_SKILL_ID,
                    "task_text": "Do something",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["output"] == "Done."
        assert data["quality_score"] == 0.9


class TestObserve:
    def test_observe_records_run(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/observe",
                json={
                    "task_text": "Do something",
                    "selected_skill_id": MOCK_SKILL_ID,
                    "success_score": 0.9,
                },
            )

        assert response.status_code == 200
        mock.observe.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — self-improvement
# ---------------------------------------------------------------------------


class TestInspect:
    def test_inspect_with_failures(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/inspect",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["failure_category"] == "instruction_gap"
        assert data["root_cause"] == "Missing guard clause"

    def test_inspect_insufficient_failures(self, client):
        mock = _mock_skills_client()
        mock.inspect = AsyncMock(return_value=None)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/inspect",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["result"] is None


class TestPreviewAmendify:
    def test_preview_returns_amendment(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/preview-amendify",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["amendment_id"] == MOCK_AMENDMENT_ID
        assert data["amendment_confidence"] == 0.82

    def test_preview_no_amendment(self, client):
        mock = _mock_skills_client()
        mock.preview_amendify = AsyncMock(return_value=None)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/preview-amendify",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        assert response.json()["result"] is None


class TestAmendify:
    def test_amendify_applies(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/amendify",
                json={"amendment_id": MOCK_AMENDMENT_ID},
            )

        assert response.status_code == 200
        assert response.json()["success"] is True


class TestRollback:
    def test_rollback_success(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/rollback-amendify",
                json={"amendment_id": MOCK_AMENDMENT_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["amendment_id"] == MOCK_AMENDMENT_ID

    def test_rollback_failure(self, client):
        mock = _mock_skills_client()
        mock.rollback_amendify = AsyncMock(return_value=False)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/rollback-amendify",
                json={"amendment_id": "bad-id"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is False


class TestEvaluateAmendify:
    def test_evaluate_returns_scores(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/evaluate-amendify",
                json={"amendment_id": MOCK_AMENDMENT_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pre_avg"] == 0.2
        assert data["post_avg"] == 0.91
        assert data["recommendation"] == "keep"


class TestAutoAmendify:
    def test_auto_amendify_full_pipeline(self, client):
        mock = _mock_skills_client()
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/auto-amendify",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        data = response.json()
        assert "inspection" in data
        assert "amendment" in data
        assert "applied" in data

    def test_auto_amendify_insufficient_failures(self, client):
        mock = _mock_skills_client()
        mock.auto_amendify = AsyncMock(return_value=None)
        with patch("cognee.cognee_skills.client.skills", mock):
            response = client.post(
                "/api/v1/skills/auto-amendify",
                json={"skill_id": MOCK_SKILL_ID},
            )

        assert response.status_code == 200
        assert response.json()["result"] is None
