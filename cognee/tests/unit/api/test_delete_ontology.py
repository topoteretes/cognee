"""Tests for the delete ontology feature.

Covers:
- OntologyService.delete_ontology: removes file and metadata
- OntologyService.delete_ontology: raises ValueError for unknown key
- OntologyService.delete_ontology: handles missing .owl file gracefully
- DELETE /api/v1/ontologies/{key} endpoint: success, 400, upload-then-delete workflow
"""

import json
import uuid
import pytest
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient

from cognee.api.v1.ontologies.ontologies import OntologyService
from cognee.api.client import app
from cognee.modules.users.methods import get_authenticated_user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_default_user():
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        email="test@example.com",
        is_active=True,
        tenant_id=str(uuid.uuid4()),
    )


@pytest.fixture
def client(test_client, mock_default_user):
    async def override_get_authenticated_user():
        return mock_default_user

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


@pytest.fixture
def ontology_service(tmp_path):
    """OntologyService with base_dir pointed at tmp_path."""
    fake_config = SimpleNamespace(data_root_directory=tmp_path)
    with patch("cognee.api.v1.ontologies.ontologies.get_base_config", return_value=fake_config):
        yield OntologyService()


# ---------------------------------------------------------------------------
# Unit tests for OntologyService.delete_ontology
# ---------------------------------------------------------------------------


class TestOntologyServiceDelete:
    def test_delete_ontology_removes_file_and_metadata(self, ontology_service, tmp_path):
        """delete_ontology removes the .owl file and metadata entry."""
        user = SimpleNamespace(id="user-1")

        user_dir = tmp_path / "user-1"
        user_dir.mkdir()
        owl_file = user_dir / "my_ontology.owl"
        owl_file.write_text("<rdf>test</rdf>")

        metadata = {
            "my_ontology": {
                "filename": "test.owl",
                "size_bytes": 14,
                "uploaded_at": "2024-01-01T00:00:00",
            }
        }
        (user_dir / "metadata.json").write_text(json.dumps(metadata))

        ontology_service.delete_ontology("my_ontology", user)

        assert not owl_file.exists()
        updated_metadata = json.loads((user_dir / "metadata.json").read_text())
        assert "my_ontology" not in updated_metadata

    def test_delete_ontology_raises_for_unknown_key(self, ontology_service, tmp_path):
        """delete_ontology raises ValueError if ontology_key not in metadata."""
        user = SimpleNamespace(id="user-1")

        user_dir = tmp_path / "user-1"
        user_dir.mkdir()
        (user_dir / "metadata.json").write_text(json.dumps({}))

        with pytest.raises(ValueError, match="not found"):
            ontology_service.delete_ontology("nonexistent", user)

    def test_delete_ontology_handles_missing_owl_file(self, ontology_service, tmp_path):
        """delete_ontology succeeds even if .owl file is already missing."""
        user = SimpleNamespace(id="user-1")

        user_dir = tmp_path / "user-1"
        user_dir.mkdir()

        metadata = {
            "orphan_key": {
                "filename": "gone.owl",
                "size_bytes": 0,
                "uploaded_at": "2024-01-01T00:00:00",
            }
        }
        (user_dir / "metadata.json").write_text(json.dumps(metadata))

        ontology_service.delete_ontology("orphan_key", user)

        updated_metadata = json.loads((user_dir / "metadata.json").read_text())
        assert "orphan_key" not in updated_metadata

    def test_delete_ontology_preserves_other_entries(self, ontology_service, tmp_path):
        """Deleting one ontology preserves other ontology entries."""
        user = SimpleNamespace(id="user-1")

        user_dir = tmp_path / "user-1"
        user_dir.mkdir()

        metadata = {
            "keep_this": {"filename": "keep.owl", "size_bytes": 10, "uploaded_at": "2024-01-01"},
            "delete_this": {"filename": "del.owl", "size_bytes": 10, "uploaded_at": "2024-01-01"},
        }
        (user_dir / "metadata.json").write_text(json.dumps(metadata))
        (user_dir / "delete_this.owl").write_text("<rdf/>")
        (user_dir / "keep_this.owl").write_text("<rdf/>")

        ontology_service.delete_ontology("delete_this", user)

        updated_metadata = json.loads((user_dir / "metadata.json").read_text())
        assert "keep_this" in updated_metadata
        assert "delete_this" not in updated_metadata
        assert (user_dir / "keep_this.owl").exists()

    def test_delete_ontology_rejects_path_traversal_key(self, ontology_service, tmp_path):
        """delete_ontology rejects keys that resolve outside the user directory."""
        user = SimpleNamespace(id="user-1")
        user_dir = tmp_path / "user-1"
        user_dir.mkdir()

        metadata = {
            "../escape": {
                "filename": "escape.owl",
                "size_bytes": 10,
                "uploaded_at": "2024-01-01",
            }
        }
        (user_dir / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(ValueError, match="Invalid ontology key"):
            ontology_service.delete_ontology("../escape", user)


# ---------------------------------------------------------------------------
# API endpoint tests for DELETE /api/v1/ontologies/{key}
# ---------------------------------------------------------------------------


def test_delete_ontology_endpoint_not_found(client):
    """DELETE with unknown key returns 400."""
    response = client.delete("/api/v1/ontologies/nonexistent_key")
    assert response.status_code == 400
    assert "not found" in response.json()["error"]


def test_upload_then_delete_ontology_endpoint(client):
    """Upload an ontology, verify it's listed, delete it, verify it's gone."""
    ontology_content = (
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    )
    unique_key = f"test_del_{uuid.uuid4().hex[:8]}"

    # Upload
    upload_resp = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", ontology_content, "application/xml"))],
        data={"ontology_key": unique_key},
    )
    assert upload_resp.status_code == 200

    # Verify listed
    list_resp = client.get("/api/v1/ontologies")
    assert list_resp.status_code == 200
    assert unique_key in list_resp.json()

    # Delete
    delete_resp = client.delete(f"/api/v1/ontologies/{unique_key}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "success"
    assert delete_resp.json()["ontology_key"] == unique_key

    # Verify gone
    list_resp2 = client.get("/api/v1/ontologies")
    assert unique_key not in list_resp2.json()


def test_delete_ontology_twice_returns_400(client):
    """Deleting the same ontology twice should return 400 on second attempt."""
    ontology_content = (
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    )
    unique_key = f"test_double_del_{uuid.uuid4().hex[:8]}"

    # Upload
    client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", ontology_content, "application/xml"))],
        data={"ontology_key": unique_key},
    )

    # First delete succeeds
    resp1 = client.delete(f"/api/v1/ontologies/{unique_key}")
    assert resp1.status_code == 200

    # Second delete fails
    resp2 = client.delete(f"/api/v1/ontologies/{unique_key}")
    assert resp2.status_code == 400
    assert "not found" in resp2.json()["error"]
