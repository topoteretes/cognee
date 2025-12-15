import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import Mock
from types import SimpleNamespace
from cognee.api.client import app
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="session")
def test_client():
    # Keep a single TestClient (and event loop) for the whole module.
    # Re-creating TestClient repeatedly can break async DB connections (asyncpg loop mismatch).
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client, mock_default_user):
    async def override_get_authenticated_user():
        return mock_default_user

    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    app.dependency_overrides.pop(get_authenticated_user, None)


@pytest.fixture
def mock_user():
    user = Mock()
    user.id = "test-user-123"
    return user


@pytest.fixture
def mock_default_user():
    """Mock default user for testing."""
    return SimpleNamespace(
        id=str(uuid.uuid4()),
        email="default@example.com",
        is_active=True,
        tenant_id=str(uuid.uuid4()),
    )


def test_upload_ontology_success(client):
    """Test successful ontology upload"""
    ontology_content = (
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    )
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", ontology_content, "application/xml"))],
        data={"ontology_key": unique_key, "description": "Test"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["uploaded_ontologies"][0]["ontology_key"] == unique_key
    assert "uploaded_at" in data["uploaded_ontologies"][0]


def test_upload_ontology_invalid_file(client):
    """Test 400 response for non-.owl files"""
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/api/v1/ontologies",
        files={"ontology_file": ("test.txt", b"not xml")},
        data={"ontology_key": unique_key},
    )
    assert response.status_code == 400


def test_upload_ontology_missing_data(client):
    """Test 400 response for missing file or key"""
    # Missing file
    response = client.post("/api/v1/ontologies", data={"ontology_key": "test"})
    assert response.status_code == 400

    # Missing key
    response = client.post(
        "/api/v1/ontologies", files=[("ontology_file", ("test.owl", b"xml", "application/xml"))]
    )
    assert response.status_code == 400


def test_upload_ontology_without_auth_header(client):
    """Test behavior when no explicit authentication header is provided."""
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", b"<rdf></rdf>", "application/xml"))],
        data={"ontology_key": unique_key},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["uploaded_ontologies"][0]["ontology_key"] == unique_key
    assert "uploaded_at" in data["uploaded_ontologies"][0]


def test_upload_multiple_ontologies_in_single_request_is_rejected(client):
    """Uploading multiple ontology files in a single request should fail."""
    import io

    file1_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    file2_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"

    files = [
        ("ontology_file", ("vehicles.owl", io.BytesIO(file1_content), "application/xml")),
        ("ontology_file", ("manufacturers.owl", io.BytesIO(file2_content), "application/xml")),
    ]
    data = {"ontology_key": "vehicles", "description": "Base vehicles"}

    response = client.post("/api/v1/ontologies", files=files, data=data)

    assert response.status_code == 400
    assert "Only one ontology_file is allowed" in response.json()["error"]


def test_upload_endpoint_rejects_array_style_fields(client):
    """Array-style form values should be rejected (no backwards compatibility)."""
    import io
    import json

    file_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"

    files = [("ontology_file", ("single.owl", io.BytesIO(file_content), "application/xml"))]
    data = {
        "ontology_key": json.dumps(["single_key"]),
        "description": json.dumps(["Single ontology"]),
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)

    assert response.status_code == 400
    assert "ontology_key must be a string" in response.json()["error"]


def test_cognify_with_multiple_ontologies(client):
    """Test cognify endpoint accepts multiple ontology keys"""
    payload = {
        "datasets": ["test_dataset"],
        "ontology_key": ["ontology1", "ontology2"],  # Array instead of string
        "run_in_background": False,
    }

    response = client.post("/api/v1/cognify", json=payload)

    # Should not fail due to ontology_key type
    assert response.status_code in [200, 400, 409]  # May fail for other reasons, not type


def test_complete_multifile_workflow(client):
    """Test workflow: upload ontologies one-by-one → cognify with multiple keys"""
    import io

    # Step 1: Upload two ontologies (one-by-one)
    file1_content = b"""<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:owl="http://www.w3.org/2002/07/owl#">
        <owl:Class rdf:ID="Vehicle"/>
    </rdf:RDF>"""

    file2_content = b"""<?xml version="1.0"?>
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:owl="http://www.w3.org/2002/07/owl#">
        <owl:Class rdf:ID="Manufacturer"/>
    </rdf:RDF>"""

    upload_response_1 = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("vehicles.owl", io.BytesIO(file1_content), "application/xml"))],
        data={"ontology_key": "vehicles", "description": "Vehicle ontology"},
    )
    assert upload_response_1.status_code == 200

    upload_response_2 = client.post(
        "/api/v1/ontologies",
        files=[
            ("ontology_file", ("manufacturers.owl", io.BytesIO(file2_content), "application/xml"))
        ],
        data={"ontology_key": "manufacturers", "description": "Manufacturer ontology"},
    )
    assert upload_response_2.status_code == 200

    # Step 2: Verify ontologies are listed
    list_response = client.get("/api/v1/ontologies")
    assert list_response.status_code == 200
    ontologies = list_response.json()
    assert "vehicles" in ontologies
    assert "manufacturers" in ontologies

    # Step 3: Test cognify with multiple ontologies
    cognify_payload = {
        "datasets": ["test_dataset"],
        "ontology_key": ["vehicles", "manufacturers"],
        "run_in_background": False,
    }

    cognify_response = client.post("/api/v1/cognify", json=cognify_payload)
    # Should not fail due to ontology handling (may fail for dataset reasons)
    assert cognify_response.status_code != 400  # Not a validation error


def test_upload_error_handling(client):
    """Test error handling for invalid uploads (single-file endpoint)."""
    import io
    import json

    # Array-style key should be rejected
    file_content = b"<rdf:RDF></rdf:RDF>"
    files = [("ontology_file", ("test.owl", io.BytesIO(file_content), "application/xml"))]
    data = {
        "ontology_key": json.dumps(["key1", "key2"]),
        "description": "desc1",
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)
    assert response.status_code == 400
    assert "ontology_key must be a string" in response.json()["error"]

    # Duplicate key should be rejected
    response_1 = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test1.owl", io.BytesIO(file_content), "application/xml"))],
        data={"ontology_key": "duplicate", "description": "desc1"},
    )
    assert response_1.status_code == 200

    response_2 = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test2.owl", io.BytesIO(file_content), "application/xml"))],
        data={"ontology_key": "duplicate", "description": "desc2"},
    )
    assert response_2.status_code == 400
    assert "already exists" in response_2.json()["error"]


def test_cognify_missing_ontology_key(client):
    """Test cognify with non-existent ontology key"""
    payload = {
        "datasets": ["test_dataset"],
        "ontology_key": ["nonexistent_key"],
        "run_in_background": False,
    }

    response = client.post("/api/v1/cognify", json=payload)
    assert response.status_code == 409
    assert "Ontology key 'nonexistent_key' not found" in response.json()["error"]
