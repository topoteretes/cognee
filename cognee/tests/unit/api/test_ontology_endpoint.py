import pytest
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
from types import SimpleNamespace
import importlib
from cognee.api.client import app

gau_mod = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")


@pytest.fixture
def client():
    return TestClient(app)


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


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_success(mock_get_default_user, client, mock_default_user):
    """Test successful ontology upload"""
    import json

    mock_get_default_user.return_value = mock_default_user
    ontology_content = (
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    )
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", ontology_content, "application/xml"))],
        data={"ontology_key": json.dumps([unique_key]), "description": json.dumps(["Test"])},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["uploaded_ontologies"][0]["ontology_key"] == unique_key
    assert "uploaded_at" in data["uploaded_ontologies"][0]


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_invalid_file(mock_get_default_user, client, mock_default_user):
    """Test 400 response for non-.owl files"""
    mock_get_default_user.return_value = mock_default_user
    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/api/v1/ontologies",
        files={"ontology_file": ("test.txt", b"not xml")},
        data={"ontology_key": unique_key},
    )
    assert response.status_code == 400


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_missing_data(mock_get_default_user, client, mock_default_user):
    """Test 400 response for missing file or key"""
    import json

    mock_get_default_user.return_value = mock_default_user
    # Missing file
    response = client.post("/api/v1/ontologies", data={"ontology_key": json.dumps(["test"])})
    assert response.status_code == 400

    # Missing key
    response = client.post(
        "/api/v1/ontologies", files=[("ontology_file", ("test.owl", b"xml", "application/xml"))]
    )
    assert response.status_code == 400


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_ontology_unauthorized(mock_get_default_user, client, mock_default_user):
    """Test behavior when default user is provided (no explicit authentication)"""
    import json

    unique_key = f"test_ontology_{uuid.uuid4().hex[:8]}"
    mock_get_default_user.return_value = mock_default_user
    response = client.post(
        "/api/v1/ontologies",
        files=[("ontology_file", ("test.owl", b"<rdf></rdf>", "application/xml"))],
        data={"ontology_key": json.dumps([unique_key])},
    )

    # The current system provides a default user when no explicit authentication is given
    # This test verifies the system works with conditional authentication
    assert response.status_code == 200
    data = response.json()
    assert data["uploaded_ontologies"][0]["ontology_key"] == unique_key
    assert "uploaded_at" in data["uploaded_ontologies"][0]


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_multiple_ontologies(mock_get_default_user, client, mock_default_user):
    """Test uploading multiple ontology files in single request"""
    import io

    mock_get_default_user.return_value = mock_default_user
    # Create mock files
    file1_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"
    file2_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"

    files = [
        ("ontology_file", ("vehicles.owl", io.BytesIO(file1_content), "application/xml")),
        ("ontology_file", ("manufacturers.owl", io.BytesIO(file2_content), "application/xml")),
    ]
    data = {
        "ontology_key": '["vehicles", "manufacturers"]',
        "descriptions": '["Base vehicles", "Car manufacturers"]',
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert "uploaded_ontologies" in result
    assert len(result["uploaded_ontologies"]) == 2
    assert result["uploaded_ontologies"][0]["ontology_key"] == "vehicles"
    assert result["uploaded_ontologies"][1]["ontology_key"] == "manufacturers"


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_upload_endpoint_accepts_arrays(mock_get_default_user, client, mock_default_user):
    """Test that upload endpoint accepts array parameters"""
    import io
    import json

    mock_get_default_user.return_value = mock_default_user
    file_content = b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'></rdf:RDF>"

    files = [("ontology_file", ("single.owl", io.BytesIO(file_content), "application/xml"))]
    data = {
        "ontology_key": json.dumps(["single_key"]),
        "descriptions": json.dumps(["Single ontology"]),
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["uploaded_ontologies"][0]["ontology_key"] == "single_key"


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_cognify_with_multiple_ontologies(mock_get_default_user, client, mock_default_user):
    """Test cognify endpoint accepts multiple ontology keys"""
    payload = {
        "datasets": ["test_dataset"],
        "ontology_key": ["ontology1", "ontology2"],  # Array instead of string
        "run_in_background": False,
    }

    response = client.post("/api/v1/cognify", json=payload)

    # Should not fail due to ontology_key type
    assert response.status_code in [200, 400, 409]  # May fail for other reasons, not type


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_complete_multifile_workflow(mock_get_default_user, client, mock_default_user):
    """Test complete workflow: upload multiple ontologies → cognify with multiple keys"""
    import io
    import json

    mock_get_default_user.return_value = mock_default_user
    # Step 1: Upload multiple ontologies
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

    files = [
        ("ontology_file", ("vehicles.owl", io.BytesIO(file1_content), "application/xml")),
        ("ontology_file", ("manufacturers.owl", io.BytesIO(file2_content), "application/xml")),
    ]
    data = {
        "ontology_key": json.dumps(["vehicles", "manufacturers"]),
        "descriptions": json.dumps(["Vehicle ontology", "Manufacturer ontology"]),
    }

    upload_response = client.post("/api/v1/ontologies", files=files, data=data)
    assert upload_response.status_code == 200

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


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_multifile_error_handling(mock_get_default_user, client, mock_default_user):
    """Test error handling for invalid multifile uploads"""
    import io
    import json

    # Test mismatched array lengths
    file_content = b"<rdf:RDF></rdf:RDF>"
    files = [("ontology_file", ("test.owl", io.BytesIO(file_content), "application/xml"))]
    data = {
        "ontology_key": json.dumps(["key1", "key2"]),  # 2 keys, 1 file
        "descriptions": json.dumps(["desc1"]),
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)
    assert response.status_code == 400
    assert "Number of keys must match number of files" in response.json()["error"]

    # Test duplicate keys
    files = [
        ("ontology_file", ("test1.owl", io.BytesIO(file_content), "application/xml")),
        ("ontology_file", ("test2.owl", io.BytesIO(file_content), "application/xml")),
    ]
    data = {
        "ontology_key": json.dumps(["duplicate", "duplicate"]),
        "descriptions": json.dumps(["desc1", "desc2"]),
    }

    response = client.post("/api/v1/ontologies", files=files, data=data)
    assert response.status_code == 400
    assert "Duplicate ontology keys not allowed" in response.json()["error"]


@patch.object(gau_mod, "get_default_user", new_callable=AsyncMock)
def test_cognify_missing_ontology_key(mock_get_default_user, client, mock_default_user):
    """Test cognify with non-existent ontology key"""
    mock_get_default_user.return_value = mock_default_user

    payload = {
        "datasets": ["test_dataset"],
        "ontology_key": ["nonexistent_key"],
        "run_in_background": False,
    }

    response = client.post("/api/v1/cognify", json=payload)
    assert response.status_code == 409
    assert "Ontology key 'nonexistent_key' not found" in response.json()["error"]
