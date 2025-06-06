import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from fastapi import UploadFile

from cognee.api.client import app # Main FastAPI app
from cognee.modules.users.models import User # User model

ADD_ROUTER_PATH = "cognee.api.v1.add.routers.get_add_router"

@pytest.fixture
def mock_user_fixture():
    return User(
        id="test_user_id",
        email="test@example.com",
        name="Test User",
        hashed_password="a_valid_password_hash_for_testing",
        is_active=True,
        is_superuser=False,
        is_verified=True,
        family_name="TestFamily",
        organization="TestOrg",
        phone_number="1234567890"
    )

@patch(f"{ADD_ROUTER_PATH}.get_authenticated_user")
@patch(f"{ADD_ROUTER_PATH}.cognee_add", new_callable=AsyncMock)
def test_add_text_content_via_json(mock_router_cognee_add, mock_get_auth_user, mock_user_fixture):
    mock_get_auth_user.return_value = mock_user_fixture
    client = TestClient(app)

    test_dataset_name = "my_json_dataset"
    test_content = "This is a test text content via JSON."

    response = client.post(
        f"/api/v1/add/?datasetName={test_dataset_name}",
        json={"content": test_content}
    )

    assert response.status_code == 200, response.text
    mock_router_cognee_add.assert_called_once_with(
        test_content,
        test_dataset_name,
        user=mock_user_fixture
    )

@patch(f"{ADD_ROUTER_PATH}.get_authenticated_user")
@patch(f"{ADD_ROUTER_PATH}.cognee_add", new_callable=AsyncMock)
def test_add_file_content(mock_router_cognee_add, mock_get_auth_user, mock_user_fixture):
    mock_get_auth_user.return_value = mock_user_fixture
    client = TestClient(app)

    test_dataset_name = "my_file_dataset"
    file_content = b"This is a test file."
    file_name = "test_file.txt"

    response = client.post(
        f"/api/v1/add/?datasetName={test_dataset_name}", # datasetName as query param
        files={"data": (file_name, file_content, "text/plain")} # Files part, 'data' is key for List[UploadFile]
    )
    assert response.status_code == 200, response.text

    assert mock_router_cognee_add.call_count == 1
    args, kwargs = mock_router_cognee_add.call_args

    assert isinstance(args[0], list)
    assert len(args[0]) == 1

    assert isinstance(args[0][0], UploadFile)
    assert args[0][0].filename == file_name

    assert args[1] == test_dataset_name
    assert kwargs["user"] == mock_user_fixture
