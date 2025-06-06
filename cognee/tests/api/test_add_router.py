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
def test_add_text_content_via_form(mock_router_cognee_add, mock_get_auth_user, mock_user_fixture): # Renamed function
    mock_get_auth_user.return_value = mock_user_fixture
    client = TestClient(app)

    test_dataset_name = "my_text_form_dataset" # Changed dataset name for clarity
    test_content = "This is a test text content via form."

    response = client.post(
        "/api/v1/add/", # URL updated
        data={"datasetName": test_dataset_name, "text_content": test_content} # Sending as form data
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
        "/api/v1/add/", # URL updated
        data={"datasetName": test_dataset_name}, # datasetName as form data
        files={"data": (file_name, file_content, "text/plain")}
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
