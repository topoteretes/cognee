import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, ANY
from fastapi import UploadFile
from types import SimpleNamespace # Added for mock dataset object

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


@patch(f"{ADD_ROUTER_PATH}.get_relational_engine")
@patch(f"{ADD_ROUTER_PATH}.create_dataset")
@patch(f"{ADD_ROUTER_PATH}.get_dataset")
@patch(f"{ADD_ROUTER_PATH}.cognee_add", new_callable=AsyncMock)
@patch(f"{ADD_ROUTER_PATH}.get_authenticated_user")
def test_add_text_to_new_dataset_auto_creates(
    mock_get_auth_user,
    mock_router_cognee_add,
    mock_router_get_dataset,
    mock_router_create_dataset,
    mock_router_get_engine,
    mock_user_fixture, # Fixture for authenticated user
):
    mock_get_auth_user.return_value = mock_user_fixture
    client = TestClient(app)

    new_dataset_name = "auto_created_dataset"
    test_content = "Some text for the new dataset"

    # Configure mock for get_dataset (router's import) to return None (dataset not found)
    mock_router_get_dataset.return_value = None

    # Configure mock for create_dataset (router's import)
    # It needs to return an object with a 'name' attribute
    mock_created_dataset_object = SimpleNamespace(name=new_dataset_name)
    mock_router_create_dataset.return_value = mock_created_dataset_object

    # Configure mock for get_relational_engine and its session
    mock_engine_instance = AsyncMock()
    mock_db_session = AsyncMock() # This will be the yielded session

    # Make get_async_session an async context manager
    # The __aenter__ should return the mock_db_session
    mock_engine_instance.get_async_session.return_value.__aenter__.return_value = mock_db_session
    mock_engine_instance.get_async_session.return_value.__aexit__.return_value = None # for context manager exit

    mock_router_get_engine.return_value = mock_engine_instance

    # Perform the request
    response = client.post(
        "/api/v1/add/",
        data={
            "datasetName": new_dataset_name,
            "text_content": test_content,
        }
    )

    assert response.status_code == 200, response.text

    # Assertions
    # 1. get_dataset was called to check if dataset exists
    mock_router_get_dataset.assert_called_once_with(
        user_id=mock_user_fixture.id,
        dataset_name=new_dataset_name
    )

    # 2. get_relational_engine was called to get the engine for creating a session
    mock_router_get_engine.assert_called_once()

    # 3. create_dataset was called because dataset was not found
    mock_router_create_dataset.assert_called_once_with(
        dataset_name=new_dataset_name,
        user=mock_user_fixture,
        session=mock_db_session # Ensure it's called with the session from the context manager
    )

    # 4. cognee_add was called with the content and the (newly created) dataset's name
    mock_router_cognee_add.assert_called_once_with(
        test_content,
        new_dataset_name, # This should be the name of the dataset resolved (mocked as new_dataset_name)
        user=mock_user_fixture
    )
