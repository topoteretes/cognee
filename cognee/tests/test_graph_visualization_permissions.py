import asyncio
import os
import pathlib

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import cognee
from cognee.api.client import app
from cognee.modules.users.methods import create_user, get_default_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets

# Use pytest-asyncio to handle all async tests
pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for our test module."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def client():
    """Create an async HTTP client for testing"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture(scope="module")
async def setup_environment():
    """
    Set up a clean environment for the test, creating necessary users and datasets.
    This fixture runs once before all tests and cleans up afterwards.
    """
    # 1. Enable permissions feature
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    # 2. Set up an independent test directory
    base_dir = pathlib.Path(__file__).parent
    cognee.config.data_root_directory(str(base_dir / ".data_storage/test_graph_viz"))
    cognee.config.system_root_directory(str(base_dir / ".cognee_system/test_graph_viz"))

    # 3. Clean up old data
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # 4. Add document for default user
    explanation_file_path = os.path.join(base_dir, "test_data/Natural_language_processing.txt")
    await cognee.add([explanation_file_path], dataset_name="NLP")
    default_user = await get_default_user()
    nlp_cognify_result = await cognee.cognify(["NLP"], user=default_user)

    def extract_dataset_id_from_cognify(cognify_result):
        """Extract dataset_id from cognify output dictionary"""
        for dataset_id, pipeline_result in cognify_result.items():
            return dataset_id
        return None

    dataset_id = extract_dataset_id_from_cognify(nlp_cognify_result)

    yield dataset_id

    # 5. Clean up data after tests are finished
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def get_authentication_headers(client: AsyncClient, email: str, password: str) -> dict:
    """Authenticates and returns the Authorization header."""
    login_data = {"username": email, "password": password}
    response = await client.post("/api/v1/auth/login", data=login_data, timeout=15)

    assert response.status_code == 200, "Failed to log in and get token"

    token_data = response.json()
    access_token = token_data["access_token"]

    return {"Authorization": f"Bearer {access_token}"}


async def test_owner_can_access_graph(client: AsyncClient, setup_environment: int):
    """
    Test Case 1: The dataset owner should be able to access the graph data successfully.
    """
    dataset_id = setup_environment
    default_user_email = "default_user@example.com"
    default_user_password = "default_password"

    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/graph",
        headers=await get_authentication_headers(client, default_user_email, default_user_password),
    )
    assert response.status_code == 200, (
        f"Owner failed to get the knowledge graph visualization. Response: {response.json()}"
    )
    data = response.json()
    assert len(data) > 1, "The graph data is not valid."

    print("✅ Owner can access the graph visualization successfully.")


async def test_granting_permission_enables_access(client: AsyncClient, setup_environment: int):
    """
    Test Case 2: A user without any permissions should be denied access (404 Not Found).
    After granting permission, the user should be able to access the graph data.
    """
    dataset_id = setup_environment
    # Create a user without any permissions to the dataset
    test_user_email = "test_user@example.com"
    test_user_password = "test_password"
    test_user = await create_user(test_user_email, test_user_password)

    # Test the access to graph visualization for the test user without any permissions
    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/graph",
        headers=await get_authentication_headers(client, test_user_email, test_user_password),
    )
    assert response.status_code == 403, (
        "Access to graph visualization should be denied without READ permission."
    )
    assert (
        response.json()["detail"]
        == "Request owner does not have necessary permission: [read] for all datasets requested. [PermissionDeniedError]"
    )
    print("✅ Access to graph visualization should be denied without READ permission.")

    # Grant permission to the test user
    default_user = await get_default_user()
    await authorized_give_permission_on_datasets(
        test_user.id, [dataset_id], "read", default_user.id
    )

    # Test the access to graph visualization for the test user
    response_for_test_user = await client.get(
        f"/api/v1/datasets/{dataset_id}/graph",
        headers=await get_authentication_headers(client, test_user_email, test_user_password),
    )
    assert response_for_test_user.status_code == 200, (
        "Access to graph visualization should succeed for user with been granted read permission"
    )
    print(
        "✅ Access to graph visualization should succeed for user with been granted read permission"
    )

    # Test the graph data is the same for the test user and the default user
    default_user_email = "default_user@example.com"
    default_user_password = "default_password"
    response_for_default_user = await client.get(
        f"/api/v1/datasets/{dataset_id}/graph",
        headers=await get_authentication_headers(client, default_user_email, default_user_password),
    )
    assert response_for_test_user.json() == response_for_default_user.json(), (
        "The graph data for the test user and the default user is not the same."
    )
    print("✅ The graph data for the test user and the default user is the same.")
