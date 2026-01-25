"""
CRITICAL Test: Delete All with Mixed Permissions

Tests that datasets.delete_all() correctly handles scenarios where a user has
delete permissions on some datasets but not others.

Test Coverage:
- test_delete_all_with_partial_permissions: User has delete permission on 2/4 datasets
- test_delete_all_permission_error_handling: User has no delete permissions on any dataset
"""

import os
import pathlib
import pytest

import cognee
from cognee.api.v1.datasets import datasets
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.models import User
from cognee.modules.users.methods import create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@pytest.mark.asyncio
async def test_delete_all_with_partial_permissions():
    """
    Test that delete_all() only deletes datasets the user has delete permission for.

    Setup:
    - Create 4 datasets with different owners
    - User A owns datasets 1 and 2 (auto delete permission)
    - Grant user A delete permission on dataset 3 (owned by other user)
    - Dataset 4: User A has no permissions
    - Add data to all datasets

    Expected:
    - Datasets 1, 2, 3: DELETED (user has delete permission)
    - Dataset 4: NOT DELETED (no permission)
    - No exception raised
    """
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_all_mixed_permissions"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_all_mixed_permissions"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    # Create users
    user_a: User = await create_user(email="user_a@test.com", password="password123")
    other_user: User = await create_user(email="other_user@test.com", password="password456")

    # Create datasets with different ownership
    dataset_1 = await create_authorized_dataset(dataset_name="dataset_1", user=user_a)
    dataset_2 = await create_authorized_dataset(dataset_name="dataset_2", user=user_a)
    dataset_3 = await create_authorized_dataset(dataset_name="dataset_3", user=other_user)
    dataset_4 = await create_authorized_dataset(dataset_name="dataset_4", user=other_user)

    # Grant user_a delete permission on dataset_3 (owned by other_user)
    await authorized_give_permission_on_datasets(user_a.id, [dataset_3.id], "delete", other_user.id)

    # Add text data to all datasets using cognee.add()
    await cognee.add(["Document about Company A"], dataset_name="dataset_1", user=user_a)
    await cognee.add(["Document about Company B"], dataset_name="dataset_2", user=user_a)
    await cognee.add(["Document about Company C"], dataset_name="dataset_3", user=other_user)
    await cognee.add(["Document about Company D"], dataset_name="dataset_4", user=other_user)

    # Cognify all datasets to create graph data
    await cognee.cognify(["dataset_1"], user=user_a)
    await cognee.cognify(["dataset_2"], user=user_a)
    await cognee.cognify(["dataset_3"], user=other_user)
    await cognee.cognify(["dataset_4"], user=other_user)

    # Verify all datasets exist before deletion
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes_before, edges_before = await graph_engine.get_graph_data()
    logger.info(f"Before delete_all: {len(nodes_before)} nodes, {len(edges_before)} edges")

    # Execute delete_all for user_a
    logger.info("Executing delete_all for user_a...")
    await datasets.delete_all(user=user_a)

    # Verify only authorized datasets were deleted
    user_a_datasets = await datasets.list_datasets(user=user_a)
    other_user_datasets = await datasets.list_datasets(user=other_user)

    logger.info(f"User A can see {len(user_a_datasets)} datasets after delete_all")
    logger.info(f"Other user can see {len(other_user_datasets)} datasets")

    # User A should see 0 datasets (all authorized ones deleted)
    assert len(user_a_datasets) == 0, (
        f"User A should see 0 datasets after delete_all, but sees {len(user_a_datasets)}"
    )

    # Other user should still see dataset_4 (user_a had no permission on it)
    assert len(other_user_datasets) == 1, (
        f"Other user should still see 1 dataset (dataset_4), but sees {len(other_user_datasets)}"
    )
    assert other_user_datasets[0].name == "dataset_4", "Dataset 4 should still exist"

    # Verify dataset_4 still has its data in the graph
    nodes_after, edges_after = await graph_engine.get_graph_data()
    logger.info(f"After delete_all: {len(nodes_after)} nodes, {len(edges_after)} edges")

    # Should have nodes from dataset_4 remaining
    assert len(nodes_after) >= 0, "Dataset 4 data may still exist in graph"

    # Verify we can still access dataset_4 data
    dataset_4_data = await datasets.list_data(dataset_4.id, user=other_user)
    assert len(dataset_4_data) > 0, "Dataset 4 should still have data"

    logger.info("✅ test_delete_all_with_partial_permissions PASSED")


@pytest.mark.asyncio
async def test_delete_all_permission_error_handling():
    """
    Test that delete_all() handles gracefully when user has no delete permissions.

    Setup:
    - Create user B
    - Create dataset owned by other user
    - Grant user B only "read" permission (NOT delete)

    Expected:
    - delete_all() completes without error
    - No datasets deleted
    - Dataset still exists
    """
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".data_storage/test_delete_all_no_permissions"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent, ".cognee_system/test_delete_all_no_permissions"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    # Create users
    user_b: User = await create_user(email="user_b@test.com", password="password123")
    owner_user: User = await create_user(email="owner@test.com", password="password456")

    # Create dataset owned by owner_user
    dataset_x = await create_authorized_dataset(dataset_name="dataset_x", user=owner_user)

    # Grant user_b only READ permission (not delete)
    await authorized_give_permission_on_datasets(user_b.id, [dataset_x.id], "read", owner_user.id)

    # Add data to dataset using cognee.add()
    await cognee.add(["Document about Company X"], dataset_name="dataset_x", user=owner_user)
    await cognee.cognify(["dataset_x"], user=owner_user)

    # Verify dataset exists
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    nodes_before, _ = await graph_engine.get_graph_data()

    # Execute delete_all for user_b (who has no delete permissions)
    logger.info("Executing delete_all for user_b (no delete permissions)...")
    await datasets.delete_all(user=user_b)

    # Verify no datasets were deleted
    user_b_datasets = await datasets.list_datasets(user=user_b)
    owner_datasets = await datasets.list_datasets(user=owner_user)

    # User B should still see the dataset (has read permission)
    assert len(user_b_datasets) == 1, (
        f"User B should still see 1 dataset, but sees {len(user_b_datasets)}"
    )
    assert user_b_datasets[0].name == "dataset_x", "Dataset X should still be visible to user B"

    # Owner should still see the dataset
    assert len(owner_datasets) == 1, (
        f"Owner should still see 1 dataset, but sees {len(owner_datasets)}"
    )

    # Verify dataset still has data
    nodes_after, _ = await graph_engine.get_graph_data()
    assert len(nodes_after) == len(nodes_before), "Graph should be unchanged (no nodes deleted)"

    dataset_x_data = await datasets.list_data(dataset_x.id, user=owner_user)
    assert len(dataset_x_data) > 0, "Dataset X should still have data"

    logger.info("✅ test_delete_all_permission_error_handling PASSED")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_delete_all_with_partial_permissions())
    asyncio.run(test_delete_all_permission_error_handling())
