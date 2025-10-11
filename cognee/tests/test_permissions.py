import os
import cognee
import pathlib

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user, create_user
from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
from cognee.modules.data.methods import get_dataset_data

logger = get_logger()


async def main():
    # Enable permissions feature
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "True"

    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_permissions")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_permissions")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    explanation_file_path_nlp = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )

    # Add document for default user
    await cognee.add([explanation_file_path_nlp], dataset_name="NLP")
    default_user = await get_default_user()

    explanation_file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    # Add document for test user
    test_user = await create_user("user@example.com", "example")
    await cognee.add([explanation_file_path_quantum], dataset_name="QUANTUM", user=test_user)

    nlp_cognify_result = await cognee.cognify(["NLP"], user=default_user)
    quantum_cognify_result = await cognee.cognify(["QUANTUM"], user=test_user)

    # Extract dataset_ids from cognify results
    def extract_dataset_id_from_cognify(cognify_result):
        """Extract dataset_id from cognify output dictionary"""
        for dataset_id, pipeline_result in cognify_result.items():
            return dataset_id  # Return the first (and likely only) dataset_id
        return None

    # Get dataset IDs from cognify results
    default_user_dataset_id = extract_dataset_id_from_cognify(nlp_cognify_result)
    print("User is", default_user_dataset_id)
    test_user_dataset_id = extract_dataset_id_from_cognify(quantum_cognify_result)

    # Check if default_user can only see information from the NLP dataset
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
    )
    assert len(search_results) == 1, "The search results list lenght is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")
    assert search_results[0]["dataset_name"] == "NLP", (
        f"Dict must contain dataset name 'NLP': {search_results[0]}"
    )

    # Check if test_user can only see information from the QUANTUM dataset
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=test_user,
    )
    assert len(search_results) == 1, "The search results list lenght is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")
    assert search_results[0]["dataset_name"] == "QUANTUM", (
        f"Dict must contain dataset name 'QUANTUM': {search_results[0]}"
    )

    # Try to add document with default_user to test_users dataset (test write permission enforcement)
    add_error = False
    try:
        await cognee.add(
            [explanation_file_path_nlp],
            dataset_name="QUANTUM",
            dataset_id=test_user_dataset_id,
            user=default_user,
        )
    except PermissionDeniedError:
        add_error = True
    assert add_error, "PermissionDeniedError was not raised during add as expected"

    # Try to cognify with default_user the test_users dataset (test write permission enforcement)
    cognify_error = False
    try:
        await cognee.cognify(datasets=[test_user_dataset_id], user=default_user)
    except PermissionDeniedError:
        cognify_error = True
    assert cognify_error, "PermissionDeniedError was not raised during cognify as expected"

    # Try to add permission for a dataset default_user does not have share permission for
    give_permission_error = False
    try:
        await authorized_give_permission_on_datasets(
            default_user.id,
            [test_user_dataset_id],
            "write",
            default_user.id,
        )
    except PermissionDeniedError:
        give_permission_error = True
    assert give_permission_error, (
        "PermissionDeniedError was not raised during assignment of permission as expected"
    )

    # Actually give permission to default_user to write on test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "write",
        test_user.id,
    )

    # Add new data to test_users dataset from default_user
    await cognee.add(
        [explanation_file_path_nlp],
        dataset_name="QUANTUM",
        dataset_id=test_user_dataset_id,
        user=default_user,
    )
    await cognee.cognify(datasets=[test_user_dataset_id], user=default_user)

    # Actually give permission to default_user to read on test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "read",
        test_user.id,
    )

    # Check if default_user can see from test_users datasets now
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
        dataset_ids=[test_user_dataset_id],
    )
    assert len(search_results) == 1, "The search results list length is not one."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    assert search_results[0]["dataset_name"] == "QUANTUM", (
        f"Dict must contain dataset name 'QUANTUM': {search_results[0]}"
    )

    # Check if default_user can only see information from both datasets now
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What is in the document?",
        user=default_user,
    )
    assert len(search_results) == 2, "The search results list length is not two."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    # Try deleting data from test_user dataset with default_user without delete permission
    delete_error = False
    try:
        # Get the dataset data to find the ID of the first data item (text)
        test_user_dataset_data = await get_dataset_data(test_user_dataset_id)
        text_data_id = test_user_dataset_data[0].id

        await cognee.delete(
            data_id=text_data_id, dataset_id=test_user_dataset_id, user=default_user
        )
    except PermissionDeniedError:
        delete_error = True

    assert delete_error, "PermissionDeniedError was not raised during delete operation as expected"

    # Try deleting data from test_user dataset with test_user
    # Get the dataset data to find the ID of the first data item (text)
    test_user_dataset_data = await get_dataset_data(test_user_dataset_id)
    text_data_id = test_user_dataset_data[0].id

    await cognee.delete(data_id=text_data_id, dataset_id=test_user_dataset_id, user=test_user)

    # Actually give permission to default_user to delete data for test_users dataset
    await authorized_give_permission_on_datasets(
        default_user.id,
        [test_user_dataset_id],
        "delete",
        test_user.id,
    )

    # Try deleting data from test_user dataset with default_user after getting delete permission
    # Get the dataset data to find the ID of the remaining data item (explanation_file_path_nlp)
    test_user_dataset_data = await get_dataset_data(test_user_dataset_id)
    explanation_file_data_id = test_user_dataset_data[0].id

    await cognee.delete(
        data_id=explanation_file_data_id, dataset_id=test_user_dataset_id, user=default_user
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
