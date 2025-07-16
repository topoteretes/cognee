import cognee
import asyncio
from uuid import UUID
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name


async def delete_by_cognify_output():
    """
    Extract dataset_id from cognify output and delete data from those datasets
    """

    # Example cognify outputs as shown in the test
    tst = {UUID("3e4a2926-d665-5865-9fc6-df8628329f4e"): "PipelineRunCompleted(...)"}
    tst2 = {UUID("3a1ca899-cd9e-50b1-b29e-66c39d12bb17"): "PipelineRunCompleted(...)"}

    # Extract dataset_ids from cognify results
    def extract_dataset_ids(cognify_result):
        """Extract dataset_ids from cognify output dictionary"""
        dataset_ids = []
        for dataset_id, pipeline_result in cognify_result.items():
            dataset_ids.append(dataset_id)
        return dataset_ids

    # Get dataset IDs from both results
    dataset_ids_1 = extract_dataset_ids(tst)
    dataset_ids_2 = extract_dataset_ids(tst2)

    print(f"Dataset IDs from tst: {dataset_ids_1}")
    print(f"Dataset IDs from tst2: {dataset_ids_2}")

    # Get default user for permissions
    default_user = await get_default_user()

    # Method 1: Delete specific data items within datasets
    print("\n🗑️ Method 1: Delete specific data items from datasets")

    for dataset_id in dataset_ids_1 + dataset_ids_2:
        try:
            # Get all data items in the dataset
            dataset_data = await get_dataset_data(dataset_id)

            print(f"\nDataset {dataset_id} contains {len(dataset_data)} data items:")

            # Delete each data item in the dataset
            for data_item in dataset_data:
                try:
                    result = await cognee.delete(
                        data_id=data_item.id, dataset_id=dataset_id, user=default_user
                    )
                    print(f"✅ Deleted data item {data_item.id}: {result}")
                except Exception as e:
                    print(f"❌ Failed to delete data item {data_item.id}: {e}")

        except Exception as e:
            print(f"❌ Failed to process dataset {dataset_id}: {e}")

    # Method 2: Delete entire datasets (if such functionality exists)
    print("\n🗑️ Method 2: Delete entire datasets")

    for dataset_id in dataset_ids_1 + dataset_ids_2:
        try:
            # This would be the equivalent of deleting the entire dataset
            # You might need to implement this based on your codebase
            result = await cognee.delete(dataset_id=dataset_id, user=default_user)
            print(f"✅ Deleted dataset {dataset_id}: {result}")
        except Exception as e:
            print(f"❌ Failed to delete dataset {dataset_id}: {e}")


async def delete_from_real_cognify_output(cognify_result):
    """
    Delete data from actual cognify output

    Args:
        cognify_result: Dictionary from cognify() call, format:
        {UUID('dataset_id'): PipelineRunCompleted(dataset_id=UUID('dataset_id'), ...)}
    """

    default_user = await get_default_user()

    for dataset_id, pipeline_result in cognify_result.items():
        print(f"\n🔄 Processing dataset: {dataset_id}")

        # Extract dataset_id from pipeline result if needed
        if hasattr(pipeline_result, "dataset_id"):
            actual_dataset_id = pipeline_result.dataset_id
        else:
            actual_dataset_id = dataset_id

        try:
            # Get all data in the dataset
            dataset_data = await get_dataset_data(actual_dataset_id)

            print(f"Found {len(dataset_data)} data items in dataset {actual_dataset_id}")

            # Delete each data item
            for data_item in dataset_data:
                try:
                    result = await cognee.delete(
                        data_id=data_item.id, dataset_id=actual_dataset_id, user=default_user
                    )
                    print(f"✅ Deleted data item {data_item.id}", str(result))
                except Exception as e:
                    print(f"❌ Failed to delete data item {data_item.id}: {e}")

        except Exception as e:
            print(f"❌ Failed to process dataset {actual_dataset_id}: {e}")


async def main():
    """Main function to demonstrate both approaches"""

    print("🧪 Testing deletion from cognify output")
    print("=" * 50)

    # Example usage with your actual cognify results
    # Replace these with your actual cognify results

    # If you have the actual cognify results, use them like this:
    # tst = await cognee.cognify(["tech_companies_1"])
    # tst2 = await cognee.cognify(["tech_companies_2"])
    #
    # await delete_from_real_cognify_output(tst)
    # await delete_from_real_cognify_output(tst2)

    # For demonstration with mock data
    await delete_by_cognify_output()


if __name__ == "__main__":
    asyncio.run(main())
