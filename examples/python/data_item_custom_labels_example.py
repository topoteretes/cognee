"""
Example demonstrating the use of DataItem with custom labels.

This example shows how to provide custom labels for data items when adding them to Cognee.
Custom labels help with human-friendly identification of data, especially useful for text data
where the default name is just the content hash.
"""

import asyncio
import os
import pathlib
import cognee
from cognee import DataItem
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_datasets, get_dataset_data


async def main():
    # Setup Cognee directories
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/data_item_example")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/data_item_example")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    # Cleanup previous data
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    user = await get_default_user()

    # Example 1: Add a single text item with a custom label
    print("\n--- Example 1: Single DataItem with Custom Label ---")
    text_data = "This is important company information about Q3 financial results."
    data_item_1 = DataItem(data=text_data, label="Q3 Financial Report")

    await cognee.add(
        data=data_item_1,
        dataset_name="reports_dataset",
        user=user,
    )

    # Example 2: Add multiple text items with different labels
    print("\n--- Example 2: Multiple DataItems with Different Labels ---")
    items = [
        DataItem(data="Revenue increased by 25% compared to Q2.", label="Revenue Growth Analysis"),
        DataItem(
            data="Customer acquisition cost decreased by 15%.", label="CAC Optimization Report"
        ),
        DataItem(data="Product roadmap includes AI-powered features.", label="Product Strategy Q4"),
    ]

    await cognee.add(
        data=items,
        dataset_name="reports_dataset",
        user=user,
    )

    # Example 3: Mix DataItem with labels and plain text (no labels)
    print("\n--- Example 3: Mixed Data (with and without labels) ---")
    mixed_data = [
        DataItem(
            data="Customer satisfaction score is 4.7/5.0", label="Customer Satisfaction Metrics"
        ),
        "Churn rate is down to 2% from 3% last quarter.",  # No label
        DataItem(data="Employee retention improved significantly.", label="HR Performance Review"),
    ]

    await cognee.add(
        data=mixed_data,
        dataset_name="reports_dataset",
        user=user,
    )

    # Example 4: Display data with labels
    print("\n--- Example 4: Displaying Data with Labels ---")
    datasets = await get_datasets(user.id)
    reports_dataset = next((d for d in datasets if d.name == "reports_dataset"), None)

    if reports_dataset:
        dataset_data = await get_dataset_data(reports_dataset.id)
        print(f"\nDataset: {reports_dataset.name}")
        print(f"Total items: {len(dataset_data)}\n")

        for data_item in dataset_data:
            label_info = f"Label: {data_item.label}" if data_item.label else "Label: (None)"
            print(f"  ID: {str(data_item.id)[:8]}...")
            print(f"  Name: {data_item.name}")
            print(f"  {label_info}")
            print()

    # Cleanup
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("✓ DataItem example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
