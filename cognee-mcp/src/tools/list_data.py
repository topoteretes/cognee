"""Tool for listing datasets and their data items."""

import sys
from uuid import UUID
from contextlib import redirect_stdout
import mcp.types as types
from cognee.shared.logging_utils import get_logger

from src.shared import context

logger = get_logger()


async def list_data(dataset_id: str = None) -> list:
    """
    List all datasets and their data items with IDs for deletion operations.

    This function helps users identify data IDs and dataset IDs that can be used
    with the delete tool. It provides a comprehensive view of available data.

    Parameters
    ----------
    dataset_id : str, optional
        If provided, only list data items from this specific dataset.
        If None, lists all datasets and their data items.
        Should be a valid UUID string.

    Returns
    -------
    list
        A list containing a single TextContent object with formatted information
        about datasets and data items, including their IDs for deletion.

    Notes
    -----
    - Use this tool to identify data_id and dataset_id values for the delete tool
    - The output includes both dataset information and individual data items
    - UUIDs are displayed in a format ready for use with other tools
    """

    with redirect_stdout(sys.stderr):
        try:
            output_lines = []

            if dataset_id:
                # Detailed data listing for specific dataset is only available in direct mode
                if context.cognee_client.use_api:
                    return [
                        types.TextContent(
                            type="text",
                            text="❌ Detailed data listing for specific datasets is not available in API mode.\nPlease use the API directly or use direct mode.",
                        )
                    ]

                from cognee.modules.users.methods import get_default_user
                from cognee.modules.data.methods import get_dataset, get_dataset_data

                logger.info(f"Listing data for dataset: {dataset_id}")
                dataset_uuid = UUID(dataset_id)
                user = await get_default_user()

                dataset = await get_dataset(user.id, dataset_uuid)

                if not dataset:
                    return [
                        types.TextContent(type="text", text=f"❌ Dataset not found: {dataset_id}")
                    ]

                # Get data items in the dataset
                data_items = await get_dataset_data(dataset.id)

                output_lines.append(f"📁 Dataset: {dataset.name}")
                output_lines.append(f"   ID: {dataset.id}")
                output_lines.append(f"   Created: {dataset.created_at}")
                output_lines.append(f"   Data items: {len(data_items)}")
                output_lines.append("")

                if data_items:
                    for i, data_item in enumerate(data_items, 1):
                        output_lines.append(f"   📄 Data item #{i}:")
                        output_lines.append(f"      Data ID: {data_item.id}")
                        output_lines.append(f"      Name: {data_item.name or 'Unnamed'}")
                        output_lines.append(f"      Created: {data_item.created_at}")
                        output_lines.append("")
                else:
                    output_lines.append("   (No data items in this dataset)")

            else:
                # List all datasets - works in both modes
                logger.info("Listing all datasets")
                datasets = await context.cognee_client.list_datasets()

                if not datasets:
                    return [
                        types.TextContent(
                            type="text",
                            text="📂 No datasets found.\nUse the cognify tool to create your first dataset!",
                        )
                    ]

                output_lines.append("📂 Available Datasets:")
                output_lines.append("=" * 50)
                output_lines.append("")

                for i, dataset in enumerate(datasets, 1):
                    # In API mode, dataset is a dict; in direct mode, it's formatted as dict
                    if isinstance(dataset, dict):
                        output_lines.append(f"{i}. 📁 {dataset.get('name', 'Unnamed')}")
                        output_lines.append(f"   Dataset ID: {dataset.get('id')}")
                        output_lines.append(f"   Created: {dataset.get('created_at', 'N/A')}")
                    else:
                        output_lines.append(f"{i}. 📁 {dataset.name}")
                        output_lines.append(f"   Dataset ID: {dataset.id}")
                        output_lines.append(f"   Created: {dataset.created_at}")
                    output_lines.append("")

                if not context.cognee_client.use_api:
                    output_lines.append("💡 To see data items in a specific dataset, use:")
                    output_lines.append('   list_data(dataset_id="your-dataset-id-here")')
                    output_lines.append("")
                output_lines.append("🗑️  To delete specific data, use:")
                output_lines.append('   delete(data_id="data-id", dataset_id="dataset-id")')

            result_text = "\n".join(output_lines)
            logger.info("List data operation completed successfully")

            return [types.TextContent(type="text", text=result_text)]

        except ValueError as e:
            error_msg = f"❌ Invalid UUID format: {str(e)}"
            logger.error(error_msg)
            return [types.TextContent(type="text", text=error_msg)]

        except Exception as e:
            error_msg = f"❌ Failed to list data: {str(e)}"
            logger.error(f"List data error: {str(e)}")
            return [types.TextContent(type="text", text=error_msg)]
