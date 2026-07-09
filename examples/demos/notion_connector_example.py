import asyncio
import os
import cognee
from cognee.tasks.ingestion.connectors import notion

async def main():
    # 1. Reset cognee state to start clean
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Reset complete.\n")

    # 2. Retrieve Notion API Key from environment
    api_key = os.getenv("NOTION_API_KEY")
    if not api_key:
        print("=" * 60)
        print("Warning: NOTION_API_KEY environment variable is not set.")
        print("To run this example with a live Notion account, please set it:")
        print("export NOTION_API_KEY='your-notion-integration-token'")
        print("=" * 60)
        print("Exiting example gracefully.")
        return

    print("Initializing Notion connector and ingesting workspace pages...")
    
    # 3. Trigger Cognee ingestion using the Notion connector
    # Behind the scenes, resolve_dlt_sources will detect the Notion DLT source,
    # run the ingestion pipeline, and map pages to DataItems.
    try:
        remember_result = await cognee.remember(
            notion(api_key=api_key),
            dataset_name="notion_workspace_dataset"
        )
        print("\nIngestion completed successfully!")
        print(remember_result)
    except Exception as e:
        print(f"Error during Notion ingestion: {e}")
        return

    # 4. Search within the ingested memory
    query = "What projects or tasks are mentioned in the Notion pages?"
    print(f"\nRecalling memory with query: '{query}'")
    
    search_results = await cognee.recall(query)
    print("\nSearch results:")
    for idx, result in enumerate(search_results):
        print(f"{idx + 1}. {result}")

if __name__ == "__main__":
    asyncio.run(main())
