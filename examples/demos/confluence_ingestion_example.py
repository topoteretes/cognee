import asyncio
import os
import cognee

async def main():
    # Load configuration from environment variables
    confluence_url = os.environ.get("CONFLUENCE_URL")
    email = os.environ.get("CONFLUENCE_EMAIL")
    api_token = os.environ.get("CONFLUENCE_API_TOKEN")

    if not all([confluence_url, email, api_token]):
        print("Please set CONFLUENCE_URL, CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN environment variables.")
        return

    # Cognee configuration
    cognee.config.system_root_directory(".")

    # We use cognee.remember() with a dict configuration.
    # This automatically invokes the Confluence DLT connector under the hood.
    print("Ingesting Confluence data...")
    await cognee.remember(
        {
            "confluence_url": confluence_url,
            "email": email,
            "api_token": api_token,
            # "space_keys": ["ENG"], # optionally filter by space key
        },
        dataset_name="confluence_wiki",
        primary_key="id",
        write_disposition="merge",
    )
    print("Ingestion complete!")

if __name__ == "__main__":
    asyncio.run(main())
