"""Example: Ingesting an Amazon Redshift table into Cognee memory.

Demonstrates how to configure the Redshift community connector and ingest
table data into Cognee using standard AWS environment variables.

Required environment variables:
    AWS_REGION (e.g. "us-east-1")
    REDSHIFT_DATABASE (e.g. "dev")
    REDSHIFT_WORKGROUP_NAME (for Serverless) OR REDSHIFT_CLUSTER_ID (for Provisioned)
    REDSHIFT_SECRET_ARN (optional, for AWS Secrets Manager authentication)
"""

import asyncio
import os

from cognee_community_connector_redshift import RedshiftClient, redshift_source

import cognee


async def main():
    # 1. Read configuration from environment variables
    database = os.getenv("REDSHIFT_DATABASE", "dev")
    workgroup_name = os.getenv("REDSHIFT_WORKGROUP_NAME")
    cluster_identifier = os.getenv("REDSHIFT_CLUSTER_ID")
    secret_arn = os.getenv("REDSHIFT_SECRET_ARN")
    region_name = os.getenv("AWS_REGION", "us-east-1")

    if not workgroup_name and not cluster_identifier:
        print("Please set REDSHIFT_WORKGROUP_NAME or REDSHIFT_CLUSTER_ID in your environment.")
        return

    # 2. Initialize the Redshift Data API client
    client = RedshiftClient(
        database=database,
        workgroup_name=workgroup_name,
        cluster_identifier=cluster_identifier,
        secret_arn=secret_arn,
        region_name=region_name,
        timeout_seconds=300,
    )

    # 3. Create the DLT source for a specific table
    source = redshift_source(
        client=client,
        table_name="users",
        primary_key="id",
        schema="public",
        timestamp_column="updated_at",
        soft_delete_column="is_deleted",
        reconcile_deletions=True,
    )

    # 4. Ingest into Cognee memory dataset
    print("Ingesting public.users table from Amazon Redshift into Cognee...")
    await cognee.remember(
        source,
        dataset_name="redshift_data",
        primary_key="id",
        write_disposition="merge",
    )
    print("Ingestion complete. Cognifying memory...")
    await cognee.cognify(dataset_name="redshift_data")

    # 5. Query Cognee memory
    search_results = await cognee.recall("Find users ingested from Redshift", dataset_name="redshift_data")
    print("\nRecall results:")
    for result in search_results:
        print("-", result)


if __name__ == "__main__":
    asyncio.run(main())
