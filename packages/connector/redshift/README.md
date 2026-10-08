# Amazon Redshift Community Connector for Cognee

An open-source data-source connector for ingesting data from **Amazon Redshift** into **Cognee** semantic memory graphs.

This connector uses the **Amazon Redshift Data API** (`boto3`), enabling seamless data extraction over standard AWS HTTPS REST endpoints without requiring direct database line-of-sight (port 5439) to Redshift clusters deployed inside private VPCs.

---

## Features

- **Redshift Data API Transport**: Works over HTTPS via AWS SDK (`boto3`), bypassing VPC network isolation restrictions.
- **Serverless & Provisioned Support**: Compatible with both Amazon Redshift Serverless workgroups and Provisioned clusters.
- **Flexible IAM & Credential Auth**: Native AWS IAM credential chain, AWS Secrets Manager ARNs, or temporary DB user credentials.
- **Safe Incremental Synchronization**: Cursor tracking using composite `(timestamp_column, primary_key)` logic with tie-breaking for equal timestamps.
- **Safe Upstream Deletion Reconciliation**: Supports soft-delete columns (`is_deleted`) and optional primary-key inventory reconciliation during full syncs. Tombstones are emitted **only** after complete, successful source reads.
- **Standalone Table Metadata**: Generates a dedicated table schema/comment document (`meta:<schema>.<table_name>`) alongside clean row documents.

---

## Installation

```bash
pip install cognee-community-connector-redshift
```

Or when developing inside the Cognee workspace:

```bash
pip install -e packages/connector/redshift
```

---

## Required AWS IAM Permissions

The IAM user or role executing the connector requires the following Redshift Data API permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "redshift-data:ExecuteStatement",
        "redshift-data:DescribeStatement",
        "redshift-data:GetStatementResult",
        "redshift-data:ListSchemas",
        "redshift-data:ListTables"
      ],
      "Resource": "*"
    }
  ]
}
```

If using AWS Secrets Manager to store database credentials:

```json
{
  "Effect": "Allow",
  "Action": [
    "secretsmanager:GetSecretValue"
  ],
  "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-redshift-secret-*"
}
```

---

## Environment Variables

| Variable | Description | Example |
| :--- | :--- | :--- |
| `AWS_REGION` | AWS Region where Redshift is located | `us-east-1` |
| `REDSHIFT_DATABASE` | Database name | `dev` |
| `REDSHIFT_WORKGROUP_NAME` | Workgroup name (for Redshift Serverless) | `my-workgroup` |
| `REDSHIFT_CLUSTER_ID` | Cluster Identifier (for Provisioned Redshift) | `my-cluster` |
| `REDSHIFT_SECRET_ARN` | (Optional) AWS Secrets Manager Secret ARN | `arn:aws:secretsmanager:...` |
| `REDSHIFT_DB_USER` | (Optional) Redshift database user name | `awsuser` |

---

## Usage Example

```python
import asyncio
import os
import cognee
from cognee_community_connector_redshift import RedshiftClient, redshift_source

async def main():
    # 1. Initialize Redshift Data API Client
    client = RedshiftClient(
        database=os.getenv("REDSHIFT_DATABASE", "dev"),
        workgroup_name=os.getenv("REDSHIFT_WORKGROUP_NAME"),
        cluster_identifier=os.getenv("REDSHIFT_CLUSTER_ID"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        timeout_seconds=300,
    )

    # 2. Construct the Redshift Table Source
    source = redshift_source(
        client=client,
        table_name="users",
        primary_key="id",
        schema="public",
        timestamp_column="updated_at",
        soft_delete_column="is_deleted",
        reconcile_deletions=True,
    )

    # 3. Ingest into Cognee Memory
    await cognee.remember(
        source,
        dataset_name="redshift_data",
        primary_key="id",
        write_disposition="merge",
    )

    # 4. Process into Semantic Knowledge Graph
    await cognee.cognify(dataset_name="redshift_data")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Incremental Synchronization & Deletion Invariants

1. **Incremental Cursor**:
   - Queries `WHERE <timestamp_column> >= :last_timestamp ORDER BY <timestamp_column> ASC, <primary_key> ASC`.
   - Filters out previously seen rows sharing the exact same timestamp via candidate primary-key tracking.
   - Resource state checkpoints are committed **only after** all result pages are completely fetched.

2. **Deletion Reconciliation**:
   - Missing primary keys are reconciled against `known_pks` stored in resource state.
   - **Safety Invariant**: If any API call or statement execution fails or times out, extraction aborts without generating deletion tombstones.

---

## Network Architecture & VPC Access

Redshift clusters inside private VPC subnets cannot be reached directly via TCP port 5439 unless direct peering, VPN, or SSH bastions are configured. By leveraging the **Redshift Data API**, queries execute asynchronously via AWS HTTPS endpoints (`redshift-data.<region>.amazonaws.com`), allowing standard environments to ingest data cleanly without complex tunnel management.
