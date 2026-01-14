# Datasets

> The core unit of data in Cognee's permission system

# Datasets: The Core Unit of Data

A dataset is a logical container for related documents and their processed knowledge graphs. All data in Cognee belongs to a dataset. When you add documents to Cognee using `cognee.add()`, they are processed and stored within a specific dataset.

<Info>**Dataset-scoped permissions** — All permissions in Cognee are defined at the dataset level, never for individual documents.</Info>

## Ownership and Permissions

When a principal creates a dataset, they become its **owner**. A principal is any entity that can have permissions, like a [user](./users), [tenant](./tenants), or [role](./roles). Ownership cannot be changed. The owner has full control and can grant permissions to others.

There are four types of permissions you can grant on a dataset:

* **Read** — View documents and query the knowledge graph.
* **Write** — Add, modify, or remove documents and data.
* **Delete** — Remove the entire dataset.
* **Share** — Grant permissions to other principals.

## Dataset Isolation: How Access Is Enforced

Cognee can enforce strict data isolation between datasets, but it's important to understand when this happens.

* **Isolation is optional**: Dataset boundaries are only enforced when the `ENABLE_BACKEND_ACCESS_CONTROL` setting is `true`.
* **Without isolation**: If this setting is `false`, dataset parameters are ignored during searches, and queries will run across all data in the system, regardless of permissions.
* **Database support**: True isolation is currently supported when using the following database backends (others do not support dataset isolation.):
  * **Relational Databases**: SQLite, Postgres
  * **Vector Databases**: LanceDB, Qdrant
  * **Graph Databases**: Kùzu, Neo4j
  * **Hybrid Databases**: FalkorDB

See [ACL](./acl) for details on how permissions are stored and checked. For setup instructions, see [Permissions Setup](/setup-configuration/permissions).

## Using Datasets in Operations

Datasets integrate with Cognee's main operations:

* **`add`**: Direct new content into a specific dataset by name or ID. If no dataset is specified, a default `main_dataset` is used.
* **`cognify`**: Choose which dataset(s) to transform into AI memory stored in the graph and vector stores.
* **`search`**: Scope queries to run only against datasets you have read access to.
* **`memify`**: Apply optional semantic enrichment on a per-dataset basis.

## Technical Details

<Accordion title="Operation Permission Requirements">
  Different operations require different permissions:

  * `add`/`cognify` operations → require `write` permission
  * `search` operations → require `read` permission
  * `delete` operations → require `delete` permission
  * Permission management → requires `share` permission
</Accordion>

<Accordion title="Dataset Creation Methods">
  Cognee provides two helper methods for creating datasets:

  * `create_dataset()`: This is a lower-level function that only inserts the dataset record. It expects the caller to manage the Access Control List (ACL) entries separately.
  * `create_authorized_dataset()`: This is the recommended method for most user-facing flows. It wraps `create_dataset()` and then immediately grants the creator full `read/write/delete/share` permissions. This ensures the dataset is usable as soon as it's created, especially when `ENABLE_BACKEND_ACCESS_CONTROL` is active.
</Accordion>

<Accordion title="Dataset Model Fields">
  The core dataset metadata is stored in a relational (SQL) database. The `datasets` table includes:

  * `id`: Unique identifier (UUID primary key)
  * `name`: Human-readable name
  * `owner_id`: ID of the principal who created the dataset
  * `created_at`: Timestamp when created
  * `updated_at`: Timestamp when last modified
</Accordion>

## Limitations

* Dataset ownership cannot be transferred.
* When access control is enabled, the graph and vector stores are enforced as Kùzu and LanceDB.
* Cross-dataset searches are not supported directly. Queries are always scoped to a single dataset. To search multiple datasets, you must run separate queries for each one you have access to.

<Columns cols={2}>
  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    See how datasets work with Add, Cognify, and Search
  </Card>

  <Card title="Building Blocks" icon="puzzle" href="/core-concepts/building-blocks/datapoints">
    Learn about the DataPoints that populate datasets
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt