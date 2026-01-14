# Overview

> Introduction to Cognee's permission system and access control architecture

# Cognee Permissions System

The Cognee permission system manages access to data through an access control architecture. This system provides data isolation and access control through dataset-scoped permissions and per-dataset storage, enabling multiple users or organizations to use the same Cognee instance while keeping their data completely separate.

<Info>**Enable Backend Access Control (EBAC)** is the configuration flag that activates this multi-tenant mode, enforcing user authentication and complete data isolation.</Info>

## Core Components

The permission system is built around several key concepts:

* **Dataset** — The basic unit of data in Cognee. All documents and their processed knowledge graphs belong to a dataset. Permissions are always defined at the dataset level. See [Datasets](./datasets) for details.

* **Principal** — Any entity that can hold permissions. Principals come in three forms: [Users](./users), [Tenants](./tenants), and [Roles](./roles). This unified design supports flexible access control across individuals and organizations.

* **User** — An individual who creates and interacts with datasets. Users can own datasets and be granted permissions on others. Each user belongs to at most one tenant.

* **Tenant** — An organization or group. Tenants contain users and can be granted permissions on datasets, which apply to all members.

* **Role** — A group of users within a tenant. Roles can also be granted dataset permissions, which apply to their members.

* **ACL** — The Access Control List records all permission assignments. Each entry links a principal to a dataset with a specific permission type. See [ACL](./acl) for details.

## Permission Types

There are four types of permissions that can be granted on datasets:

* **Read** — View documents and query the knowledge graph
* **Write** — Add, modify, or remove documents and data
* **Delete** — Remove the entire dataset
* **Share** — Grant permissions to other principals

## How It Works

When `ENABLE_BACKEND_ACCESS_CONTROL` is set to true, Cognee runs in access control mode:

* **Authentication becomes mandatory** (even if `REQUIRE_AUTHENTICATION=false`)
* **Data isolation is enforced** at the user + dataset level for graph and vector stores
* **Database routing is automatic** — Kùzu (graph) and LanceDB (vector) are configured per request via context variables
* **Supported databases**: SQLite/Postgres (relational), LanceDB (vector), Kùzu (graph)
* **Custom providers are ignored** — EBAC enforces Kùzu and LanceDB regardless of user configuration

See [Setup Configuration](../../setup-configuration/permissions) for configuration details.

## Permission Resolution

When a user tries to access data, the system evaluates their effective permissions by combining:

1. **Direct user permissions** — explicitly granted to the user
2. **Role permissions** — inherited through the user's role memberships
3. **Tenant permissions** — inherited through the user's tenant membership

The system doesn't store "effective permissions" anywhere—it calculates them on demand by querying ACL entries. This approach ensures permissions are always current and allows for complex permission inheritance without data duplication.

## Data Storage Layout

When EBAC is enabled, Cognee automatically organizes data by user and dataset:

**Filesystem layout**:

```
.cognee_system/databases/<user_uuid>/
├── <dataset_uuid>.pkl         # Kùzu graph database
└── <dataset_uuid>.lance.db/   # LanceDB vector database

.data_storage/<tenant_uuid_or_user_uuid>/
└── ...                        # Raw and processed files
```

**Key points:**

* Each user gets their own database directory
* Each dataset gets its own database files within the user's directory
* File storage is organized by tenant (if user belongs to one) or by user ID
* This structure prevents any cross-user data access at the filesystem level

<Columns cols={2}>
  <Card title="Datasets" icon="database" href="/core-concepts/permissions-system/datasets">
    Learn about datasets as the core unit of data
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/permissions">
    Configure multi-tenant mode and access control
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt