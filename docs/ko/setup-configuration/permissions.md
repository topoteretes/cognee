# Permissions Setup

> Configure Cognee's permission system and access control

Enable Cognee's permission system for data isolation and access control. For detailed concepts, see [Cognee Permissions System](/core-concepts/permissions-system/overview).

## Enable Permission System

Set the environment variable to enable access control:

```dotenv  theme={null}
ENABLE_BACKEND_ACCESS_CONTROL=true
REQUIRE_AUTHENTICATION=true
```

<Warning>
  **Database Override**: Permission mode enforces Kùzu (graph) and LanceDB (vector). Custom providers are ignored.
</Warning>

## Database Setup

Choose your relational database:

* **SQLite** — Local development (auto-creates files)
* **Postgres** — Production (requires manual setup)

See [Relational Databases](./relational-databases) for detailed configuration.

## Authentication

### API Server

Start the server with authentication:

```bash  theme={null}
uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000
```

**Default credentials (development only):**

* Username: `default_user@example.com`
* Password: `default_password`

### Programmatic Access

See [Permission Snippets](/guides/permission-snippets) for complete programmatic examples.

## Data Organization

Data is automatically organized by user and dataset. Each user gets isolated storage:

```
.cognee_system/databases/<user_uuid>/
├── <dataset_uuid>.pkl         # Kùzu graph database
└── <dataset_uuid>.lance.db/   # LanceDB vector database
```

## Troubleshooting

**Permission Denied**: Verify user has required permission on the dataset.

**Data Isolation**: Check per-user database files exist:

```bash  theme={null}
ls -la .cognee_system/databases/<user_uuid>/
```

**Database Conflicts**: Custom providers are ignored in permission mode.

<Columns cols={2}>
  <Card title="Permission System" icon="brain" href="/core-concepts/permissions-system/overview">
    Learn about users, tenants, roles, and ACL
  </Card>

  <Card title="Usage Guide" icon="book-open" href="/guides/permission-snippets">
    How to use permission features
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt