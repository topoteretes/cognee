# Relational Databases

> Configure relational databases for metadata and state storage in Cognee

Relational databases store metadata, document information, and system state in Cognee. They track documents, chunks, and provenance (where data came from and how it's linked).

<Info>
  **New to configuration?**

  See the [Setup Configuration Overview](./overview) for the complete workflow:

  install extras → create `.env` → choose providers → handle pruning.
</Info>

## Supported Providers

Cognee supports two relational database options:

* **SQLite** — File-based database, works out of the box (default)
* **Postgres** — Production-ready database for multi-process concurrency

## Configuration

<Accordion title="Environment Variables">
  Set these environment variables in your `.env` file:

  * `DB_PROVIDER` — The database provider (sqlite, postgres)
  * `DB_NAME` — Database name
  * `DB_HOST` — Database host (Postgres only)
  * `DB_PORT` — Database port (Postgres only)
  * `DB_USERNAME` — Database username (Postgres only)
  * `DB_PASSWORD` — Database password (Postgres only)
</Accordion>

## Setup Guides

<AccordionGroup>
  <Accordion title="SQLite (Default)">
    SQLite is file-based and requires no additional setup. It's perfect for local development and single-user scenarios.

    ```dotenv  theme={null}
    DB_PROVIDER="sqlite"
    DB_NAME="cognee_db"
    ```

    **Installation**: SQLite is included by default with Cognee. No additional installation required.

    **Data Location**: Data is stored under the Cognee system directory. You can override the root with `SYSTEM_ROOT_DIRECTORY` in your `.env` file.
  </Accordion>

  <Accordion title="Postgres">
    Postgres is recommended for production environments, multi-process concurrency, or when you need external hosting.

    ```dotenv  theme={null}
    DB_PROVIDER="postgres"
    DB_NAME="cognee_db"
    DB_HOST="127.0.0.1"            # use host.docker.internal when running inside Docker
    DB_PORT="5432"
    DB_USERNAME="cognee"
    DB_PASSWORD="cognee"
    ```

    **Installation**: Install the Postgres extras:

    ```bash  theme={null}
    pip install "cognee[postgres]"
    # or for binary version
    pip install "cognee[postgres-binary]"
    ```

    **Docker Setup**: Use the built-in Postgres service:

    ```bash  theme={null}
    docker compose --profile postgres up -d
    ```

    **Docker Networking**: When running Cognee in Docker and Postgres on your host, set:

    ```dotenv  theme={null}
    DB_HOST="host.docker.internal"
    ```
  </Accordion>
</AccordionGroup>

## Advanced Options

<Accordion title="Migration Configuration">
  Use migration settings to extract data from a relational database and load it into the graph store.

  ```dotenv  theme={null}
  MIGRATION_DB_PROVIDER="sqlite"   # or postgres
  MIGRATION_DB_PATH="/path/to/migration/directory"
  MIGRATION_DB_NAME="migration_database.sqlite"
  # For Postgres migrations
  # MIGRATION_DB_HOST=127.0.0.1
  # MIGRATION_DB_PORT=5432
  # MIGRATION_DB_USERNAME=cognee
  # MIGRATION_DB_PASSWORD=cognee
  ```
</Accordion>

<Accordion title="Backend Access Control">
  Enable per-user dataset isolation for multi-tenant scenarios.

  ```dotenv  theme={null}
  ENABLE_BACKEND_ACCESS_CONTROL="true"
  ```

  This feature is available for both SQLite and Postgres.
</Accordion>

## Troubleshooting

<Accordion title="Common Issues">
  **Postgres Connectivity**: Verify the database is listening on `DB_HOST:DB_PORT` and credentials are correct:

  ```bash  theme={null}
  psql -h 127.0.0.1 -U cognee -d cognee_db
  ```

  **Docker Networking**: Use `host.docker.internal` for host-to-container access on macOS/Windows.

  **SQLite Concurrency**: SQLite has limited write concurrency; prefer Postgres for heavy multi-user workloads.
</Accordion>

## When to Use Each

* **SQLite**: Local development, single-user applications, simple deployments
* **Postgres**: Production environments, multi-user applications, external hosting, co-location with pgvector

<Columns cols={3}>
  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Configure vector databases for embedding storage
  </Card>

  <Card title="Graph Stores" icon="network" href="/setup-configuration/graph-stores">
    Set up graph databases for knowledge graphs
  </Card>

  <Card title="Overview" icon="settings" href="/setup-configuration/overview">
    Return to setup configuration overview
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt