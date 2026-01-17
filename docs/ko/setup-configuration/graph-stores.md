# Graph Stores

> Configure graph databases for knowledge graph storage and relationship reasoning in Cognee

Graph stores capture entities and relationships in knowledge graphs. They enable Cognee to understand structure and navigate connections between concepts, providing powerful reasoning capabilities.

<Info>
  **New to configuration?**

  See the [Setup Configuration Overview](./overview) for the complete workflow:

  install extras → create `.env` → choose providers → handle pruning.
</Info>

## Supported Providers

Cognee supports multiple graph store options:

* **Kuzu** — Local file-based graph database (default)
* **Kuzu-remote** — Kuzu with HTTP API access
* **Neo4j** — Production-ready graph database
* **Neptune** — Amazon Neptune cloud graph database
* **Neptune Analytics** — Amazon Neptune Analytics hybrid solution

## Configuration

<Accordion title="Environment Variables">
  Set these environment variables in your `.env` file:

  * `GRAPH_DATABASE_PROVIDER` — The graph store provider (kuzu, kuzu-remote, neo4j, neptune, neptune\_analytics)
  * `GRAPH_DATABASE_URL` — Database URL or connection string
  * `GRAPH_DATABASE_USERNAME` — Database username (optional)
  * `GRAPH_DATABASE_PASSWORD` — Database password (optional)
  * `GRAPH_DATABASE_NAME` — Database name (optional)
</Accordion>

## Setup Guides

<AccordionGroup>
  <Accordion title="Kuzu (Default)">
    Kuzu is file-based and requires no network setup. It's perfect for local development and single-user scenarios.

    ```dotenv  theme={null}
    GRAPH_DATABASE_PROVIDER="kuzu"
    # Optional: override location
    # SYSTEM_ROOT_DIRECTORY=/absolute/path/.cognee_system
    # The graph file will default to <SYSTEM_ROOT_DIRECTORY>/databases/cognee_graph_kuzu
    ```

    **Installation**: Kuzu is included by default with Cognee. No additional installation required.

    **Data Location**: The graph is stored on disk. Path defaults under the Cognee system directory and is created automatically.

    <Warning>
      **Concurrency Limitation**: Kuzu uses file-based locking and is not suitable for concurrent use from different agents or processes. For multi-agent scenarios, use Neo4j instead.
    </Warning>
  </Accordion>

  <Accordion title="Kuzu (Remote API)">
    Use Kuzu with an HTTP API when you need remote access or want to run Kuzu as a service.

    ```dotenv  theme={null}
    GRAPH_DATABASE_PROVIDER="kuzu-remote"
    GRAPH_DATABASE_URL="http://localhost:8000"
    GRAPH_DATABASE_USERNAME="<optional>"
    GRAPH_DATABASE_PASSWORD="<optional>"
    ```

    **Installation**: Requires a running Kuzu service exposing an HTTP API.
  </Accordion>

  <Accordion title="Neo4j">
    Neo4j is recommended for production environments and multi-user scenarios.

    ```dotenv  theme={null}
    ENABLE_BACKEND_ACCESS_CONTROL="true"
    GRAPH_DATABASE_PROVIDER="neo4j"
    GRAPH_DATABASE_URL="bolt://localhost:7687"
    GRAPH_DATABASE_NAME="neo4j"
    GRAPH_DATABASE_USERNAME="neo4j"
    GRAPH_DATABASE_PASSWORD="pleaseletmein"
    ```

    **Installation**: Install Neo4j extras:

    ```bash  theme={null}
    pip install "cognee[neo4j]"
    ```

    **Docker Setup**: Start the bundled Neo4j service with APOC + GDS plugins:

    ```bash  theme={null}
    docker compose --profile neo4j up -d
    ```
  </Accordion>

  <Accordion title="Neptune (Graph-only)">
    Use Amazon Neptune for cloud-based graph storage.

    ```dotenv  theme={null}
    GRAPH_DATABASE_PROVIDER="neptune"
    GRAPH_DATABASE_URL="neptune://<GRAPH_ID>"
    # AWS credentials via environment or default SDK chain
    ```

    **Installation**: Install Neptune extras:

    ```bash  theme={null}
    pip install "cognee[neptune]"
    ```

    **Note**: AWS credentials should be configured via environment variables or AWS SDK.
  </Accordion>

  <Accordion title="Neptune Analytics (Hybrid)">
    Use Amazon Neptune Analytics as a hybrid vector + graph backend.

    ```dotenv  theme={null}
    GRAPH_DATABASE_PROVIDER="neptune_analytics"
    GRAPH_DATABASE_URL="neptune-graph://<GRAPH_ID>"
    # AWS credentials via environment or default SDK chain
    ```

    **Installation**: Install Neptune extras:

    ```bash  theme={null}
    pip install "cognee[neptune]"
    ```

    **Note**: This is the same as the vector store configuration. Neptune Analytics serves both purposes.
  </Accordion>
</AccordionGroup>

## Advanced Options

<Accordion title="Backend Access Control">
  Enable per-user dataset isolation for multi-tenant scenarios.

  ```dotenv  theme={null}
  ENABLE_BACKEND_ACCESS_CONTROL="true"
  ```

  This feature is available for Kuzu and other supported graph stores.
</Accordion>

## Provider Comparison

<Accordion title="Graph Store Comparison">
  | Provider          | Setup           | Performance | Use Case              |
  | ----------------- | --------------- | ----------- | --------------------- |
  | Kuzu              | Zero setup      | Good        | Local development     |
  | Kuzu-remote       | Server required | Good        | Remote access         |
  | Neo4j             | Server required | Excellent   | Production            |
  | Neptune           | AWS required    | Excellent   | Cloud solution        |
  | Neptune Analytics | AWS required    | Excellent   | Hybrid cloud solution |
</Accordion>

## Important Considerations

<Accordion title="Data Location">
  * **Local providers** (Kuzu): Graph files are created automatically under `SYSTEM_ROOT_DIRECTORY`
  * **Remote providers** (Neo4j, Neptune): Require running services or cloud setup
  * **Path management**: Local graphs are managed automatically, no manual path configuration needed
</Accordion>

<Accordion title="Performance Notes">
  * **Kuzu**: Single-file storage with good local performance
  * **Neo4j**: Excellent for production workloads with proper indexing
  * **Neptune**: Cloud-scale performance with managed infrastructure
  * **Hybrid solutions**: Combine graph and vector capabilities in one system
</Accordion>

## Notes

* **Backend Access Control**: When enabled, Kuzu supports per-user dataset isolation
* **Path Management**: Local Kuzu databases are created automatically under the system directory
* **Cloud Integration**: Neptune providers require AWS credentials and proper IAM permissions

<Columns cols={3}>
  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Configure vector databases for embedding storage
  </Card>

  <Card title="Relational Databases" icon="database" href="/setup-configuration/relational-databases">
    Set up SQLite or Postgres for metadata storage
  </Card>

  <Card title="Overview" icon="settings" href="/setup-configuration/overview">
    Return to setup configuration overview
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt