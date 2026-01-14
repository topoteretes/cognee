# Qdrant

> Use Qdrant as a vector store through a community-maintained adapter

Qdrant is a vector search engine that stores embeddings and performs similarity searches. It supports both cloud-hosted and self-hosted deployments.

<Note>
  Cognee can use Qdrant as a [vector store](/setup-configuration/vector-stores) backend through this [community-maintained](/setup-configuration/community-maintained/overview) [adapter](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/qdrant).
</Note>

## Installation

This adapter is a separate package from core Cognee. Before installing, complete the [Cognee installation](/getting-started/installation) and ensure your environment is configured with [LLM and embedding providers](/setup-configuration/overview). After that, install the adapter package:

```bash  theme={null}
uv pip install cognee-community-vector-adapter-qdrant
```

## Configuration

<Tabs>
  <Tab title="Docker (Local)">
    Run a local Qdrant instance:

    ```bash  theme={null}
    docker run -p 6333:6333 -p 6334:6334 \
        -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
        qdrant/qdrant
    ```

    Configure in Python:

    ```python  theme={null}
    from cognee_community_vector_adapter_qdrant import register
    from cognee import config

    register()

    config.set_vector_db_config({
        "vector_db_provider": "qdrant",
        "vector_db_url": "http://localhost:6333",
        "vector_db_key": "",
    })
    ```

    Or via environment variables:

    ```dotenv  theme={null}
    VECTOR_DB_PROVIDER="qdrant"
    VECTOR_DB_URL="http://localhost:6333"
    VECTOR_DB_KEY=""
    ```
  </Tab>

  <Tab title="Qdrant Cloud">
    Get your API key and URL from the [Qdrant Cloud](https://qdrant.tech/documentation/cloud/) dashboard.

    ```python  theme={null}
    from cognee_community_vector_adapter_qdrant import register
    from cognee import config

    register()

    config.set_vector_db_config({
        "vector_db_provider": "qdrant",
        "vector_db_url": "https://your-cluster.qdrant.io",
        "vector_db_key": "your_api_key",
    })
    ```

    Or via environment variables:

    ```dotenv  theme={null}
    VECTOR_DB_PROVIDER="qdrant"
    VECTOR_DB_URL="https://your-cluster.qdrant.io"
    VECTOR_DB_KEY="your_api_key"
    ```
  </Tab>
</Tabs>

## Important Notes

<Accordion title="Adapter Registration">
  Import and call `register()` from the adapter package before using Qdrant with Cognee. This registers the adapter with Cognee's provider system.
</Accordion>

<Accordion title="Embedding Dimensions">
  Ensure `EMBEDDING_DIMENSIONS` matches your embedding model. See [Embedding Providers](/setup-configuration/embedding-providers) for configuration.

  Changing dimensions requires recreating collections or running `prune.prune_system()`.
</Accordion>

## Resources

<CardGroup cols={3}>
  <Card title="Qdrant Docs" icon="book" href="https://qdrant.tech/documentation/">
    Official documentation
  </Card>

  <Card title="Adapter Source" icon="github" href="https://github.com/topoteretes/cognee-community/tree/main/packages/vector/qdrant">
    GitHub repository
  </Card>

  <Card title="Extended Example" icon="lightbulb" href="https://github.com/topoteretes/cognee-community/tree/main/packages/vector/qdrant/example.py">
    FAQ docs assistant example.
  </Card>
</CardGroup>

<Columns cols={3}>
  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Official vector providers
  </Card>

  <Card title="Community Overview" icon="users" href="/setup-configuration/community-maintained/overview">
    All community integrations
  </Card>

  <Card title="Setup Overview" icon="settings" href="/setup-configuration/overview">
    Configuration guide
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt