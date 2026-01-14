# FalkorDB

> Use FalkorDB as both a graph and vector store (hybrid store) through a community-maintained adapter

FalkorDB is an open-source graph database optimized for GraphRAG.
It supports both cloud-hosted and self-hosted deployments.

<Note>
  Cognee can use FalkorDB as both a [vector store](/setup-configuration/vector-stores) and a
  [graph store](/setup-configuration/graph-stores) backend through this
  [community-maintained](/setup-configuration/community-maintained/overview) [adapter](https://github.com/topoteretes/cognee-community/tree/main/packages/hybrid/falkordb).
</Note>

## Installation

This adapter is a separate package from core Cognee.
Before installing, complete the [Cognee installation](/getting-started/installation) and ensure
your environment is configured with [LLM and embedding providers](/setup-configuration/overview).
After that, install the adapter package:

```bash  theme={null}
uv pip install cognee-community-hybrid-adapter-falkor
```

## Configuration

Run a local FalkorDB instance:

```bash  theme={null}
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:edge
```

Configure in Python:

```python  theme={null}
from cognee_community_hybrid_adapter_falkor import register
from cognee import config

config.set_vector_db_config(
        {
            "vector_db_provider": "falkor",
            "vector_db_url": "localhost",
            "vector_db_port": 6379,
        }
    )
config.set_graph_db_config(
    {
        "graph_database_provider": "falkor",
        "graph_database_url": "localhost",
        "graph_database_port": 6379,
    }
)
```

Or via environment variables:

```dotenv  theme={null}
VECTOR_DB_PROVIDER="falkor"
VECTOR_DB_URL="http://localhost:6379"
VECTOR_DB_KEY=""

GRAPH_DATABASE_PROVIDER="falkor"
GRAPH_DATABASE_URL="localhost"
GRAPH_DATABASE_PORT="6379"
```

## Important Notes

<Accordion title="Adapter Registration">
  Import `register` from the adapter package before using FalkorDB with Cognee. This registers the adapter with Cognee's provider system.
</Accordion>

<Accordion title="Embedding Dimensions">
  Ensure `EMBEDDING_DIMENSIONS` matches your embedding model. See [Embedding Providers](/setup-configuration/embedding-providers) for configuration.

  Changing dimensions requires recreating collections or running `prune.prune_system()`.
</Accordion>

## Resources

<CardGroup cols={3}>
  <Card title="FalkorDB Docs" icon="book" href="https://docs.falkordb.com/">
    Official documentation
  </Card>

  <Card title="Adapter Source" icon="github" href="https://github.com/topoteretes/cognee-community/tree/main/packages/hybrid/falkordb">
    GitHub repository
  </Card>

  <Card title="Extended Example" icon="lightbulb" href="https://github.com/topoteretes/cognee-community/tree/main/packages/hybrid/falkordb/examples/example.py">
    FAQ docs assistant example.
  </Card>
</CardGroup>

<Columns cols={3}>
  <Card title="Graph Stores" icon="database" href="/setup-configuration/graph-stores">
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