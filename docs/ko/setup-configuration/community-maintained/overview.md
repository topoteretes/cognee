# Adapters Overview

> Adapters and extensions built by the Cognee community

Community-maintained integrations are adapters built and maintained by the Cognee community. These extend Cognee's functionality with additional providers and services.

<Note>
  Community integrations are maintained separately from the core Cognee package. For issues or contributions, visit the [cognee-community repository](https://github.com/topoteretes/cognee-community).
</Note>

## Available Integrations

### Vector Stores

* **[Qdrant](/setup-configuration/community-maintained/qdrant)** — High-performance vector search engine
* **[Milvus](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/milvus)** — Cloud-native vector database (docs coming soon)
* **[Pinecone](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/pinecone)** — Managed vector database (docs coming soon)
* **[Weaviate](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/weaviate)** — Open-source vector search engine (docs coming soon)
* **[Redis](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/redis)** — Redis with vector search capabilities (docs coming soon)
* **[Azure AI Search](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/azureaisearch)** — Azure cognitive search service (docs coming soon)
* **[OpenSearch](https://github.com/topoteretes/cognee-community/tree/main/packages/vector/opensearch)** — OpenSearch vector engine (docs coming soon)

### Hybrid Stores

* **[DuckDB](https://github.com/topoteretes/cognee-community/tree/main/packages/hybrid/duckdb)** — In-process analytical database (docs coming soon)
* **[FalkorDB](/setup-configuration/community-maintained/falkordb)** — Graph database with vector support (docs coming soon)

### Graph Stores

* **[Memgraph](https://github.com/topoteretes/cognee-community/tree/main/packages/graph/memgraph)** — In-memory graph database (docs coming soon)
* **[NetworkX](https://github.com/topoteretes/cognee-community/tree/main/packages/graph/networkx)** — Python graph library adapter (docs coming soon)

### Observability

* **[KeywordsAI](https://github.com/topoteretes/cognee-community/tree/main/packages/observability/keywordsai)** — LLM monitoring and analytics (docs coming soon)

## Contributing

To contribute a new community integration:

1. Fork the [cognee-community repository](https://github.com/topoteretes/cognee-community)
2. Follow the adapter development guide
3. Submit a pull request with your integration
4. Add documentation following the existing patterns

## Support

For community integration support:

* Check the integration's README in the repository
* Open issues in the cognee-community repository
* Join the [Discord community](https://discord.gg/cqF6RhDYWz) for help

<Columns cols={2}>
  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Official vector store providers
  </Card>

  <Card title="Setup Overview" icon="settings" href="/setup-configuration/overview">
    Configuration overview
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt