
## FAQ

### General

**What is Cognee?**
Cognee is an open-source knowledge engine that lets you ingest data in any format or structure and continuously learns to provide the right context for AI agents. It combines vector search, graph databases, and cognitive science approaches to make documents both searchable by meaning and connected by relationships as they change and evolve.

**How is Cognee different from traditional RAG?**
Traditional RAG relies solely on vector similarity search. Cognee adds a knowledge graph layer that captures relationships between entities, enabling multi-hop reasoning, temporal awareness, and structured retrieval. This means your agent can understand not just what's similar, but how concepts relate to each other over time.

**Is Cognee production-ready?**
Yes. Cognee is used in production by teams building AI agents that require personalized, dynamic memory. It supports both managed (Cognee Cloud) and self-hosted deployments.

### Installation & Setup

**What are the system requirements?**
Cognee requires Python 3.10+ and works on Linux, macOS, and Windows. For the full stack (vector search + graph database), we recommend at least 4GB RAM.

**How do I install Cognee?**
```bash
pip install cognee
```

**What databases does Cognee use?**
Cognee uses a combination of vector databases (for semantic search) and graph databases (for relationship storage). The default setup uses local databases, but you can configure PostgreSQL, Neo4j, or other backends for production.

### Data Ingestion

**What data formats does Cognee support?**
Cognee can ingest data in any format or structure — PDFs, text files, CSV, JSON, URLs, databases, and more. The knowledge engine automatically extracts entities and relationships from your data.

**How does Cognee learn from my data?**
Cognee uses a multi-step pipeline: it ingests raw data, extracts entities and relationships, builds a knowledge graph, and creates vector embeddings. As new data arrives, the graph evolves and learns from patterns in your information.

**Can I update or delete ingested data?**
Yes. Cognee supports incremental updates and data removal. The knowledge graph adapts as your data changes, maintaining consistency across the system.

### Memory & Retrieval

**How does Cognee provide memory for AI agents?**
Cognee stores ingested data as structured knowledge (entities, relationships, timelines) and semantic embeddings. When an agent needs context, Cognee retrieves both semantically similar content and relationally connected information, giving the agent a richer understanding.

**What is the difference between `cognee.remember()` and `cognee.recall()`?**
`cognee.remember()` stores information into the knowledge engine (ingestion + graph building). `cognee.recall()` retrieves relevant context based on a query, using both vector similarity and graph traversal.

### Deployment

**What deployment options are available?**
- **Cognee Cloud**: Fully managed service, no infrastructure to maintain
- **Modal**: Serverless, auto-scaling, GPU workloads
- **Railway**: Simplest PaaS with native Postgres
- **Fly.io**: Edge deployment with persistent volumes
- **Render**: Simple PaaS with managed Postgres
- **Daytona**: Cloud sandboxes

See the [`distributed/`](distributed/) folder for deployment scripts and configurations.

**Can I self-host Cognee?**
Yes. All deployment options except Cognee Cloud are self-hosted. See the deployment table in the README for commands.

### Troubleshooting

**My knowledge graph is empty after ingestion. What should I check?**
Verify that your data source is accessible and the ingestion pipeline completed successfully. Check the logs for any errors during entity extraction. Ensure your data contains enough structured information for the engine to extract meaningful entities and relationships.

**How do I configure a custom LLM provider?**
Cognee uses LLMs for entity extraction and relationship building. Configure your provider via environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or set a custom endpoint in your configuration.

**Where can I get help?**
- [Discord](https://discord.gg/bcy8xFAtfd) — join the community
- [Reddit r/AIMemory](https://www.reddit.com/r/AIMemory/) — discussions and tips
- [GitHub Issues](https://github.com/topoteretes/cognee/issues) — bug reports and feature requests
- [Docs](https://docs.cognee.ai) — detailed documentation
