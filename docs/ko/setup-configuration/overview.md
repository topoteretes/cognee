# Setup Configuration

> Configure Cognee to use your preferred LLM, embedding engine, and storage backends

Configure Cognee to use your preferred LLM, embedding engine, relational database, vector store, and graph store via environment variables in a local `.env` file.

This section provides beginner-friendly guides for setting up different backends, with detailed technical information available in expandable sections.

## What You Can Configure

Cognee uses a flexible architecture that lets you choose the best tools for your needs. We recommend starting with the defaults to get familiar with Cognee, then customizing each component as needed:

* **[LLM Providers](./llm-providers)** — Choose from OpenAI, Azure OpenAI, Google Gemini, Anthropic, Ollama, or custom providers (like vLLM) for text generation and reasoning tasks
* **[Structured Output Backends](./structured-output-backends)** — Configure LiteLLM + Instructor or BAML for reliable data extraction from LLM responses
* **[Embedding Providers](./embedding-providers)** — Select from OpenAI, Azure OpenAI, Google Gemini, Mistral, Ollama, Fastembed, or custom embedding services to create vector representations for semantic search
* **[Relational Databases](./relational-databases)** — Use SQLite for local development or Postgres for production to store metadata, documents, and system state
* **[Vector Stores](./vector-stores)** — Store embeddings in LanceDB, PGVector, ChromaDB, FalkorDB, or Neptune Analytics for similarity search
* **[Graph Stores](./graph-stores)** — Build knowledge graphs with Kuzu, Kuzu-remote, Neo4j, Neptune, or Neptune Analytics to manage relationships and reasoning
* **[Dataset Separation & Access Control](./permissions)** — Configure dataset-level permissions and isolation
* **[Sessions & Caching](../core-concepts/sessions-and-caching)** — Enable conversational memory with Redis or filesystem cache adapters

<Warning>
  Dataset isolation is not enabled by default; see [how to enable it](../core-concepts/permissions-system/datasets#dataset-isolation).
</Warning>

## Observability & Telemetry

Cognee includes built-in telemetry to help you monitor and debug your knowledge graph operations. You can control telemetry behavior with environment variables:

* **`TELEMETRY_DISABLED`** (boolean, optional): Set to `true` to disable all telemetry collection (default: `false`)

When telemetry is enabled, Cognee automatically collects:

* Search query performance metrics
* Processing pipeline execution times
* Error rates and debugging information
* System resource usage

<Info>
  Telemetry data helps improve Cognee's performance and reliability. It's collected anonymously and doesn't include your actual data content.
</Info>

## Configuration Workflow

1. Install Cognee with all optional dependencies:
   * **Local setup**: `uv sync --all-extras`
   * **Library**: `pip install "cognee[all]"`
2. Create a `.env` file in your project root (if you haven't already) — see [Installation](/getting-started/installation) for details
3. Choose your preferred providers and follow the configuration instructions from the guides below

<Warning>
  **Configuration Changes**: If you've already run Cognee with default settings and are now changing your configuration (e.g., switching from SQLite to Postgres, or changing vector stores), you should call pruning operations before the next cognification to ensure data consistency.
</Warning>

<Warning>
  **LLM/Embedding Configuration**: If you configure only LLM or only embeddings, the other defaults to OpenAI. Ensure you have a working OpenAI API key, or configure both LLM and embeddings to avoid unexpected defaults.
</Warning>

<Columns cols={3}>
  <Card title="LLM Providers" icon="brain" href="/setup-configuration/llm-providers">
    Configure OpenAI, Azure, Gemini, Anthropic, Ollama, or custom LLM providers (like vLLM)
  </Card>

  <Card title="Structured Output Backends" icon="code" href="/setup-configuration/structured-output-backends">
    Configure LiteLLM + Instructor or BAML for reliable data extraction
  </Card>

  <Card title="Embedding Providers" icon="layers" href="/setup-configuration/embedding-providers">
    Set up OpenAI, Mistral, Ollama, Fastembed, or custom embedding services
  </Card>
</Columns>

<Columns cols={3}>
  <Card title="Relational Databases" icon="database" href="/setup-configuration/relational-databases">
    Choose between SQLite for local development or Postgres for production
  </Card>

  <Card title="Vector Stores" icon="database" href="/setup-configuration/vector-stores">
    Configure LanceDB, PGVector, ChromaDB, FalkorDB, or Neptune Analytics
  </Card>

  <Card title="Graph Stores" icon="network" href="/setup-configuration/graph-stores">
    Set up Kuzu, Neo4j, or Neptune for knowledge graph storage
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt