# Overview

> Learn about Cognee's core concepts, architecture, and how to get started

## Introduction

Cognee is an open source tool and platform that transforms your raw data into intelligent, searchable memory. It combines vector search with graph databases to make your data both searchable by meaning and connected by relationships.

<Info>**Dual storage architecture**  gives you both semantic search and structural reasoning</Info>

<Tip>**Modular design** composes [Tasks](./building-blocks/tasks), [Pipelines](./building-blocks/pipelines), and [DataPoints](./building-blocks/datapoints)</Tip>

<Note>**Main operations** handle the complete workflow from ingestion to search: add, cognify, memify, search.</Note>

## Table of Contents

<Accordion title="Architecture">
  Cognee uses three complementary storage systems, each playing a different role:

  * **Relational store** — Tracks documents, chunks, and provenance (where data came from and how it's linked)
  * **Vector store** — Holds embeddings for semantic similarity (numerical representations that find conceptually related content)
  * **Graph store** — Captures entities and relationships in a knowledge graph (nodes and edges that show connections between concepts)

  This architecture makes your data both **searchable** (via vectors) and **connected** (via graphs). Cognee ships with lightweight defaults that run locally, and you can swap in production-ready backends when needed.

  For detailed information about the storage architecture, see [Architecture](./architecture).
</Accordion>

<Accordion title="Building Blocks">
  Cognee's processing system is built from three fundamental components:

  * **[DataPoints](./building-blocks/datapoints)** — Structured data units that become graph nodes, carrying both content and metadata for indexing
  * **[Tasks](./building-blocks/tasks)** — Individual processing units that transform data, from text analysis to relationship extraction
  * **[Pipelines](./building-blocks/pipelines)** — Orchestration of Tasks into coordinated workflows, like assembly lines for data transformation

  These building blocks work together to create a flexible system where you can:

  * Use built-in Tasks for common operations
  * Create custom Tasks for domain-specific logic by extending DataPoints
  * Compose Tasks into Pipelines that match your workflow
</Accordion>

<Accordion title="Main Operations">
  Cognee provides four main operations that users interact with:

  * **[Add](./main-operations/add)** — Ingest and prepare data for processing, handling various file formats and data sources
  * **[Cognify](./main-operations/cognify)** — Create knowledge graphs from processed data through cognitive processing and entity extraction
  * **[Memify](./main-operations/memify)** — Optional semantic enrichment of the graph for enhanced understanding *(coming soon)*
  * **[Search](./main-operations/search)** — Query and retrieve information using semantic similarity, graph traversal, or hybrid approaches

  **Note:** Search works great with just the basic Add → Cognify → Search workflow. Memify is an optional enhancement that will provide additional semantic enrichment when available.
</Accordion>

<Accordion title="Further Concepts">
  Beyond the core workflow, Cognee offers advanced features for sophisticated knowledge management:

  * **[Node Sets](./further-concepts/node-sets)** — Tagging and organization system that helps categorize and filter your knowledge base content
  * **[Ontologies](./further-concepts/ontologies)** — External knowledge grounding through RDF/XML ontologies that connect your data to established knowledge structures

  These concepts extend Cognee's capabilities for:

  * **Organization** — Managing growing knowledge bases with systematic tagging
  * **Knowledge grounding** — Connecting your data to external, validated knowledge sources
  * **Domain expertise** — Leveraging existing ontologies for specialized fields like medicine, finance, or research
</Accordion>

## Next steps

A good way to learn Cognee is to start with its [architecture](./architecture), move on to [building blocks](./building-blocks/datapoints), practice the [main operations](./main-operations/add), and finally explore [advanced features](./further-concepts/node-sets).

<Columns cols={3}>
  <Card title="Architecture" icon="building" href="/core-concepts/architecture">
    Understand Cognee's three storage systems and how they work together
  </Card>

  <Card title="Building Blocks" icon="puzzle" href="/core-concepts/building-blocks/datapoints">
    Learn about DataPoints, Tasks, and Pipelines that power the system
  </Card>

  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    Master Add, Cognify, and Search operations for your workflows
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt