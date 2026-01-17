# Architecture

> Understanding Cognee's storage architecture and system components

# Cognee Architecture

## Why multiple stores

No single database can handle all aspects of memory. Cognee combines three complementary storage systems. Each one plays a different role, and together they make your data both **searchable** and **connected**.

* **Relational store** — Tracks your documents, their chunks, and provenance
  (i.e. where each piece of data came from and how it's linked to the source).

* **Vector store** — Holds embeddings for semantic similarity
  (i.e. numerical representations that let Cognee find conceptually related text, even if the wording is different).

* **Graph store** — Captures entities and relationships in a knowledge graph
  (i.e. nodes and edges that let Cognee understand structure and navigate connections between concepts).

Cognee ships with lightweight defaults that run locally, and you can swap in production-ready backends when needed (see [Setup](/getting-started/installation)).

## What is stored where

Roughly speaking:

* The **relational store** handles document-level metadata and provenance.
* The **vector store** contains semantic fingerprints of chunks and [DataPoints](./building-blocks/datapoints).
* The **graph store** captures higher-level structure in the form of entities and relationships.

There is some overlap: for efficiency, parts of the same information may be indexed in more than one store.

## How they are used

The stores play different roles depending on the phase:

* The **relational store** matters most during *cognification*, keeping track of documents, chunks, and where each piece of information comes from.
* The **vector** and **graph** stores come into play during *search and retrieval*:
  * **Semantic searches** (vector): find conceptually related passages based on embeddings
  * **Structural searches** (graph): explore entities and relationships using Cypher directly
  * **Hybrid searches** (vector + graph): combine both perspectives to surface results that are contextually rich and structurally precise.

<Columns cols={3}>
  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    See how Add, Cognify, and Search use the storage systems
  </Card>

  <Card title="Building Blocks" icon="puzzle" href="/core-concepts/building-blocks/datapoints">
    Learn about DataPoints, Tasks, and Pipelines that feed into storage
  </Card>

  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Explore different query types and modes that leverage the architecture
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt