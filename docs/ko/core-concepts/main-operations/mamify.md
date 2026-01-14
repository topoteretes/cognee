# Memify

> Semantic enrichment of existing knowledge graphs with derived facts

## What is the memify operation

The `.memify` operation enriches existing knowledge graphs by extracting derived facts and creating new associations from your already-processed data. Unlike [Add](../main-operations/add) and [Cognify](../main-operations/cognify), memify works on existing graph structures to add semantic understanding and deeper contextual relationships.

* **Graph enrichment**: operates on existing knowledge graphs created by [Cognify](../main-operations/cognify)
* **Derived facts**: creates new nodes and edges from existing context without re-ingesting data
* **Semantic enhancement**: adds coding rules, associations, and other derived knowledge
* **Pipeline-based**: uses extraction and enrichment tasks to process subgraphs
* **Incremental**: can be run multiple times to add new derived facts as needed

## Where memify fits

Use `.memify` after you've completed the [Add](../main-operations/add) → [Cognify](../main-operations/cognify) workflow:

* **Prerequisites**: requires an existing knowledge graph with chunks, embeddings, and graph structure
* **Enhancement phase**: adds semantic understanding and derived facts to your existing data
* **Optional enrichment**: not required for basic search, but adds valuable context and associations

## What happens under the hood

The `.memify` pipeline processes your existing knowledge graph through two main phases:

1. **Extraction phase** — pulls relevant subgraphs or chunks from your existing knowledge graph
2. **Enrichment phase** — applies enrichment tasks to create new nodes and edges from existing context

The default memify tasks include:

* **Extract subgraph chunks**: identifies relevant portions of your graph for processing
* **Add rule associations**: creates coding rules and other derived facts from the extracted context

## After memify finishes

When `.memify` completes:

* **New derived facts** are added to your knowledge graph as additional nodes and edges
* **Enhanced searchability**: specialized search types like `SearchType.CODING_RULES` become available
* **Richer context**: your existing data now includes semantic associations and derived knowledge
* **No data re-ingestion**: all enrichment happens on your existing graph structure

## Examples and details

<Accordion title="Default behavior">
  * **Extraction**: `extract_subgraph_chunks` - pulls relevant chunks from your graph
  * **Enrichment**: `add_rule_associations` - creates coding rules and associations
  * **Output**: new nodes and edges added to your existing knowledge graph
</Accordion>

<Accordion title="Custom tasks">
  * You can specify custom extraction and enrichment tasks
  * Extraction tasks determine what parts of the graph to process
  * Enrichment tasks define what derived facts to create
  * Tasks can be chained together for complex enrichment workflows
</Accordion>

<Accordion title="Search integration">
  * Enriched graphs support specialized search types
  * `SearchType.CODING_RULES` for finding coding guidelines
  * Other search modes can leverage the new derived facts
  * Enhanced context improves answer quality and relevance
</Accordion>

<Accordion title="Incremental processing">
  * Can be run multiple times on the same dataset
  * Only processes new or updated graph elements by default
  * Safe to re-run as it adds rather than replaces existing data
</Accordion>

<Columns cols={3}>
  <Card title="Cognify" icon="brain-cog" href="/core-concepts/main-operations/cognify">
    Build the knowledge graph that memify enriches
  </Card>

  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Query the enriched graph with specialized search types
  </Card>

  <Card title="Custom Tasks" icon="workflow" href="/guides/custom-tasks-pipelines">
    Learn how to create custom memify tasks
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt