# NodeSets

> Tagging and grouping data in Cognee

## What are NodeSets?

A **NodeSet** lets you group parts of your AI memory at the dataset level. You create them as a simple list of tags when adding data to Cognee:
await cognee.add(..., node\_set=\["projectA","finance"])
These tags travel with your data into the knowledge graph, where they become first-class nodes connected with belongs\_to\_set edges — and you can later filter searches to only those subsets.

## How they flow through Cognee

* **[Add](../main-operations/add)**:
  * NodeSets are attached as simple tags to datasets or documents
  * This happens when you first ingest data
* **[Cognify](../main-operations/cognify)**:
  * carried into Documents and Chunks
  * materialized as real `NodeSet` nodes in the graph
  * connected with `belongs_to_set` edges
* **[Search](../main-operations/search)**:
  * NodeSets act as entry points into the graph
  * Queries can be scoped to only nodes linked to specific NodeSets
  * This lets you search within a tagged subset of your data

## Why they matter

* Provide a lightweight way to organize and tag your data
* Enable graph-based filtering, traversal, and reporting
* Ideal for creating project-, domain-, or user-defined subsets of your knowledge graph

## Example

```python  theme={null}
import asyncio
import cognee

async def main():
    # reset Cognee’s memory and metadata for a clean run
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # add a document linked only to the "AI_Memory" node set
    await cognee.add(
        "Cognee builds AI memory from raw documents.",
        node_set=["AI_Memory"]
    )

    # add a document linked to both "AI_Memory" and "Graph_RAG" node sets
    await cognee.add(
        "Cognee combines vector search with graph reasoning.",
        node_set=["AI_Memory", "Graph_RAG"]
    )

    # build the knowledge graph by extracting entities and relationships
    await cognee.cognify()

if __name__ == "__main__":
    asyncio.run(main())
```

## What just happened?

* You reset Cognee’s memory so you’re working with a clean graph.
* You added two documents, each tagged with one or more `NodeSet` labels.
  * The first document is only linked to `AI_Memory`.
  * The second document is linked to both `AI_Memory` and `Graph_RAG`.
* When you ran `cognify()`, Cognee:
  * Created `NodeSet` nodes (`AI_Memory`, `Graph_RAG`) in the graph.
  * Attached each document to the corresponding NodeSets.
  * Extracted entities and relationships from the documents, then linked those entities back to the same NodeSets.

This means the tags you add flow down into the extracted entities:

* **“Cognee”** appears in both documents → connects to **both NodeSets**.
* **“AI memory”** appears only in the first → connects only to **AI\_Memory**.
* **“Vector search”** appears only in the second → connects to **both** since that document belongs to **AI\_Memory** and **Graph\_RAG**.

Your NodeSets now unlock powerful search and navigation capabilities:

* You can filter searches by NodeSet.
* You can scope queries to specific NodeSets.
* You can navigate data by project or domain using NodeSets.

<Columns cols={3}>
  <Card title="Add" icon="plus" href="/core-concepts/main-operations/add">
    Where NodeSets are first attached
  </Card>

  <Card title="Cognify" icon="brain-cog" href="/core-concepts/main-operations/cognify">
    How NodeSets are promoted into graph nodes
  </Card>

  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Use NodeSets as anchors in queries
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt