# Cognify

> Transforming ingested data into a knowledge graph with embeddings, chunks, and summaries

## What is the cognify operation

The `.cognify` operation takes the data you ingested with [Add](../main-operations/add) and turns plain text into structured knowledge: chunks, embeddings, summaries, nodes, and edges that live in Cognee's vector and graph stores. It prepares your data for downstream operations like [Search](../main-operations/search).

* **Transforms ingested data**: builds chunks, embeddings, and summaries; always comes **after [Add](../main-operations/add)**
* **Graph creation**: extracts entities and relationships to form a knowledge graph
* **Vector indexing**: makes everything searchable via embeddings
* **Dataset-scoped**: runs per dataset, respecting ownership and permissions
* **Incremental loading**: you can run `.cognify` multiple times as your dataset grows, and Cognee will skip what's already processed

## What happens under the hood

The `.cognify` pipeline is made of six ordered [Tasks](../building-blocks/tasks). Each task takes the output of the previous one and moves your data closer to becoming a searchable knowledge graph.

1. **Classify documents** — wrap each ingested file as a `Document` object with metadata and optional node sets
2. **Check permissions** — enforce that you have the right to modify the target dataset
3. **Extract chunks** — split documents into smaller pieces (paragraphs, sections)
4. **Extract graph** — use LLMs to identify entities and relationships, inserting them into the graph DB
5. **Summarize text** — generate summaries for each chunk, stored as `TextSummary` [DataPoints](../building-blocks/datapoints)
6. **Add data points** — embed nodes and summaries, write them into the vector store, and update graph edges

The result is a fully searchable, structured knowledge graph connected to your data.

## After cognify finishes

When `.cognify` completes for a dataset:

* **DocumentChunks** exist in memory as the granular breakdown of your files
* **Summaries** are stored and indexed in the vector database for semantic search
* **Knowledge graph nodes and edges** are committed to the graph database
* **Dataset metadata** is updated with token counts and pipeline status
* Your dataset is now **query-ready**: you can run [Search](../main-operations/search) or graph queries immediately

## Examples and details

<Accordion title="Pipeline tasks (detailed)">
  1. **Classify documents**
     * Turns raw `Data` rows into `Document` objects
     * Chooses the right document type (PDF, text, image, audio, etc.)
     * Attaches metadata and optional node sets

  2. **Check permissions**
     * Verifies that the user has write access to the dataset

  3. **Extract chunks**
     * Splits documents into `DocumentChunk`s using a chunker
     * Updates token counts in the relational DB

  4. **Extract graph**
     * Calls the LLM to extract entities and relationships
     * Deduplicates nodes and edges, commits to the graph DB

  5. **Summarize text**
     * Generates concise summaries per chunk
     * Stores them as `TextSummary` [DataPoints](../building-blocks/datapoints) for vector search

  6. **Add data points**
     * Converts summaries and other [DataPoints](../building-blocks/datapoints) into graph + vector nodes
     * Embeds them in the vector store, persists in the graph DB
</Accordion>

<Accordion title="Datasets and permissions">
  * Cognify always runs on a dataset
  * You must have **write access** to the dataset
  * Permissions are enforced at pipeline start
  * Each dataset maintains its own cognify status and token counts
</Accordion>

<Accordion title="Incremental loading">
  * By default, `.cognify` processes all data in a dataset
  * With `incremental_loading=True`, only new or updated files are processed
  * Saves time and compute for large, evolving datasets
</Accordion>

<Accordion title="Final outcome">
  * Vector database contains embeddings for summaries and nodes
  * Graph database contains entities and relationships
  * Relational database tracks token counts and pipeline run status
  * Your dataset is now ready for [Search](../main-operations/search) (semantic or graph-based)
</Accordion>

<Columns cols={3}>
  <Card title="Add" icon="plus" href="/core-concepts/main-operations/add">
    First bring data into Cognee
  </Card>

  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Query embeddings or graph structures built by Cognify
  </Card>

  <Card title="Building Blocks" icon="puzzle" href="/core-concepts/building-blocks/datapoints">
    Learn about DataPoints, Tasks, and Pipelines
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt