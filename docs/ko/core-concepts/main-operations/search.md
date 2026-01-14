# Search

> Query your AI memory with vectors, graphs, and LLMs

## What is search

`search` lets you ask questions over everything you've ingested and cognified.\
Under the hood, Cognee blends **vector similarity**, **graph structure**, and **LLM reasoning** to return answers with context and provenance.

## The big picture

* **Dataset-aware**: searches run against one or more datasets you can read *(requires `ENABLE_BACKEND_ACCESS_CONTROL=true`)*
* **Multiple modes**: from simple chunk lookup to graph-aware Q\&A
* **Hybrid retrieval**: vectors find relevant pieces; graphs provide structure; LLMs compose answers
* **Conversational memory**: use `session_id` to maintain conversation history across searches *(requires caching enabled)*
* **Safe by default**: permissions are checked before any retrieval
* **Observability**: telemetry is emitted for query start/completion

<Warning>
  **Dataset scoping** requires specific configuration. See [permissions system](../permissions-system/datasets#dataset-isolation) for details on access control requirements and supported database setups.
</Warning>

## Where search fits

Use `search` after you've run `.add` and `.cognify`.
At that point, your dataset has chunks, summaries, embeddings, and a knowledge graph—so queries can leverage both **similarity** and **structure**.

## How it works (conceptually)

1. **Scope & permissions**\
   Resolve target datasets (by name or id) and enforce read access.

2. **Mode dispatch**\
   Pick a search mode (default: **graph-aware completion**) and route to its retriever.

3. **Retrieve → (optional) generate**\
   Collect context via vectors and/or graph traversal; some modes then ask an LLM to compose a final answer.

4. **Return results**\
   Depending on mode: answers, chunks/summaries with metadata, graph records, Cypher results, or code contexts.

For a practical guide to using search with examples and detailed parameter explanations, see [Search Basics](/guides/search-basics).

<Accordion title="GRAPH_COMPLETION (default)" defaultOpen={true}>
  Graph-aware question answering.

  * **What it does**: Finds relevant graph triplets using vector hints across indexed fields, resolves them into readable context, and asks an LLM to answer your question grounded in that context.
  * **Why it’s useful**: Combines fuzzy matching (vectors) with precise structure (graph) so answers reflect relationships, not just nearby text.
  * **Typical output**: A natural-language answer with references to the supporting graph context.
</Accordion>

<Accordion title="RAG_COMPLETION">
  Retrieve-then-generate over text chunks.

  * **What it does**: Pulls top-k chunks via vector search, stitches a context window, then asks an LLM to answer.
  * **When to use**: You want fast, text-only RAG without graph structure.
  * **Output**: An LLM answer grounded in retrieved chunks.
</Accordion>

<Accordion title="CHUNKS">
  Direct chunk retrieval.

  * **What it does**: Returns the most similar text chunks to your query via vector search.
  * **When to use**: You want raw passages/snippets to display or post-process.
  * **Output**: Chunk objects with metadata.
</Accordion>

<Accordion title="SUMMARIES">
  Search over precomputed summaries.

  * **What it does**: Vector search on `TextSummary` content for concise, high-signal hits.
  * **When to use**: You prefer short summaries instead of full chunks.
  * **Output**: Summary objects with provenance.
</Accordion>

<Accordion title="GRAPH_SUMMARY_COMPLETION">
  Graph-aware summary answering.

  * **What it does**: Builds graph context like GRAPH\_COMPLETION, then condenses it before answering.
  * **When to use**: You want a tighter, summary-first response.
  * **Output**: A concise answer grounded in graph context.
</Accordion>

<Accordion title="GRAPH_COMPLETION_COT">
  Chain-of-thought over the graph.

  * **What it does**: Iterative rounds of graph retrieval and LLM reasoning to refine the answer.
  * **When to use**: Complex questions that benefit from stepwise reasoning.
  * **Output**: A refined answer produced through multiple reasoning steps.
</Accordion>

<Accordion title="GRAPH_COMPLETION_CONTEXT_EXTENSION">
  Iterative context expansion.

  * **What it does**: Starts with initial graph context, lets the LLM suggest follow-ups, fetches more graph context, repeats.
  * **When to use**: Open-ended queries that need broader exploration.
  * **Output**: An answer assembled after expanding the relevant subgraph.
</Accordion>

<Accordion title="NATURAL_LANGUAGE">
  Natural language to Cypher to execution.

  * **What it does**: Infers a Cypher query from your question using the graph schema, runs it, returns the results.
  * **When to use**: You want structured graph answers without writing Cypher.
  * **Output**: Executed graph results.
</Accordion>

<Accordion title="CYPHER">
  Run Cypher directly.

  * **What it does**: Executes your Cypher query against the graph database.
  * **When to use**: You know the schema and want full control.
  * **Output**: Raw query results.
</Accordion>

<Accordion title="CODE">
  Code-focused retrieval.

  * **What it does**: Interprets your intent (files/snippets), searches code embeddings and related graph nodes, and assembles relevant source.
  * **When to use**: Codebases indexed by Cognee.
  * **Output**: Structured code contexts and related graph information.
</Accordion>

<Accordion title="FEELING_LUCKY">
  Automatic mode selection.

  * **What it does**: Uses an LLM to pick the most suitable search mode for your query, then runs it.
  * **When to use**: You’re not sure which mode fits best.
  * **Output**: Results from the selected mode.
</Accordion>

<Accordion title="FEEDBACK">
  Store feedback on recent interactions.

  * **What it does**: Records user feedback on recent answers and links it to the associated graph elements for future tuning.
  * **When to use**: Closing the loop on quality and relevance.
  * **Output**: A feedback record tied to recent interactions.
</Accordion>

<Columns cols={2}>
  <Card title="Add" icon="plus" href="/core-concepts/main-operations/add">
    First bring data into Cognee
  </Card>

  <Card title="Cognify" icon="brain-cog" href="/core-concepts/main-operations/cognify">
    Build the knowledge graph that search queries
  </Card>

  <Card title="Architecture" icon="building" href="/core-concepts/architecture">
    Understand how vector and graph stores work together
  </Card>

  <Card title="Sessions and Caching" icon="message-square" href="/core-concepts/sessions-and-caching">
    Enable conversational memory with sessions
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt