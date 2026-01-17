# Introduction

> Cognee organizes your data into AI memory. 

<img src="https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=e2c1d3a47bbdd6b77b14ff368186242d" alt="How does AI memory work?" data-og-width="3851" width="3851" data-og-height="1438" height="1438" data-path="images/how-does-ai-memory-work.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=280&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=676ddc9aedcd7d238115f9b2eea45f16 280w, https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=560&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=ab4740f8580d79e2b1acce760d325038 560w, https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=840&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=595f2755c8a11adfb787efe1de5dbc91 840w, https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=1100&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=daeaf30cde7ae27339117efb79aa6f3f 1100w, https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=1650&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=9c731cb949e0703c1a74359a42e7c47a 1650w, https://mintcdn.com/cognee/SLlciL7PTYZfGdB1/images/how-does-ai-memory-work.png?w=2500&fit=max&auto=format&n=SLlciL7PTYZfGdB1&q=85&s=06682afcd72a23bb9d76a94d04273d0d 2500w" />

Give Cognee your documents, and it creates a graph of raw information, extracted concepts, and meaningful relationships you can query.

## Why AI memory matters

When you call an LLM, each request is stateless: it doesn't remember what happened in the last call, and it doesn't know about the rest of your documents.

That makes it hard to build applications that actually use your documents and carry context forward. You need a memory layer that can link your documents together and create the right context for every LLM call.

## How Cognee works

When it comes to your data, Cognee knows what matters. There are three key operations in Cognee:

* **`.add` — Prepare for cognification**\
  Send in your data asynchronously. Cognee cleans and prepares your data so that the memory layer can be created.

* **`.cognify` — Build a knowledge graph with embeddings**\
  Cognee splits your documents into chunks, extract entities, relations, and links it all into a queryable graph, that serves as the basis for the memory layer.

* **`.search` — Query with context**\
  Queries combine vector similarity with graph traversal. Depending on the mode, cognee can fetch raw nodes, explore relationships, or generate natural-language answers through RAG. It always creates the right context for the LLM.

* **`.memify` — Semantic enrichment of the graph** *(coming soon, stay tuned)*\
  Enhance the knowledge graph with semantic understanding and deeper contextual relationships.

## Ready to get started?

<CardGroup cols={2}>
  <Card title="Set up your environment" href="/getting-started/installation" icon="download">
    **Installation Guide**

    Set up your environment and install Cognee to start building AI memory.
  </Card>

  <Card title="Run your first example" href="/getting-started/quickstart" icon="play">
    **Quickstart Tutorial**

    Get started with Cognee by running your first knowledge graph example.
  </Card>

  <Card title="Keep exploring" href="/core-concepts/overview" icon="compass">
    **Core Concepts**

    Dive deeper into Cognee's powerful features and capabilities.
  </Card>
</CardGroup>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt