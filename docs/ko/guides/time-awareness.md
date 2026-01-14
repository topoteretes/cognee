# Temporal Cognify

> Step-by-step guide to using temporal mode for time-aware queries

A minimal guide to Cognee's temporal mode. If you already know the regular add → cognify → search flow, this adds one switch at cognify time and one search type for time-aware questions.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have data that contains dates/times (years or full dates)

## What Temporal Mode Does

* Builds events and timestamps from your text during cognify
* Lets you ask time-based questions like "before 1980", "after 2010", or "between 2000 and 2006"
* Uses `SearchType.TEMPORAL` to retrieve the most relevant events and answer with temporal context

## Step 1: Add Data

Add data with temporal information using the standard `add` function.

```python  theme={null}
import cognee

text = """
In 1998 the project launched. In 2001 version 1.0 shipped. In 2004 the team merged
with another group. In 2010 support for v1 ended.
"""

await cognee.add(text, dataset_name="timeline_demo")
```

<Note>
  This simple example uses one string that gets treated as a single document. In practice, you can add multiple documents, files, or entire datasets - the temporal processing works the same way across all your data.
</Note>

## Step 2: Cognify with Temporal Mode

Set `temporal_cognify=True` to extract events/timestamps instead of the default entity-graph pipeline.

```python  theme={null}
await cognee.cognify(datasets=["timeline_demo"], temporal_cognify=True)
```

<Info>
  Only datasets you pass (or all by default) are processed. Temporal mode runs an event/timestamp pipeline and stores temporal nodes in the graph.
</Info>

<Note>
  This example uses a single dataset for simplicity. In practice, you can process multiple datasets simultaneously by passing a list of dataset names, or omit the `datasets` parameter to process all available datasets.
</Note>

## Step 3: Ask Time-aware Questions

Use `SearchType.TEMPORAL` and phrase your query with time hints.

```python  theme={null}
from cognee.api.v1.search import SearchType

# Before / after queries
await cognee.search(
    query_type=SearchType.TEMPORAL, 
    query_text="What happened before 2000?", 
    top_k=10
)

await cognee.search(
    query_type=SearchType.TEMPORAL, 
    query_text="What happened after 2010?", 
    top_k=10
)

# Between queries
await cognee.search(
    query_type=SearchType.TEMPORAL, 
    query_text="Events between 2001 and 2004", 
    top_k=10
)

# Scoped descriptions
await cognee.search(
    query_type=SearchType.TEMPORAL, 
    query_text="Key project milestones between 1998 and 2010", 
    top_k=10
)
```

<Tip>
  * If the query has clear dates, the retriever filters events by time and ranks them
  * If no dates are detected, it falls back to event/entity graph retrieval and still answers
  * Increase `top_k` to inspect more candidate events
</Tip>

## Optional: Limit to Specific Datasets

```python  theme={null}
await cognee.search(
    query_type=SearchType.TEMPORAL,
    query_text="What happened after 2004?",
    datasets=["timeline_demo"],
    top_k=10,
)
```

## Using the HTTP API

If your server is running, you can run temporal search via the API by setting `search_type` to `"TEMPORAL"`:

```bash  theme={null}
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
  -d '{
        "search_type": "TEMPORAL",
        "query": "What happened between 2001 and 2004?",
        "top_k": 10
      }'
```

<Note>
  For now, enabling temporal processing at cognify time is easiest in Python with `temporal_cognify=True`.
</Note>

## Code in Action

Check `examples/python/temporal_example.py` for a complete script that:

* Adds two biographies (with dates)
* Runs `cognee.cognify(temporal_cognify=True)`
* Queries with `SearchType.TEMPORAL` for time-aware answers

<Columns cols={2}>
  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore temporal API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt