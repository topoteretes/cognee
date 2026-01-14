# Custom Prompts

> Step-by-step guide to using custom prompts to control graph extraction

A minimal guide to shaping graph extraction with a custom LLM prompt. You'll pass your prompt via `custom_prompt` to `cognee.cognify()` to control entity types, relationship labels, and extraction rules.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have some text or files to process

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee.api.v1.search import SearchType

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""

async def main():
    await cognee.add([
        "Alice moved to Paris in 2010, while Bob has always lived in New York.",
        "Andreas was born in Venice, but later settled in Lisbon.",
        "Diana and Tom were born and raised in Helsingy. Diana currently resides in Berlin, while Tom never moved."
    ])
    await cognee.cognify(custom_prompt=custom_prompt)

    res = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Where does Alice live?",
    )
    print(res)

if __name__ == "__main__":
    asyncio.run(main())
```

<Note>
  This simple example uses a few strings for demonstration. In practice, you can add multiple documents, files, or entire datasets - the custom prompt processing works the same way across all your data.
</Note>

## What Just Happened

### Step 1: Add Your Data

```python  theme={null}
await cognee.add([
    "Alice moved to Paris in 2010, while Bob has always lived in New York.",
    "Andreas was born in Venice, but later settled in Lisbon.",
    "Diana and Tom were born and raised in Helsingy. Diana currently resides in Berlin, while Tom never moved."
])
```

This adds text data to Cognee using the standard `add` function. The same approach works with multiple documents, files, or entire datasets.

### Step 2: Write a Custom Prompt

```python  theme={null}
custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""
```

The custom prompt overrides the default system prompt used during entity/relationship extraction. It constrains node types, enforces relationship naming, and reduces noise.

<Note>
  `custom_prompt` is ignored when `temporal_cognify=True`.
</Note>

### Step 3: Cognify with Your Custom Prompt

```python  theme={null}
await cognee.cognify(custom_prompt=custom_prompt)
```

This processes your data using the custom prompt to control extraction behavior. You can also scope to specific datasets by passing the `datasets` parameter.

### Step 4: Ask Questions

```python  theme={null}
res = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="Where does Alice live?",
)
```

Use `SearchType.GRAPH_COMPLETION` to get answers that leverage your custom extraction rules.

<Columns cols={3}>
  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="Ontology Quickstart" icon="git-branch" href="/guides/ontology-support">
    Learn about ontology integration
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt