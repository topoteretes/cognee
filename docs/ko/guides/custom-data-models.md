# Custom Data Models

> Step-by-step guide to creating custom data models and using add_data_points

A minimal guide to creating custom data models and inserting them directly into the knowledge graph using `add_data_points`.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have some structured data you want to model

## What Custom Data Models Do

* Define your own Pydantic models that inherit from `DataPoint`
* Insert structured data directly into the knowledge graph without `cognify`
* Create relationships between data points programmatically
* Control exactly what gets indexed and how

## Code in Action

```python  theme={null}
import asyncio
from typing import Any
from pydantic import SkipValidation

import cognee
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.tasks.storage import add_data_points

class Person(DataPoint):
    name: str
    # Keep it simple for forward refs / mixed values
    knows: SkipValidation[Any] = None  # single Person or list[Person]
    # Recommended: specify which fields to index for search
    metadata: dict = {"index_fields": ["name"]}

async def main():
    # Start clean (optional in your app)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    alice = Person(name="Alice")
    bob = Person(name="Bob")
    charlie = Person(name="Charlie")

    # Create relationships - field name becomes edge label
    alice.knows = bob
    # You can also do lists: alice.knows = [bob, charlie]
    
    # Optional: add weights and custom relationship types
    bob.knows = (Edge(weight=0.9, relationship_type="friend_of"), charlie)

    await add_data_points([alice, bob, charlie])

asyncio.run(main())
```

<Note>
  This example shows the complete workflow with metadata for indexing and optional edge weights. In practice, you can create complex nested models with multiple relationships and sophisticated data structures.
</Note>

## What Just Happened

### Step 1: Define Your Data Model

```python  theme={null}
class Person(DataPoint):
    name: str
    knows: SkipValidation[Any] = None
    # Recommended: specify which fields to index for search
    metadata: dict = {"index_fields": ["name"]}
```

Create a Pydantic model that inherits from `DataPoint`. Use `SkipValidation[Any]` for fields that will hold other DataPoints to avoid forward reference issues. **Metadata is recommended** - it tells Cognee which fields to embed and store in the vector database for search.

### Step 2: Create Data Instances

```python  theme={null}
alice = Person(name="Alice")
bob = Person(name="Bob")
charlie = Person(name="Charlie")
```

Instantiate your models with the data you want to store. Each instance becomes a node in the knowledge graph.

### Step 3: Create Relationships

```python  theme={null}
alice.knows = bob
# Optional: add weights and custom relationship types
bob.knows = (Edge(weight=0.9, relationship_type="friend_of"), charlie)
```

Assign DataPoint instances to fields to create edges. The field name becomes the relationship label by default. **Weights are optional** - you can use `Edge` to add weights, custom relationship types, or other metadata to your relationships.

### Step 4: Insert into Graph

```python  theme={null}
await add_data_points([alice, bob, charlie])
```

This converts your DataPoint instances into nodes and edges in the knowledge graph, automatically handling the graph structure and indexing. The `name` field gets embedded and stored in the vector database for search.

## Use in Custom Tasks and Pipelines

This approach is particularly useful when creating custom tasks and pipelines where you need to:

* Insert structured data programmatically
* Define specific relationships between known entities
* Control exactly what gets indexed and how
* Integrate with external data sources or APIs

You can combine this with `cognify` to extract knowledge from unstructured text, then add your own structured data on top.

<Columns cols={3}>
  <Card title="Low-Level LLM" icon="cpu" href="/guides/low-level-llm">
    Learn about direct LLM interaction
  </Card>

  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt