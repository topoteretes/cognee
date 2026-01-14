# Custom Tasks and Pipelines

> Step-by-step guide to creating custom tasks and pipelines

A minimal guide to creating custom tasks and pipelines. You'll build a two-step pipeline: the LLM extracts People directly, then you insert them into the knowledge graph.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have some text data to process

## What Custom Tasks and Pipelines Do

* Define custom processing steps using `Task` objects
* Chain multiple operations together in a pipeline
* Use LLMs to extract structured data from text
* Insert structured data directly into the knowledge graph
* Control the entire data processing workflow

## Code in Action

```python  theme={null}
import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, SkipValidation

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points
from cognee.modules.pipelines import Task, run_pipeline

class Person(DataPoint):
    name: str
    # Optional relationships (we'll let the LLM populate this)
    knows: List["Person"] = []
    # Make names searchable in the vector store
    metadata: Dict[str, Any] = {"index_fields": ["name"]}

class People(BaseModel):
    persons: List[Person]

async def extract_people(text: str) -> List[Person]:
    system_prompt = (
        "Extract people mentioned in the text. "
        "Return as `persons: Person[]` with each Person having `name` and optional `knows` relations. "
        "If the text says someone knows someone set `knows` accordingly. "
        "Only include facts explicitly stated."
    )
    people = await LLMGateway.acreate_structured_output(text, system_prompt, People)
    return people.persons

async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    text = "Alice knows Bob."

    tasks = [
        Task(extract_people),  # input: text -> output: list[Person]
        Task(add_data_points)  # input: list[Person] -> output: list[Person]
    ]

    async for _ in run_pipeline(tasks=tasks, data=text, datasets=["people_demo"]):
        pass



if __name__ == "__main__":
    asyncio.run(main())
```

<Note>
  This simple example uses a two-step pipeline for demonstration. In practice, you can create complex pipelines with multiple custom tasks, data transformations, and processing steps.
</Note>

## What Just Happened

### Step 1: Define Your Data Models

```python  theme={null}
class Person(DataPoint):
    name: str
    knows: SkipValidation[Any] = None
    metadata: Dict[str, Any] = {"index_fields": ["name"]}

class People(BaseModel):
    persons: List[Person]
```

Create Pydantic models for your data. `Person` inherits from `DataPoint` for graph insertion, while `People` is a simple container for the LLM output. **Metadata is recommended** to make fields searchable in the vector database.

### Step 2: Create Your Custom Task

```python  theme={null}
async def extract_people(text: str) -> List[Person]:
    system_prompt = (
        "Extract people mentioned in the text. "
        "Return as `persons: Person[]` with each Person having `name` and optional `knows` relations. "
        "If the text says someone knows someone set `knows` accordingly. "
        "Only include facts explicitly stated."
    )
    people = await LLMGateway.acreate_structured_output(text, system_prompt, People)
    return people.persons
```

This task uses the LLM to extract structured data from text. The LLM fills `People` objects with `Person` instances, including relationships via the `knows` field.

<Tip>
  `acreate_structured_output` is backend-agnostic (BAML or LiteLLM+Instructor). Configure via `STRUCTURED_OUTPUT_FRAMEWORK` in `.env`.
</Tip>

### Step 3: Build Your Pipeline

```python  theme={null}
tasks = [
    Task(extract_people),  # input: text -> output: list[Person]
    Task(add_data_points)  # input: list[Person] -> output: list[Person]
]

async for _ in run_pipeline(tasks=tasks, data=text, datasets=["people_demo"]):
    pass
```

Chain your tasks together in a pipeline. The first task extracts people from text, the second inserts them into the knowledge graph. `add_data_points` automatically creates nodes and edges from the `knows` relationships.

Under the hood, `run_pipeline(...)` automatically initializes databases and checks LLM/embeddings configuration, so you don't need to worry about setup. Once the pipeline completes, your Cognee memory with graph and embeddings is created and ready for interaction.

You can now search your data using the standard search methods:

```python  theme={null}
from cognee.api.v1.search import SearchType

# Search the processed data
results = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="Who does Alice know?",
    datasets=["people_demo"]
)
print(results)
```

## Use Cases

This approach is particularly useful when you need to:

* Extract structured data from unstructured text
* Process data through multiple custom steps
* Control the entire data processing workflow
* Combine LLM extraction with programmatic data insertion
* Build complex data processing pipelines

<Columns cols={3}>
  <Card title="Custom Data Models" icon="circle-stop" href="/guides/custom-data-models">
    Learn about custom data models
  </Card>

  <Card title="Low-Level LLM" icon="cpu" href="/guides/low-level-llm">
    Learn about direct LLM interaction
  </Card>

  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt