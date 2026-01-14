# Memify Quickstart

> Step-by-step guide to enriching existing knowledge graphs with derived facts

A minimal guide to running a small enrichment pass over your existing knowledge graph to add useful derived facts (e.g., coding rules) without re-ingesting data.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have an existing knowledge graph (add → cognify completed)

## What Memify Does

* Pulls a subgraph (or whole graph) into a mini-pipeline
* Applies enrichment tasks to create new nodes/edges from existing context
* Defaults: extracts relevant chunks and adds coding rule associations

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee import SearchType

async def main():
    # 1) Add two short chats and build a graph
    await cognee.add([
        "We follow PEP8. Add type hints and docstrings.",
        "Releases should not be on Friday. Susan must review PRs.",
    ], dataset_name="rules_demo")
    await cognee.cognify(datasets=["rules_demo"])  # builds graph

    # 2) Enrich the graph (uses default memify tasks)
    await cognee.memify(dataset="rules_demo")

    # 3) Query the new coding rules
    rules = await cognee.search(
        query_type=SearchType.CODING_RULES,
        query_text="List coding rules",
        node_name=["coding_agent_rules"],
    )
    print("Rules:", rules)

asyncio.run(main())
```

<Note>
  This simple example uses basic text data for demonstration. In practice, you can enrich large knowledge graphs with complex derived facts and associations.
</Note>

## What Just Happened

### Step 1: Build Your Knowledge Graph

```python  theme={null}
await cognee.add([
    "We follow PEP8. Add type hints and docstrings.",
    "Releases should not be on Friday. Susan must review PRs.",
], dataset_name="rules_demo")
await cognee.cognify(datasets=["rules_demo"])
```

First, create your knowledge graph using the standard add → cognify workflow. Memify works on existing graphs, so you need this foundation first.

### Step 2: Enrich with Memify

```python  theme={null}
await cognee.memify(dataset="rules_demo")
```

This runs the default memify tasks on your existing graph. No data parameter means it processes the existing graph, optionally filtering with `node_name` and `node_type`.

### Step 3: Query Enriched Data

```python  theme={null}
rules = await cognee.search(
    query_type=SearchType.CODING_RULES,
    query_text="List coding rules",
    node_name=["coding_agent_rules"],
)
```

Search for the newly created derived facts using specialized search types like `SearchType.CODING_RULES`.

## Customizing Tasks (Optional)

```python  theme={null}
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations

await cognee.memify(
    extraction_tasks=[Task(extract_subgraph_chunks)],
    enrichment_tasks=[Task(add_rule_associations, rules_nodeset_name="coding_agent_rules")],
    dataset="rules_demo",
)
```

You can customize the memify pipeline by specifying your own extraction and enrichment tasks.

## What Happens Under the Hood

The default memify tasks are equivalent to:

* **Extraction**: `Task(extract_subgraph_chunks)` - pulls relevant chunks from your graph
* **Enrichment**: `Task(add_rule_associations, rules_nodeset_name="coding_agent_rules")` - creates new associations and rules

This creates derived knowledge without re-processing your original data.

<Columns cols={3}>
  <Card title="Custom Data Models" icon="circle-stop" href="/guides/custom-data-models">
    Learn about custom data models
  </Card>

  <Card title="Custom Tasks" icon="workflow" href="/guides/custom-tasks-pipelines">
    Learn about custom tasks and pipelines
  </Card>

  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt