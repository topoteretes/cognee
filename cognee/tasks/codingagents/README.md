# Coding Agents

## Overview

The coding agents module provides utilities for extracting developer coding rules and best practices from text and associating them with their original sources in Cognee’s knowledge graph. It uses LLM-powered structured extraction to identify rules from conversations, documentation, or commit messages.

> [!NOTE]
> This module runs **automatically** in `cognee.memify()` (the enrichment pipeline), but is **not** enabled by default in the standard `cognee.cognify()` pipeline.

**Pipeline Position:** ingestion → graph extraction → **coding rule association** → storage / indexing

## Components

### Functions

| Function | Description |
|----------|-------------|
| `add_rule_associations(data, rules_nodeset_name, ...)` | Extracts rules via LLM from the `data`, adds them to the graph, and creates source links.|
| `get_existing_rules(rules_nodeset_name)` | Retrieves existing rules from the graph for a specific nodeset |
| `get_origin_edges(data, rules)` | Searches for the original `DocumentChunk` that matches the input `data` and creates `rule_associated_from` edges linking the new `Rule` nodes to that source chunk. |

### Data Models (extend `DataPoint`)

#### `Rule`
Represents a single extracted developer rule.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The coding rule text content. |
| `belongs_to_set` | `NodeSet` | Reference to the parent `NodeSet` (e.g., "coding_agent_rules"). |
| `metadata` | `dict` | Indexing configuration (indexes `rule` field). |

#### `RuleSet`
A collection of rules extracted in a single pass.

| Field | Type | Description |
|-------|------|-------------|
| `rules` | `List[Rule]` | List of extracted `Rule` objects. |


## Usage

### Automatic (via Memify)

The `memify` pipeline includes rule associations by default.

```python
import cognee
from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules


await cognee.add(["agent.md"])# Add data (text or file paths)
await cognee.cognify() # Create Knowledge Graph
await cognee.memify()# Enrich Graph (Extract Rules automatically)

rules = await get_existing_rules("coding_agent_rules")
if rules:
    for rule in rules:
        print(f"{rule}")
```

### Manual Rule Association

You can run the task directly on specific data.

```python
from cognee.tasks.codingagents.coding_rule_associations import add_rule_associations

await add_rule_associations(
    data="Always use type hints in Python functions.",
    rules_nodeset_name="coding_agent_rules"
)
```

### Retrieval

```python
from cognee.tasks.codingagents.coding_rule_associations import get_existing_rules

rules = await get_existing_rules("coding_agent_rules")
for rule in rules:
    print(f"{rule}")
```

### Advanced: Manual Graph Construction

You can manually create `Rule` objects and link them to content using `get_origin_edges` if you want to bypass the LLM extraction.
```python
from cognee.tasks.codingagents.coding_rule_associations import Rule, get_origin_edges

# 1. Define your rule (the abstract guideline)
rule = Rule(text="Use snake_case for function names.")

# 2. Link it to the source (the text that implies the rule)
# 'data' is used to find the original document chunk in the graph
edges = await get_origin_edges(
    data="We strictly follow PEP8. Function names must use snake_case.",
    rules=[rule]
)
```

## Configuration

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for LLM provider (required) |
| `LLM_PROVIDER` | Provider name (default: openai) |
| `LLM_MODEL` | Model name |

## Dependencies

**Internal:** `cognee.infrastructure.databases.graph`, `cognee.infrastructure.databases.vector`, `cognee.infrastructure.llm`, `cognee.modules.engine.models`

**External:** `pydantic`, LLM provider

## Related

- [cognee docs](https://docs.cognee.ai) | [Ingestion](../ingestion/) | [Graph](../graph/)
