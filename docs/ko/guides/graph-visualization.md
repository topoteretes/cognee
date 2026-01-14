# Graph Visualization

> Step-by-step guide to rendering interactive knowledge graphs

A minimal guide to rendering your current knowledge graph to an interactive HTML file with one call.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Have some data processed with `cognify` (knowledge graph exists)

## What Graph Visualization Shows

* Nodes (entities, types, chunks, summaries) with color coding
* Edges with labels and weights; tooltips show extra edge properties
* Interactive features: drag nodes, zoom/pan, hover edges for details

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee.api.v1.visualize.visualize import visualize_graph

async def main():
    await cognee.add(["Alice knows Bob.", "NLP is a subfield of CS."])
    await cognee.cognify()

    await visualize_graph("./graph_after_cognify.html")

asyncio.run(main())
```

<Note>
  This simple example uses basic text data for demonstration. In practice, you can visualize complex knowledge graphs with thousands of nodes and relationships.
</Note>

## What Just Happened

### Step 1: Create Your Knowledge Graph

```python  theme={null}
await cognee.add(["Alice knows Bob.", "NLP is a subfield of CS."])
await cognee.cognify()
```

First, create your knowledge graph using the standard add → cognify workflow. The visualization works on existing graphs.

### Step 2: Generate Visualization

```python  theme={null}
await visualize_graph("./graph_after_cognify.html")
```

This creates an interactive HTML file with your knowledge graph. You can specify a custom path or use the default location.

## Quick Options

### Default Location

```python  theme={null}
from cognee.api.v1.visualize.visualize import visualize_graph

# Writes HTML to your home directory by default
await visualize_graph()
```

### Custom Path

```python  theme={null}
from cognee.api.v1.visualize.visualize import visualize_graph

# Writes to the provided file path (created/overwritten)
await visualize_graph("./my_graph.html")
```

## Tips

* **Large graphs**: Rendering a very big graph can be slow. Consider building subsets (e.g., smaller datasets) before visualizing
* **Edge weights**: If present, control line thickness; multiple weights are summarized and shown in tooltips
* **Static HTML**: Files are static HTML; you can open them in any modern browser or share them as artifacts

<Columns cols={3}>
  <Card title="Code Graph" icon="code" href="/guides/code-graph">
    Learn about code graph visualization
  </Card>

  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="Custom Data Models" icon="circle-stop" href="/guides/custom-data-models">
    Learn about custom data models
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt