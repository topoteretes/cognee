# Code Graph

> Step-by-step guide to building code-level graphs from repositories

A minimal guide to building a code-level graph from a repository and searching it. The pipeline parses your repo, extracts code entities and dependencies, and optionally processes non-code docs alongside.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have a local repository path (absolute or relative)

## What Code Graph Does

* Scans a repo for supported languages and builds code nodes/edges (files, symbols, imports, call/dependency links)
* Optional: includes non-code files (markdown, docs) as a standard knowledge graph
* Enables `SearchType.CODE` for code-aware queries

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee import SearchType
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline

async def main():
    repo_path = "/path/to/your/repo"  # folder root

    # Build the code graph (code only)
    async for _ in run_code_graph_pipeline(repo_path, include_docs=False):
        pass

    # Ask a code question
    results = await cognee.search(query_type=SearchType.CODE, query_text="Where is Foo used?")
    print(results)

asyncio.run(main())
```

<Note>
  This simple example uses a basic repository for demonstration. In practice, you can process large codebases with multiple languages and complex dependency structures.
</Note>

## What Just Happened

### Step 1: Build the Code Graph

```python  theme={null}
async for _ in run_code_graph_pipeline(repo_path, include_docs=False):
    pass
```

This scans your repository for supported languages and builds code nodes/edges. The pipeline handles file parsing, symbol extraction, and dependency analysis automatically.

### Step 2: Search Your Code

```python  theme={null}
results = await cognee.search(query_type=SearchType.CODE, query_text="Where is Foo used?")
```

Use `SearchType.CODE` to ask code-aware questions about your repository. This searches through the extracted code structure, not just text content.

## Include Documentation (Optional)

Also process non-code files from the repo (slower, uses LLM for text):

```python  theme={null}
async for _ in run_code_graph_pipeline(repo_path, include_docs=True):
    pass
```

This processes markdown files, documentation, and other text files alongside your code, creating a comprehensive knowledge graph.

## Advanced Options

```python  theme={null}
async for _ in run_code_graph_pipeline(
    repo_path,
    include_docs=False,
    excluded_paths=["**/node_modules/**", "**/dist/**"],
    supported_languages=["python", "typescript"],
):
    pass
```

* **`excluded_paths`**: List of paths (globs) to skip, e.g., tests, build folders
* **`supported_languages`**: Narrow to certain languages to speed up processing

## Visualize Your Graph (Optional)

```python  theme={null}
from cognee.api.v1.visualize.visualize import visualize_graph
await visualize_graph("./graph_code.html")
```

Generate an HTML visualization of your code graph to explore the structure and relationships.

## What Happens Under the Hood

`run_code_graph_pipeline(...)` automatically handles:

* Repository scanning and file parsing
* Code entity extraction (functions, classes, imports, calls)
* Dependency analysis and relationship mapping
* Database initialization and setup
* Optional documentation processing with LLM

Once complete, your code graph is ready for search and analysis.

<Columns cols={3}>
  <Card title="Custom Tasks" icon="workflow" href="/guides/custom-tasks-pipelines">
    Learn about custom tasks and pipelines
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