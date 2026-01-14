# Distributed Execution

> Step-by-step guide to running Cognee pipelines across Modal containers

A minimal guide to running Cognee pipelines across [Modal](https://modal.com/docs) containers with a one-line toggle. Good fit for large batches or slow tasks.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have a Modal account and tokens configured locally (`modal setup`)
* Create a Modal Secret named `distributed_cognee` with your environment variables

## What Distributed Execution Does

* Distributes per-item task execution to Modal functions
* Keeps your code unchanged; you can keep using `add` → `cognify` → `search` or custom pipelines
* Scales processing across multiple containers for large datasets

## What is Modal?

[Modal](https://modal.com/docs) is a serverless cloud platform that provides compute-intensive applications without thinking about infrastructure. It's perfect for running generative AI models, large-scale batch workflows, and job queues at scale.

When you enable distributed execution, Cognee automatically uses Modal to run your processing tasks across multiple containers, making it much faster for large datasets.

## Prerequisites

Install extras with Modal support and configure your environment:

```bash  theme={null}
# Install with distributed support
pip install cognee[distributed]

# Configure Modal (creates account if needed)
modal setup

# Create Modal Secret with your environment variables
modal secret create distributed_cognee
```

Add your environment variables to the Modal Secret (e.g., `LLM_API_KEY`, DB configs, S3 creds if used).

## Code in Action

```python  theme={null}
import asyncio
import cognee
from cognee import SearchType

async def main():
    # COGNEE_DISTRIBUTED=true is picked up implicitly
    # 1) Add data (text, files, or S3 URIs)
    await cognee.add([
        "Alice knows Bob. Bob works at ACME.",
        "NLP is a subfield of computer science.",
    ], dataset_name="dist_demo")

    # 2) Build the knowledge graph (runs distributed)
    await cognee.cognify(datasets=["dist_demo"]) 

    # 3) Query
    answers = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="Who does Alice know?",
        top_k=5,
    )
    print(answers)

asyncio.run(main())
```

<Note>
  This simple example uses basic text data for demonstration. In practice, you can process large datasets, files, or S3 URIs - the distributed execution scales automatically across Modal containers.
</Note>

## What Just Happened

### Step 1: Enable Distribution

```bash  theme={null}
export COGNEE_DISTRIBUTED=true
python your_script.py
```

Set the environment variable and run your code as usual. Internally, pipelines switch from `run_tasks` to `run_tasks_distributed` (Modal) via this toggle.

### Step 2: Add Your Data

```python  theme={null}
await cognee.add([
    "Alice knows Bob. Bob works at ACME.",
    "NLP is a subfield of computer science.",
], dataset_name="dist_demo")
```

Add your data using the standard `add` function. The same approach works with files, S3 URIs, or large datasets.

### Step 3: Process Distributed

```python  theme={null}
await cognee.cognify(datasets=["dist_demo"])
```

The `cognify` operation automatically runs distributed across Modal containers when `COGNEE_DISTRIBUTED=true` is set.

### Step 4: Search Your Data

```python  theme={null}
answers = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="Who does Alice know?",
    top_k=5,
)
```

Search your processed data using the standard search methods. The results are the same whether processed locally or distributed.

## What Happens Under the Hood

When `COGNEE_DISTRIBUTED=true`:

* Tasks are distributed to Modal functions automatically
* Each task runs in its own container
* Results are collected and merged back
* Database schemas are created on first run
* Costs are tracked in your Modal workspace

<Note>
  Start small and confirm costs in your Modal workspace. For non-pipeline first calls that write to DBs, call `await setup()` once.
</Note>

<Columns cols={3}>
  <Card title="Deploy REST API" icon="server" href="/guides/deploy-rest-api-server">
    Learn about API deployment
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