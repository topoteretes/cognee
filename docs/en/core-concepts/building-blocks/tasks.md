# Tasks

> Building blocks of processing that transform data in Cognee pipelines

# Tasks: Smallest Executable Units

Tasks are Cognee's **smallest executable units** — they wrap any Python callable and give it a uniform interface for batching, error handling, and logging. While they can work with anything, Tasks are most powerful when creating or enriching [DataPoints](../building-blocks/datapoints).

## What are Tasks

Tasks are Cognee's **smallest executable units**.

* They wrap any Python callable (function, coroutine, generator, async generator).
* Give a **uniform interface** for batching, error handling, and logging.
* Can work with anything, but are **most powerful when creating or enriching [DataPoints](../building-blocks/datapoints)**.

## Why Tasks Exist

* Normalize different kinds of Python functions so they behave consistently.
* Enable **stream-based processing**: outputs flow directly into the next step.
* Provide **batching controls** for efficiency, especially with LLM or I/O-heavy operations.
* Form the **building blocks** of higher-level [Pipelines](../building-blocks/pipelines).

## Core Concepts

* **Execution**: run functions in a consistent way, regardless of sync/async/gen.
* **Batching**: configurable with `task_config`.
* **Composition**: Tasks can be chained — one Task's output is the next Task's input.
* **Flexibility**: Tasks don't need to handle DataPoints, but Cognee's defaults encourage it.

## Dependencies & Ordering

Tasks often assume a certain **input type** and produce an expected **output type**.
Example flow (educational, not exhaustive):

* Raw data → Documents
* Documents → Chunks
* Chunks → Entities and relationships
* Entities/Chunks → Summaries
* Any DataPoint → Storage

## Built-in Tasks

* **Ingestion**: `resolve_data_directories`, `ingest_data`
* **Classification**: `classify_documents`
* **Access control**: `check_permissions_on_dataset`
* **Chunking**: `extract_chunks_from_documents`
* **Graph extraction**: `extract_graph_from_data`
* **Summarization**: `summarize_text`, `summarize_code`
* **Persistence**: `add_data_points`

## Examples and details

<Accordion title="Task API & Constructor">
  ```python  theme={null}
  Task(executable, *args, task_config={...}, **kwargs)
  ```

  **Key parameters:**

  * `executable`: Any Python callable (function, coroutine, generator, async generator)
  * `task_config`: Configuration for batching, error handling, and logging
  * `default_params`: Parameters that are always passed to the executable
</Accordion>

<Accordion title="Supported Task Types">
  Cognee automatically detects and handles different Python function types:

  * **Functions**: Standard synchronous functions
  * **Coroutines**: Async functions using `async def`
  * **Generators**: Functions that yield multiple values
  * **Async Generators**: Async functions that yield multiple values

  Each type is executed appropriately within Cognee's task system.
</Accordion>

<Accordion title="Writing a Custom Task">
  ```python  theme={null}
  def my_custom_task(data_chunk):
      # Process the data chunk
      processed_data = process_chunk(data_chunk)
      
      # Create or enrich DataPoints
      datapoint = DataPoint(
          content=processed_data,
          metadata={"source": "custom_task"}
      )
      
      return datapoint

  # Wrap it in a Task
  my_task = Task(my_custom_task)
  ```

  **Why idempotent, DataPoint-focused functions are easiest to compose:**

  * Predictable inputs and outputs
  * Easy to chain together
  * Clear data flow between steps
</Accordion>

<Accordion title="Execution Flow">
  Tasks execute in sequence within [Pipelines](../building-blocks/pipelines), with each Task's output becoming the next Task's input. This creates a data transformation pipeline that builds up to the final knowledge graph.
</Accordion>

<Columns cols={3}>
  <Card title="DataPoints" icon="circle" href="/core-concepts/building-blocks/datapoints">
    The structured units that Tasks create and process
  </Card>

  <Card title="Pipelines" icon="git-merge" href="/core-concepts/building-blocks/pipelines">
    How Tasks are orchestrated into workflows
  </Card>

  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    See Tasks in action during data ingestion and processing
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt