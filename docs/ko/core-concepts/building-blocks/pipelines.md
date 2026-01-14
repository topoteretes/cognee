# Pipelines

> Orchestrating tasks into coordinated workflows for data processing

## What pipelines are

Pipelines coordinate ordered [Tasks](../building-blocks/tasks) into a reproducible workflow. Default Cognee operations like [Add](../main-operations/add) and [Cognify](../main-operations/cognify) run on top of the same execution layer. You typically do not call low-level functions directly; you trigger pipelines through these operations.

## Prerequisites

* **Dataset**: a container (name or UUID) where your data is stored and processed. Every document added to cognee belongs to a dataset.
* **User**: the identity for ownership and access control. A default user is created and used if none is provided.
* More details are available below

## How pipelines run

Somewhat unsurprisingly, the function used to run pipelines is called `run_pipeline`.

Cognee uses a **layered execution model**: a single call to `run_pipeline` orchestrates **multi-dataset processing** by running **per-file pipelines** through the sequence of tasks.

* **Statuses** are yielded as the pipeline runs and written to **databases** where appropriate
* **User access** to datasets and files is carefully verified at each layer
* **Pipeline run information** includes dataset IDs, completion status, and error handling
* **Background execution** uses queues to manage status updates and avoid database conflicts

<Accordion title="Layered execution">
  - Innermost layer: individual task execution with telemetry and recursive task running in batches
  - Middle layer: per-dataset pipeline management and task orchestration
  - Outermost layer: multi-dataset orchestration and overall pipeline execution
  - Execution modes: blocking (wait for completion) or background (return immediately with "started" status)
</Accordion>

<Accordion title="Customization approaches and tips">
  * Use [Cognify](../main-operations/cognify) with custom tasks after [Add](../main-operations/add)
  * Modify transformation steps without touching low-level functions, avoid going below `run_pipeline`
  * Custom tasks let you extend or replace default behavior
</Accordion>

<Accordion title="Users">
  * Identity: represents who owns and acts on data. If omitted, a default user is used
  * Ownership: every ingested item is tied to a user; content is deduplicated per owner
  * Permissions: enforced per dataset (read/write/delete/share) during processing and API access
</Accordion>

<Accordion title="Datasets">
  * Container: a named or UUID-scoped collection of related data and derived knowledge
  * Scoping: Add writes into a specific dataset; Cognify processes the dataset(s) you pass
  * Lifecycle: new names create datasets and grant the calling user permissions; UUIDs let you target existing datasets (given permission)
</Accordion>

<Columns cols={3}>
  <Card title="Tasks" icon="square-check" href="/core-concepts/building-blocks/tasks">
    Learn about the individual processing units that make up pipelines
  </Card>

  <Card title="DataPoints" icon="circle" href="/core-concepts/building-blocks/datapoints">
    Understand the structured outputs that pipelines produce
  </Card>

  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    See how pipelines are used in Add, Cognify, and Search workflows
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt