# Datasets

> Project-level containers for organization, permissions, and processing

## What is a dataset in Cognee?

A dataset is a named container that groups documents and their metadata. It is the main boundary for:

* Organizing content
* Running pipelines
* Applying permissions

<Warning>
  **Dataset isolation** requires specific configuration. See [permissions system](../permissions-system/datasets#dataset-isolation) for details on access control requirements and supported database setups.
</Warning>

* **[Add](../main-operations/add)**:
  * Direct new content into a specific dataset (by name or ID)
  * If it doesn’t exist, Cognee creates it and associates your permissions
  * Items ingested are linked to that dataset and deduplicated within it

* **[Cognify](../main-operations/cognify)**:
  * Choose which dataset(s) to transform into a knowledge graph
  * Loads the dataset’s content, checks rights, and runs the pipeline per dataset
  * If none are specified, processes all datasets you’re authorized to use
  * Progress is tracked per dataset for reliable re-runs

* **[Search](../main-operations/search)**:
  * Queries can be scoped by dataset
  * Results and metrics remain separated by dataset

## Access control

* Permissions (read, write, share, delete) are enforced at the dataset level
* Share one dataset with a team, keep another private
* Independently manage who can modify or distribute content

## Incremental processing

* Processing status is tracked per dataset
* After you add more data, Cognify focuses on new or changed items
* Skips what’s already completed for that dataset

## Datasets vs NodeSets

**Datasets** scope storage, permissions, and pipeline execution; **[NodeSets](../further-concepts/node-sets)** are semantic tags within a dataset.

* During Add, you can label items with one or more NodeSet names (e.g., "AI", "FinTech")
* Cognify propagates those labels into the graph by creating `NodeSet` nodes and linking derived chunks and entities via `belongs_to_set` relationships
* This lets you slice a single dataset’s graph by topic or team without creating new datasets, while dataset-level permissions still control overall access

<Columns cols={3}>
  <Card title="Add" icon="plus" href="/core-concepts/main-operations/add">
    Direct content into a dataset
  </Card>

  <Card title="Cognify" icon="brain-cog" href="/core-concepts/main-operations/cognify">
    Run pipelines per dataset
  </Card>

  <Card title="Search" icon="search" href="/core-concepts/main-operations/search">
    Scope queries by dataset
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt