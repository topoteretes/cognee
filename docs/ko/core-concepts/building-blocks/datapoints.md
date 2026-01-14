# DataPoints

> Atomic units of knowledge in Cognee

# DataPoints: Atomic Units of Knowledge

DataPoints are the smallest building blocks in Cognee.\
They represent **atomic units of knowledge** — carrying both your actual content and the context needed to process, index, and connect it.

They're the reason Cognee can turn raw documents into something that's both **searchable** (via vectors) and **connected** (via graphs).

## What are DataPoints

* **Atomic** — each DataPoint represents one concept or unit of information.
* **Structured** — implemented as [Pydantic](https://docs.pydantic.dev/) models for validation and serialization.
* **Contextual** — carry provenance, versioning, and indexing hints so every step downstream knows where data came from and how to use it.

## Core Structure

A DataPoint is just a Pydantic model with a set of standard fields.

<Accordion title="See example class definition">
  ```python  theme={null}
  class DataPoint(BaseModel):
      id: UUID = Field(default_factory=uuid4)
      created_at: int = ...
      updated_at: int = ...
      version: int = 1
      topological_rank: Optional[int] = 0
      metadata: Optional[dict] = {"index_fields": []}
      type: str = "DataPoint"
      belongs_to_set: Optional[List["DataPoint"]] = None
  ```

  Key fields:

  * `id` — unique identifier
  * `created_at`, `updated_at` — timestamps (ms since epoch)
  * `version` — for tracking changes and schema evolution
  * `metadata.index_fields` — critical: determines which fields are embedded for vector search
  * `type` — class name
  * `belongs_to_set` — groups related DataPoints
</Accordion>

## Indexing & Embeddings

The `metadata.index_fields` tells Cognee which fields to embed into the vector store.
This is the mechanism behind semantic search.

* Fields in `index_fields` → converted into embeddings
* Each indexed field → its own vector collection (`Class_field`)
* Non-indexed fields → stay as regular properties
* Choosing what to index controls search granularity

## From DataPoints to the Graph

When you call `add_data_points()`, Cognee automatically:

* Embeds the indexed fields into vectors
* Converts the object into **nodes** and **edges** in the knowledge graph
* Stores provenance in the relational store

This is how Cognee creates both **semantic similarity** (vector) and **structural reasoning** (graph) from the same unit.

## Examples and details

<Accordion title="Example: indexing only one field">
  ```python  theme={null}
  class Person(DataPoint):
      name: str
      age: int
      metadata: dict = {"index_fields": ["name"]}
  ```

  Only `"name"` is semantically searchable
</Accordion>

<Accordion title="Example: Book → Author transformation">
  ```python  theme={null}
  class Book(DataPoint):
      title: str
      author: Author
      metadata: dict = {"index_fields": ["title"]}

  # Produces:
  # `Node(Book)` with `{title, type, ...}`
  # Node(Author) with {name, type, ...}
  # Edge(Book → Author, type="author")
  ```
</Accordion>

<Accordion title="Relationship syntax options">
  ```python  theme={null}
  # Simple relationship
  `author: Author`  

  # With edge metadata
  `has_items: (Edge(weight=0.8), list[Item])`

  # List relationship
  `chapters: list[Chapter]`
  ```
</Accordion>

<Accordion title="Built-in DataPoint types">
  Cognee ships with several built-in DataPoint types:

  * **Documents** — wrappers for source files (Text, PDF, Audio, Image)
    * `Document` (`metadata.index_fields=["name"]`)
  * **Chunks** — segmented portions of documents
    * `DocumentChunk` (`metadata.index_fields=["text"]`)
  * **Summaries** — generated text or code summaries
    * `TextSummary` / `CodeSummary` (`metadata.index_fields=["text"]`)
  * **Entities** — named objects (people, places, concepts)
    * `Entity`, `EntityType` (`metadata.index_fields=["name"]`)
  * **Edges** — relationships between DataPoints
    * `Edge` — links between DataPoints
</Accordion>

<Accordion title="Example: custom DataPoint with best practices">
  ```python  theme={null}
  class Product(DataPoint):
      name: str
      description: str
      price: float
      category: Category
      
      # Index name + description for search
      metadata: dict = {"index_fields": ["name", "description"]}
  ```

  **Best Practices:**

  * **Keep it small** — one concept per DataPoint
  * **Index carefully** — only fields that matter for semantic search
  * **Use built-in types first** — extend with custom subclasses when needed
  * **Version deliberately** — track changes with `version`
  * **Group related points** — with `belongs_to_set`
</Accordion>

<Columns cols={3}>
  <Card title="Tasks" icon="square-check" href="/core-concepts/building-blocks/tasks">
    Learn how DataPoints are created and processed
  </Card>

  <Card title="Pipelines" icon="git-merge" href="/core-concepts/building-blocks/pipelines">
    See how DataPoints flow through processing workflows
  </Card>

  <Card title="Main Operations" icon="play" href="/core-concepts/main-operations/add">
    Understand how DataPoints are used in Add, Cognify, and Search
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt