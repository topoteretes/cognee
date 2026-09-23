---
name: cognee-custom-graph-models
description: Use when defining the shape of cognee's knowledge graph with graph_model= — writing DataPoint node classes, choosing identity and index fields so nodes merge and are searchable, declaring typed Edge fields and FromIdentity references, building a model from a JSON schema, or debugging duplicated nodes, missing edges, or InvalidReferenceTypeError.
---

# Custom graph models

By default cognee extracts a generic `KnowledgeGraph` of entities and
relationships. Pass your own model with `graph_model=` and the LLM fills
your node and edge types instead.

```python
from typing import Annotated, Literal
import cognee
from cognee.low_level import DataPoint, Edge, FromIdentity

class Role(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

class Person(DataPoint):
    name: str
    is_a: Annotated[Role, FromIdentity()] | None = None     # reference by name
    reports_to: list[Edge["Person", "Person"]] = []          # edge owned by Person
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

class PeopleGraph(DataPoint):                                # the root the LLM fills
    people: list[Person]
    friends_with: list[Edge[Person, Person]] = []
    family: list[Edge[Person, Person, Literal["married_to", "sibling_of"]]] = []
    metadata: dict = {"index_fields": [], "transparent": True}

await cognee.remember(text, graph_model=PeopleGraph, custom_prompt="Extract every person...")
```

Full example: `examples/guides/custom_graph_model.py`.

## Use it

### Nodes: DataPoint classes

Every node type subclasses `DataPoint` (`from cognee.low_level import
DataPoint`). Its fields become:

- **Node properties:** scalars, strings, dicts, and anything that is not a
  DataPoint.
- **Edges named after the field:** a field holding a DataPoint or a list of
  DataPoints. `members: list[Person]` becomes `members` edges.

`dict[str, Person]`, sets and plain tuples are stored as properties, not
edges.

### Identity and search: `metadata`

| Key | What it does |
|---|---|
| `identity_fields` | The node id is derived from these field values (normalized: lowercased, spaces to `_`). The same entity from two chunks or two runs becomes **one node**. |
| `index_fields` | Each field gets a vector collection named `<ClassName>_<field>`, so recall can find the node. |
| `transparent` | The node is not stored; its children take its place. Use it for a root container like `PeopleGraph`. |

**Without `identity_fields` every node gets a random id, so the same person
is duplicated in every chunk and every run.** Set it on every node type that
represents a real-world entity.

**Write `metadata` explicitly**, as in the examples above. There is also an
annotation shortcut (`from cognee.infrastructure.engine import Dedup,
Embeddable`; `name: Annotated[str, Embeddable(), Dedup()]`), but today only
half of it works:

- `Dedup()` works: ids are derived from the marked fields.
- `Embeddable()` does not index. The markers update the class-level
  default, but each instance still carries `{"index_fields": []}`, and
  indexing reads the instance, so no vector collection is created and
  recall cannot find the node.

Markers are also ignored entirely when the class declares `metadata`
itself.

### Typed edges: `list[Edge[Source, Target, Name]]`

The LLM answers edges as flat rows of identity strings (`source`, `target`),
and cognee resolves them to the extracted nodes. The third parameter
controls the relationship name:

| Declaration | Relationship name |
|---|---|
| `list[Edge[Person, Person]]` | The field name (`friends_with`) |
| `list[Edge[Person, Person, Literal["a", "b"]]]` | The LLM picks one value |
| `list[Edge[Person, Person, str]]` | Free-form from the LLM, normalized |

- **Where to declare:** on the root model for relationships with no obvious
  owner, or on the owning node. On the owner, endpoints of the owner's own
  type must be strings (`Edge["Person", "Person"]`), because the class is
  not defined yet inside its own body.
- **Always a list:** `Edge[...]` or `Edge[...] | None` on its own raises.
- **Both endpoint types need exactly one `identity_fields` entry.**

### References: `Annotated[Target, FromIdentity()]`

Instead of a nested object, the LLM answers the identity string of a node
(`is_a: "engineer"`), and cognee links to that node. Supported spellings:
`Target`, `Target | None`, `list[Target]`, `list[Target] | None`.
Anything else raises `InvalidReferenceTypeError`. The target needs exactly
one identity field, and its other required fields need defaults.

### Edge values you build by hand

`Edge(source=..., target=..., relationship_type=..., weight=...,
properties={...})`. An omitted `source` falls back to the node declaring the
field; on a parametrized field that node must be the declared `Source`
type, or it raises. On a root container, always pass `source=`. The tuple
form `(Edge(weight=0.8), target_node)` attaches edge properties to a plain
DataPoint field (`examples/guides/custom_data_models.py`).

### From JSON instead of Python

- `cognee.low_level.graph_model_from_spec(spec)`: a small entity/relation
  spec (names, fields, `one`/`many` relations) compiled to DataPoint
  classes, with identity and index on `name` by default. Example:
  `examples/guides/graph_model_from_json.py`.
- `cognee.low_level.graph_schema_to_graph_model(json_schema)`: a JSON
  Schema (needs a top-level `title`; only internal `#` refs).
- HTTP: `POST /api/v1/remember` takes a `graph_model` form field (JSON
  schema string); `POST /api/v1/cognify` takes a `graphModel` JSON object.
  `POST /api/v1/llm/infer-schema` proposes a schema from sample text.

Neither JSON path can express typed `Edge` fields or `FromIdentity`; use
Python classes for those.

## Pitfalls

- **Duplicated nodes** → missing `identity_fields`.
- **Node never shows up in recall** → no `index_fields`, or the recalling
  process never imported the model class (graph completion searches the
  collections of DataPoint classes loaded in that process).
- **Edge rows silently missing** → an unresolved row is dropped with a
  warning. Endpoints resolve by **exact type** against nodes extracted from
  the same chunk: a subclass instance does not match where its base is
  declared, and a node from another chunk or an earlier run is not a
  candidate.
- **`InvalidReferenceTypeError`, "declares Edge in a shape the LLM
  extraction cannot fill"** → an unsupported `FromIdentity` or `Edge`
  spelling. These are raised when the model is converted during
  extraction, not at class definition, so they appear mid-pipeline.
- **String endpoints fail for classes defined inside a function.** Define
  models at module level.
- **Field names that collide with DataPoint's own fields** (`id`, `type`,
  `version`, `metadata`, `created_at`, `belongs_to_set`, …) are stripped
  from what the LLM sees. Rename them.
- **A subclass that overrides `metadata`** replaces the parent's entirely.
  Dropping `identity_fields` only logs a warning. A subclass also hashes ids
  under its own class name.
- **Write a `custom_prompt`.** Without one the generic knowledge-graph
  prompt is used; your schema reaches the LLM only as structured output.
- **Not with GLiNER.** `extractor="gliner"` raises with a custom
  `graph_model`.
- **Remote mode drops it.** After `cognee.serve(url)`, `remember()` and
  `cognify()` do not forward `graph_model`; the server builds a generic
  graph.
- A custom model skips the generic path's ontology resolution, per-graph
  node dedup, and `functional_relationships`. Summaries still run.

## How it works

`extract_content_graph` converts a DataPoint model into a plain Pydantic
model for the LLM: infrastructure fields and `metadata` are stripped, typed
edge fields become row lists (`FriendsWithEdge` with `source`/`target`
strings), and `FromIdentity` fields become strings. The answer is converted
back into DataPoint instances with ids from `identity_fields`, edge rows are
resolved against the nodes in that answer, and the result is attached to the
chunk (`chunk.contains`) and stored by `add_data_points`. Ownership is
recorded per document, so `forget(data_id=...)` removes a custom-model
document's nodes while shared nodes survive.

- LLM boundary, both directions: `cognee/shared/llm_graph_model.py`
- DataPoint, metadata, ids: `cognee/infrastructure/engine/models/DataPoint.py`
- Markers: `cognee/infrastructure/engine/models/FieldAnnotations.py`
- Edge: `cognee/infrastructure/engine/models/Edge.py`
- Property vs edge decision: `cognee/modules/graph/utils/field_edges.py`
- Custom-model branch of extraction: `cognee/tasks/graph/extract_graph_from_data.py`
- JSON schema / spec paths: `cognee/shared/graph_model_utils.py`,
  `cognee/modules/graph_models/`
- Vector collections: `cognee/tasks/storage/index_data_points.py`

## Extending it

- Tests for the LLM round trip: `cognee/tests/unit/modules/graph/test_content_graph_to_data_point.py`.
  Edge typing: `cognee/tests/unit/interfaces/graph/test_typed_edge_model.py`,
  `test_typed_edges_graph.py`. Identity: `cognee/tests/unit/infrastructure/engine/test_identity_fields.py`.
  Property vs edge: `cognee/tests/unit/modules/graph/test_field_edges.py`.
- Deletion of custom-model nodes: `cognee/tests/test_delete_custom_graph.py`.
- A new `Edge` or `FromIdentity` spelling must be handled in both directions
  in `llm_graph_model.py`, and rejected with `InvalidReferenceTypeError`
  when unsupported, never silently accepted.
