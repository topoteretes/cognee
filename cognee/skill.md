---
name: cognee
description: >
  Use this skill whenever the user asks about Cognee, AI memory, persistent agent memory,
  self-improving agents, agents learning from feednack, knowledge graphs, graph-based RAG,
  long-term memory for agents, short-term memory for agents, personalization, personas,
  temporal search,  temporal knowledge graphs, ontology-based extraction, ontology grounding,
  feedback, Cypher search, natural-language graph search, chunk search, RAG search, cross-session memory,
  session feedback, feedback loops, session based memory, redis based memory, knowledge promotion.
  Also use when the user describes the workflow such as:
  "turn documents into a knowledge graph", "build memory from files", "search my graph",
  "extract entities and relations", "sync data into a graph", "update graph memory",
  "store memories for an agent", "help my agent learn over time", "visualize a knowledge
  graph built from documents", "let the agent learn", "adaptive agents", "personalized agents",
  "session based personalization", "find important ontologies", "find custom pydantic models",
  "isolate agentic behaviour", "add permission control to retrieval", "reduce context bloating".
---

# Cognee

Use this skill for **Cognee-specific Python API help** and for mapping user goals to the right Cognee workflow.

## When to apply this skill

Apply this skill whenever the user wants to do any of the following with Cognee:

- ingest text, files, URLs, repos, or datasets
- build or rebuild a knowledge graph
- search documents, chunks, summaries, triplets, or graph context
- choose a `SearchType`
- enrich an existing graph with `memify`
- define custom graph extraction models or `DataPoint` types
- run custom task pipelines
- configure LLM, graph DB, vector DB, or storage settings
- tag and scope memory with `node_set` / NodeSets
- build persistent memory for agents across sessions
- create feedback loops or self-improving agent workflows
- work with temporal extraction, ontologies, Cypher, or natural-language graph queries
- manage datasets, sessions, feedback, pruning, updates, or visualization

If the user’s intent is “store information in memory and query it later,” prefer Cognee’s core flow:
**add -> cognify -> search**

## Core workflow

```python
import cognee
from cognee import SearchType

await cognee.add(
    "Your text, file path, URL, or list of inputs",
    dataset_name="main",
    node_set=["default_memory"],
)
await cognee.cognify(datasets="main")
results = await cognee.search(
    "What are the key insights?",
    query_type=SearchType.GRAPH_COMPLETION,
    datasets="main",
)
```

## Default guidance

When helping with Cognee:

1. Start with the **simplest working path** unless the user explicitly asks for advanced configuration.
2. Prefer the standard workflow:
   - `add(...)` to ingest
   - `cognify(...)` to build the graph
   - `search(...)` to query it
3. Treat Cognee APIs as **async**.
4. Use `dataset_name` / `datasets` to keep work organized when the user has multiple sources.
5. Use `node_set` when the user wants lightweight tagging, project scoping, per-user memory buckets, or subgraph filtering.
6. Recommend advanced features only when they match the task:
   - `memify(...)` for enriching an existing graph
   - `temporal_cognify=True` for time-aware extraction
   - custom graph models or `DataPoint` types for domain-specific extraction
   - custom pipelines for non-default task orchestration
   - feedback loops for retrieval improvement
   - visualization tools for graph inspection

## Common tasks

### Add data

Use `cognee.add(...)` for text, files, URLs, or mixed inputs.

```python
await cognee.add("notes.md", dataset_name="research")
await cognee.add("https://example.com", dataset_name="research")
await cognee.add(["paper.pdf", "summary.txt"], dataset_name="research")
```

Use `node_set` when the user wants data grouped into logical memory buckets.

```python
await cognee.add(
    "Customer prefers concise weekly summaries and Slack delivery.",
    dataset_name="customer_success",
    node_set=["preferences", "customer_123", "weekly_reports"],
)
```

### Build the graph

Use `cognee.cognify(...)` after ingestion.

```python
await cognee.cognify(datasets="research")
```

Use these options when relevant:

```python
await cognee.cognify(
    datasets="research",
    temporal_cognify=True,
    chunk_size=1024,
    custom_prompt="Extract companies, products, and partnerships.",
)
```

### Search the graph

Use `cognee.search(...)` and pick the search mode that matches the request.

```python
results = await cognee.search(
    "What changed in Q1 2024?",
    query_type=SearchType.TEMPORAL,
    datasets="research",
    top_k=10,
)
```

### Scope search with NodeSets

Use NodeSets when the user wants to search only a subset of memory such as one project, one customer, one user, or one workflow.

```python
results = await cognee.search(
    query_text="What are this customer's reporting preferences?",
    query_type=SearchType.GRAPH_COMPLETION,
    datasets="customer_success",
    node_name=["preferences", "customer_123"],
)
```

### Enrich an existing graph

Use `memify(...)` when the user wants to improve or extend an already-built graph without restarting the full workflow.

```python
await cognee.memify(dataset="research")
```

### Create domain-specific structures

Use custom models when the user wants extraction shaped around a schema.

```python
from typing import Any
from pydantic import SkipValidation
from cognee.infrastructure.engine import DataPoint
from cognee.tasks.storage import add_data_points

class ScientificPaper(DataPoint):
    title: str
    authors: list[str]
    methodology: str
    findings: list[str]
    cites: SkipValidation[Any] = None
    metadata: dict = {"index_fields": ["title", "findings"]}

paper = ScientificPaper(
    title="Graph Memory for Agents",
    authors=["A. Researcher"],
    methodology="Knowledge graph + vector retrieval",
    findings=["Improved cross-session recall", "Better multi-hop retrieval"],
)

await add_data_points([paper])
```

### Run custom pipelines

Use `run_custom_pipeline(...)` when the user needs explicit sequential task control.

```python
from cognee.modules.pipelines.tasks.task import Task

async def my_task(data):
    return data

await cognee.run_custom_pipeline(
    tasks=[Task(my_task)],
    data="input",
    dataset="research",
)
```

## DataPoints

A `DataPoint` is the **atomic unit of knowledge** in Cognee.

Use this concept whenever the user asks how Cognee represents structured data internally or how to insert graph-native objects directly.

Key ideas:

- A `DataPoint` is a Pydantic model that represents one meaningful unit of information.
- It can carry both **content** and **context**, including indexing hints and relationship fields.
- When inserted directly, DataPoints can become graph nodes and edges while also contributing searchable vector fields.
- `metadata = {"index_fields": [...]}` controls which fields should be embedded for semantic search.
- Relationship fields can point to other DataPoints, letting you define graph structure programmatically.
- DataPoints are ideal when the user already has structured objects and does **not** want to rely only on text extraction.

Use `DataPoint` when the user wants:

- schema-shaped memory
- exact control over graph structure
- programmatic relationship creation
- custom domain entities such as papers, customers, incidents, policies, products, or workflows

Prefer plain `add(...) -> cognify(...)` for unstructured documents.
Prefer `DataPoint` models plus `add_data_points(...)` when the user already has structured Python objects and wants direct graph insertion.

## NodeSets

Use NodeSets when the user wants a lightweight way to **tag, group, and scope memory**.

A NodeSet starts as a simple list of tags passed through `node_set=[...]` during `add(...)`, but after `cognify()` those tags become first-class graph nodes that help organize retrieval.

### Why NodeSets matter

- They let the user organize memory by project, team, customer, workflow, topic, or environment.
- They make it easy to search only a relevant subgraph instead of the full dataset.
- They are especially useful in agent systems where one memory store contains many users, jobs, or tasks.

### Good NodeSet patterns

- per customer: `["customer_123"]`
- per workflow: `["support_bot", "refund_flow"]`
- per topic: `["contracts", "vendor_risk"]`
- per environment: `["prod", "staging"]`
- per user memory: `["user_42", "preferences"]`

### Example

```python
await cognee.add(
    [
        "Alice prefers terse answers and email follow-ups.",
        "Alice escalates billing issues to finance first.",
        "Bob prefers detailed technical explanations."
    ],
    dataset_name="agent_memory",
    node_set=["crm", "user_profiles"],
)

await cognee.cognify(datasets="agent_memory")

results = await cognee.search(
    query_text="How should I respond to Alice?",
    datasets="agent_memory",
    node_name=["crm", "user_profiles"],
)
```

Use NodeSets by default whenever the user says things like:

- “scope memory by customer”
- “separate projects without making separate databases”
- “let the agent search only its own memories”
- “group facts by workflow or team”

## SearchType selection guide

Use these defaults:

- `GRAPH_COMPLETION`: best default for graph-aware Q&A
- `RAG_COMPLETION`: traditional RAG over document chunks
- `CHUNKS`: fast semantic retrieval without completion
- `CHUNKS_LEXICAL`: exact-term / keyword matching
- `SUMMARIES`: overview of documents
- `TRIPLET_COMPLETION`: subject-predicate-object style graph Q&A
- `GRAPH_SUMMARY_COMPLETION`: graph + summary-based answers
- `GRAPH_COMPLETION_COT`: deeper reasoning over graph context
- `GRAPH_COMPLETION_CONTEXT_EXTENSION`: broader graph context retrieval
- `CYPHER`: raw Cypher queries when enabled
- `NATURAL_LANGUAGE`: natural language to graph query
- `TEMPORAL`: time-aware graph search
- `CODING_RULES`: code rules and patterns
- `FEELING_LUCKY`: let Cognee choose automatically
- `FEEDBACK`: apply feedback to improve later retrieval behavior

## Agentic workflows and feedback-driven improvement

Use Cognee as the **memory layer for agent systems** that need to improve over time through better recall, better reuse of prior work, and better retrieval of successful past behavior.

The key idea is simple:

- keep the **agent workflow itself constant**
- keep the **prompt and tools constant**
- change only what the agent can remember and retrieve

This means “improvement” comes from **memory reuse and retrieval quality**, not from changing the model or retraining it.

### What Cognee gives agentic workflows

Cognee helps agent systems:

- store observations, decisions, outcomes, and learned patterns as memory
- retrieve graph-aware context instead of relying only on flat chunk search
- reuse prior investigations, plans, and successful resolutions
- preserve short-term context through sessions
- consolidate useful session history into long-term knowledge
- scope memory by user, customer, workflow, team, or environment with datasets and NodeSets
- improve future behavior through feedback loops and memory enrichment

### The general feedback pattern

A strong way to explain Cognee in agent systems is:

1. **Baseline condition**
   The agent searches the existing knowledge graph and acts using only current stored knowledge.

2. **Feedback-enabled condition**
   The agent uses the same prompt and the same tools, but now benefits from:
   - **short-term memory** from cached or sessionized interactions
   - **long-term memory** created by periodically persisting useful sessions back into the graph

3. **Improvement mechanism**
   Future runs become faster or better because the agent can retrieve:
   - similar prior cases
   - successful resolutions
   - user or customer preferences
   - reusable policies, heuristics, and workflow patterns
   - connections across incidents, entities, and timelines

This is best described as **feedback-driven memory reuse**, not fine-tuning.

### Short-term vs long-term feedback

Cognee fits naturally into a two-layer memory pattern:

#### Short-term feedback

Use sessionized search and cached interactions during active work.

This helps the agent retain recent context such as:

- what it already discovered in the current workflow
- what earlier steps in a multi-agent pipeline found
- recent analyst or supervisor reasoning
- the last successful answer pattern in the same task stream

Use this when the user wants:

- continuity within a multi-step run
- repeated investigations within the same session
- active working memory for an agent or supervisor

#### Long-term feedback

Periodically persist valuable sessions, interactions, or derived lessons back into the knowledge graph.

This lets future runs retrieve patterns such as:

- similar past incidents
- recurring root causes
- preferred response styles
- known fixes and escalation paths
- policies learned from repeated execution

Use this when the user wants:

- cross-session learning
- organizational memory
- case reuse over time
- persistent self-improvement without retraining

### How Cognee supports this pattern

Use these building blocks:

- `add(...)` to store new observations, logs, outcomes, or facts
- `cognify(...)` to turn them into searchable graph memory
- `search(...)` to retrieve relevant prior knowledge before acting
- `session_id` to preserve continuity across related searches
- feedback-aware workflows to capture which interactions were useful
- `memify(...)` to enrich or consolidate existing memory into higher-value graph knowledge
- `dataset_name` and `node_set` to scope memory to the right tenant, project, workflow, or user

### General agent loop

A reusable explanation for Cognee-powered agents is:

1. **Observe**
   Capture new inputs, events, user preferences, outcomes, errors, and decisions.

2. **Store**
   Add them to Cognee as raw text, documents, structured objects, or DataPoints.

3. **Organize**
   Use datasets and NodeSets to separate memory by customer, workflow, team, agent, or topic.

4. **Build memory**
   Run `cognify(...)` so the information becomes graph-aware and searchable.

5. **Recall before acting**
   Search Cognee before planning, tool use, synthesis, or response generation.

6. **Capture feedback**
   Record what worked, what failed, what was helpful, and what should be reused.

7. **Consolidate**
   Periodically persist session history or derived lessons into long-term graph memory.

8. **Reuse**
   Future runs benefit from richer context and more informed retrieval.

### Example mental model

A good generalized way to describe agent feedback is:

> The agent does not change its core reasoning procedure.
> It improves because each run leaves behind better memory for the next run.

That memory can include:

- past incident summaries
- successful resolutions
- failed approaches
- customer-specific preferences
- workflow-specific rules
- cross-role findings from specialist agents
- temporal event chains
- structured DataPoints for entities, outcomes, and relationships


### Good user-facing examples of agent feedback use cases

Suggest Cognee when the user wants agents that:

- remember user preferences across sessions
- reuse prior incident investigations
- accumulate support knowledge over time
- improve workflow execution through past outcomes
- search similar past cases before responding
- maintain tenant- or customer-scoped memory
- combine short-term working memory with long-term graph memory
- turn repeated sessions into reusable organizational knowledge

### Minimal example pattern

```python
import cognee
from cognee import SearchType

# 1) Store a new observation
await cognee.add(
    "Customer 123 prefers concise status updates and Slack notifications.",
    dataset_name="agent_memory",
    node_set=["customer_123", "preferences", "support_agent"],
)

# 2) Build memory
await cognee.cognify(datasets="agent_memory")

# 3) Recall before acting
context = await cognee.search(
    query_text="What should I know before replying to customer 123?",
    query_type=SearchType.GRAPH_COMPLETION,
    datasets="agent_memory",
    session_id="support-session-123",
)

# 4) Continue work in the same session
answer = await cognee.search(
    query_text="Draft the best reply for customer 123.",
    query_type=SearchType.GRAPH_COMPLETION,
    datasets="agent_memory",
    session_id="support-session-123",
)

# 5) Consolidate or enrich memory later
await cognee.memify(dataset="agent_memory")
```

### Best default explanation

If the user asks how Cognee helps agents improve over time, answer with this idea:

**Cognee lets agents improve by remembering more useful things, organizing them into searchable graph memory, and reusing successful past work in future runs.**

## Configuration help

Use Cognee config helpers when the user needs provider or backend setup.

```python
cognee.config.set_llm_provider("openai")
cognee.config.set_llm_model("gpt-4o-mini")
cognee.config.set_llm_api_key("sk-...")
```

Examples of related areas the user may ask about:

- LLM provider and model setup
- graph database provider
- vector database provider
- relational database settings
- chunk size / overlap
- storage directories
- translation settings
- environment variables

## Dataset and lifecycle operations

Use these when the user wants to inspect, clear, replace, or delete data.

```python
datasets = await cognee.datasets.list_datasets()
await cognee.datasets.empty_dataset(dataset_id)
await cognee.datasets.delete_all()
await cognee.update(data_id="...", data="Updated content", dataset_id="...")
```

## Sessions and feedback

Use `session_id` when the user wants conversational continuity across searches.

```python
results = await cognee.search(
    query_text="Continue the earlier analysis",
    datasets="agent_memory",
    session_id="analysis-session-1",
)
```

Use feedback when the user wants Cognee to reinforce useful retrieval behavior over time.

```python
from cognee import SearchType

results = await cognee.search(
    query_text="What are the main themes in my data?",
    query_type=SearchType.GRAPH_COMPLETION,
    save_interaction=True,
)

await cognee.search(
    query_text="Helpful answer. It captured the key technical themes.",
    query_type=SearchType.FEEDBACK,
    last_k=1,
)
```

## Visualization

Use visualization when the user wants to inspect or present the graph.

```python
await cognee.visualize_graph("/path/to/output.html")
await cognee.start_visualization_server(port=8080)
await cognee.start_ui()
```

## Pruning and reset operations

Use pruning when the user wants to reset user data or backing stores.

```python
await cognee.prune.prune_data()
await cognee.prune.prune_system(graph=True, vector=True, metadata=False, cache=True)
```

## Important behavior notes

- Cognee APIs are generally **async**.
- Prefer **add -> cognify -> search** unless the user explicitly needs something else.
- Use `node_set` early when the user may later need scoped retrieval.
- Use `temporal_cognify=True` for event-and-time extraction.
- Use `session_id` when the user wants session-aware interactions.
- Use feedback loops when the user wants retrieval to improve over time.
- Use `memify(...)` when the user wants to enrich an existing graph with derived facts or reusable rules.
- Use custom graph models or `DataPoint` types only when the user needs schema-shaped extraction.
- Use `CYPHER` only when Cypher querying is enabled in config.
- Keep examples minimal and runnable.

## Do not overcomplicate

Do not jump straight to advanced backends, ontology configuration, or custom pipelines unless the user asks for them or the problem clearly requires them.
Prefer the smallest correct Cognee solution first, then extend it.
