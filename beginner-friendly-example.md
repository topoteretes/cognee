# A Beginner-Friendly Example: Your First Knowledge Graph

This walkthrough is for anyone new to Cognee or to knowledge graphs in general.
It uses one small, human-readable example so you can see exactly what Cognee
does at each step -- not just run a black-box script.

## 1. Setup

Install Cognee:

```bash
pip install cognee
```

Cognee needs an LLM to extract entities and relationships from your text, and
an embedding model for semantic search. The simplest setup uses OpenAI:

```dotenv
# .env
LLM_API_KEY="sk-..."
```

(Cognee supports many other providers -- Gemini, Anthropic, local models via
Ollama, and more. See the [Installation guide](/getting-started/installation)
for details.)

## 2. A minimal input

Knowledge graphs are easiest to understand with a tiny, concrete example
instead of a large document. Here's all the input text we'll use:

```text
Alice is a software engineer who works at Cognee.
Cognee is a company that builds AI memory systems.
Alice's manager is Bob, who leads the engineering team.
```

Store it in memory with one call:

```python
import cognee

await cognee.remember(
    "Alice is a software engineer who works at Cognee. "
    "Cognee is a company that builds AI memory systems. "
    "Alice's manager is Bob, who leads the engineering team."
)
```

## 3. How the knowledge graph is created

Under the hood, `remember()` runs a short pipeline on your text:

1. **Chunking** -- the text is split into manageable pieces (our example is
   small enough to stay as one chunk).
2. **Entity extraction** -- an LLM reads each chunk and identifies the
   distinct "things" being talked about. In our example, that's:
   - `Alice` (a person)
   - `Bob` (a person)
   - `Cognee` (a company)
3. **Relationship extraction** -- the LLM also identifies how those entities
   connect to each other:
   - `Alice` --`works_at`--> `Cognee`
   - `Alice` --`manager_is`--> `Bob`
   - `Bob` --`leads`--> `engineering team`
4. **Embedding** -- each entity and chunk is also converted into a vector, so
   it can be found later by meaning, not just by exact keyword match.

The result is a small graph with three nodes (`Alice`, `Bob`, `Cognee`)
connected by labeled edges -- not just three sentences sitting in a database.

## 4. Asking a question

Once memory is built, you can ask questions in plain English:

```python
answer = await cognee.recall(query_text="Who does Alice work for, and who is her manager?")

for result in answer:
    print(result.text)
```

Cognee automatically figures out the best way to search the graph to answer
your question -- you don't need to pick a search strategy yourself.

## 5. Interpreting the output

`recall()` returns a list of result objects. The two fields you'll use most:

| Field | What it means |
|---|---|
| `result.text` | The actual answer text (or retrieved context, if you used `only_context=True`) |
| `result.source` | Where the result came from -- `"graph"` for knowledge-graph-backed answers |

If you want to see *why* Cognee gave a particular answer, skip the final
answer-generation step and look at the raw retrieved graph context instead:

```python
context = await cognee.recall(
    query_text="Who does Alice work for, and who is her manager?",
    only_context=True,
)
```

This returns the actual graph facts Cognee used -- for example, the
`Alice --works_at--> Cognee` and `Alice --manager_is--> Bob` relationships --
before any answer was written. This is the clearest way to "see" the
knowledge graph as data, rather than just as a generated sentence.

## Next steps

- Try changing the input text to a short bio about yourself or your project,
  and see what entities and relationships Cognee extracts.
- See the [Core Concepts](/core-concepts/overview) page to learn about
  DataPoints, Tasks, and Pipelines -- the building blocks behind `remember()`.
- See [Recall](/core-concepts/main-operations/recall) for more on how
  querying works.