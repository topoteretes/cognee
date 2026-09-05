# Product journey tests

High-level contract tests that check what a user experiences, not what a
module does. Each file is one journey with content-level assertions: a fact
must come back, a count must match, a status must be honest. "Returned
something" is never a pass here.

| File | Journey | What breaks it |
|---|---|---|
| `test_quickstart.py` | Build the wheel, install into an empty venv, run README remember/recall and the CLI entry point | Packaging, entry points, first-run DB creation, import-time side effects |
| `test_memory_correctness.py` | Remember 20 fictional documents, ask 30 gold questions through recall and every search type | Retrieval or graph regressions that still return non-empty results |
| `test_session_journey.py` | Remember in a session, recall from cache, bridge into the graph with improve, delete the session | Session cache, keyword session search, improve bridging, cache deletion |
| `test_data_lifecycle.py` | remember, update, forget one item, forget dataset, forget everything, remember again; snapshot all three stores after each step | Stale vectors or graph nodes after update/delete, deletion leaving zombies |
| `test_idempotency_and_recovery.py` | Same content twice, explicit cognify re-run, injected LLM outage, retry | Duplicate ingestion, half-written graphs, dishonest pipeline status |
| `test_http_api_journey.py` | Register, login, add, cognify (blocking + background polling), search, remember/recall, isolation between users, delete; route-table snapshot | Wire-shape drift, auth holes, 500s on bad input, polling contract |
| `test_mcp_journey.py` | Spawn `cognee-mcp` over stdio like Claude Code does: list tools, session remember on a fresh system, recall, permanent remember, graph recall, status; then read the same state over HTTP | stdout pollution breaking the protocol, first-session dataset creation (SDK-192), tool surface drift, transport-specific bugs |
| `test_extras_install_smoke.py` | Resolve every extra alone and all together on Python 3.10 to 3.14 (always on); build the wheel and install each extra into an empty venv, then import `cognee`, every module the extra's requirements ship, and the cognee modules it enables (opt-in) | Extras that conflict (the docling-full vs codegraph case), extras that install but do not import, adapters broken by a dependency bump |

## Modes

`COGNEE_JOURNEY_MODE=mock` (default) swaps the two AI calls for deterministic
stand-ins from `mock_ai.py`:

- The LLM replays pre-authored knowledge graphs for corpus documents, extracts
  capitalised phrases for anything else, and answers by echoing the retrieved
  context (never the question). A fact can only pass if retrieval surfaced it.
- Embeddings are hashed bag-of-words vectors, so vector search ranks by shared
  vocabulary. Deterministic across processes and machines.

No network, no secrets. This tier runs on every PR, including forks.

`COGNEE_JOURNEY_MODE=llm` uses the real providers from the environment. The
correctness journey switches to threshold assertions and additionally enforces
`forbidden` tokens: a concise answer must not cite facts from unrelated
documents.

## Running

```bash
# everything except the slow quickstart
pytest cognee/tests/journeys -m "journey and not quickstart"

# quickstart (builds a wheel, needs uv)
COGNEE_JOURNEY_QUICKSTART=1 pytest cognee/tests/journeys -m quickstart

# real LLM
COGNEE_JOURNEY_MODE=llm pytest cognee/tests/journeys -m "journey and not quickstart"

# accept an intentional route-table change
COGNEE_UPDATE_API_SNAPSHOT=1 pytest cognee/tests/journeys/test_http_api_journey.py

# MCP journey (needs the mcp client and fastmcp in the interpreter)
uv run --with fastmcp --with mcp pytest cognee/tests/journeys/test_mcp_journey.py

# extras install smoke (slow; pick a subset with COGNEE_EXTRAS)
COGNEE_JOURNEY_EXTRAS=1 COGNEE_EXTRAS=neo4j,redis pytest cognee/tests/journeys/test_extras_install_smoke.py
```

## Golden corpus

`golden_corpus/documents.json` describes the fictional town of Kestrel Hollow.
Every fact is invented so a real model cannot answer from prior knowledge.
Each document carries the knowledge graph and summary the mock LLM replays for
it. `golden_corpus/questions.json` holds 30 questions with `expected_any`
tokens (any one satisfies) and `forbidden` tokens from other documents.

Add a document by appending to both files; keep texts to a single chunk and
give each new entity a distinctive name so hashed embeddings separate it.

## Why questions run in their own sessions

Without a `session_id`, every turn shares one per-dataset default session and
the conversational lane merges the previous turn's retrievals into the current
context. Thirty unrelated questions asked back to back would contaminate each
other, so the correctness journey isolates each question. The session journey
covers the conversational behaviour on purpose.
