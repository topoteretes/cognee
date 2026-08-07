# Session search — code walkthrough

A reading guide for the SDK-361 branch. Five flows, each a numbered trace through real
functions in the order they execute. Every step names the file and line it happens in.

They are ordered so you can read straight through: flow 1 gets you to the fork, flows 2 and 3
are the two sides of it, flow 4 zooms into one step of flow 3, and flow 5 is what happens after
the session ends. Each flow also stands alone if you only need one.

## How the flows fit together

```
        await cognee.search("...", session_id="s1")
                        │
                   ┌────┴────┐
                   │ FLOW 1  │  entry → build retriever → reach the fork
                   └────┬────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
   sequential mode                 concurrent mode  (default)
         │                             │
    ┌────┴────┐                   ┌────┴────┐
    │ FLOW 2  │                   │ FLOW 3  │      ┌────────┐
    │ 2 calls │                   │ 1 call  │- - ->│ FLOW 4 │  its step 5c, up close:
    │ in a row│                   │ + 1 free│      └────────┘  merging two retrievals
    └────┬────┘                   └────┬────┘
         │                             │
         └──────────┬──────────────────┘
                    │  both write the same two stores
             ... session ends ...
                    │
              ┌─────┴─────┐
              │  FLOW 5   │  distillation → knowledge graph
              └───────────┘
```

`- - ->` marks a zoom-in, not a step. Flow 4 is one line of flow 3 explained in full.

## Vocabulary

Fixed meanings, used consistently throughout.

| Term | Meaning |
| --- | --- |
| **turn** | one user message plus the answer to it |
| **analysis** | the LLM call that reads the user's message and decides what to remember |
| **answer call** | the LLM call that produces the text the user reads |
| **QA store** | recorded turns — question, answer, which graph objects and context entries were used |
| **context store** | durable session guidance ("prefer metric units", "I work on billing") |
| **sequential mode** | `SESSION_SEARCH_MODE=sequential` — analysis first, then retrieval, then answer |
| **concurrent mode** | `SESSION_SEARCH_MODE=concurrent` — analysis alongside the answer; the default |
| **concurrent turn** | one turn executed the way concurrent mode executes them |
| **lane** | one of two things running at the same time under `asyncio.gather` |
| **fail open** | on error, log it and return an empty result so the caller carries on. Session code does this everywhere: a broken cache must never break an answer |

*Concurrent mode* is the deployment setting; a *concurrent turn* is one turn executed that
way. Same word on purpose — setting, code, and prose all name the same mechanism.

The QA store and the context store are the whole of session state. Both modes read and write
both, through the same two functions. Hold onto that — it is why flow 5 needs only one version.

---

# Flow 1 — How a search reaches the fork

**Starts:** a user calls `await cognee.search("...", session_id="s1")`.
**Ends:** one function decides whether this search runs as a concurrent turn.

### 1 — The public API validates and resolves the user

[`search()` (L40)](../cognee/api/v1/search/search.py) checks the numeric arguments, resolves
`user` (falling back to the default user), and turns dataset names into ids.

### 2 — The user is published to the async context

[`set_session_user_context_variable(user)` (L313)](../cognee/api/v1/search/search.py)

Worth remembering. It sets a `contextvars` variable — a global scoped to the current async
task — that deeper code reads as `session_user.get()` without it being passed down through
every call signature. Step 6 below is the first place that read matters.

### 3 — Routing by dataset

[`search_function(...)` (L342)](../cognee/api/v1/search/search.py) hands off to
[`search()` (L40)](../cognee/modules/search/methods/search.py), which resolves permissions and
calls [`search_in_datasets_context` (L213)](../cognee/modules/search/methods/search.py) so the
right database is active for the dataset being queried.

### 4 — Build the retriever

[`get_retriever_output` (L54)](../cognee/modules/search/methods/get_retriever_output.py) is now
in charge of one search against one dataset. It:

- resolves `FEELING_LUCKY` to a real search type — [`_effective_search_type` (L29)](../cognee/modules/search/methods/get_retriever_output.py)
- warns if the graph is empty
- constructs the retriever — [`get_search_type_retriever_instance` (L65)](../cognee/modules/search/methods/get_retriever_output.py)

The retriever now holds this query's settings: `top_k`, prompt paths, `response_model`, and
`session_id`.

### 5 — The fork

[`try_concurrent_turn(retriever, ...)` (L71)](../cognee/modules/search/methods/get_retriever_output.py)

This one call is the entire integration surface. It returns one of two things:

| Return | Meaning | What the caller does |
| --- | --- | --- |
| `None` | not eligible | falls through to the code that was always there → **flow 2** |
| `ConcurrentTurnResult` | the turn already ran | returns it → **flow 3** |

Nothing else in the sequential path was modified. That is the point of the `None`.

### 6 — Inside the fork: four checks, cheapest first

[`try_concurrent_turn` (L194)](../cognee/modules/retrieval/session_search.py)

```
1. CacheConfig().session_search_mode != "concurrent"  →  None
2. session_user.get() is None, or the user has no id  →  None
3. can_run_as_concurrent_turn(...) is False           →  None
4. otherwise                                          →  run the turn (flow 3)
```

Check 1 is first deliberately — [L211](../cognee/modules/retrieval/session_search.py). A
deployment on sequential mode never builds a session manager and never reads the cache. The
feature costs it nothing.

Check 3 is [`can_run_as_concurrent_turn` (L60)](../cognee/modules/retrieval/session_search.py),
which rejects five situations:

| Rejected | Reason |
| --- | --- |
| `FEELING_LUCKY` | the search type isn't known yet at dispatch time |
| session unavailable | there is no conversation to be part of |
| batch query | many questions, one history — no coherent turn |
| `only_context=True` | the caller wants context, not an answer |
| retriever class not on the allowlist | see below |

### 7 — Why the allowlist compares exact types

```python
retriever_type not in _eligible_retriever_types()   # `not in`, never isinstance
```

Five classes are eligible — [L33](../cognee/modules/retrieval/session_search.py):
`CompletionRetriever`, `GraphCompletionRetriever`, `GraphSummaryCompletionRetriever`,
`HybridRetriever`, `TripletRetriever`.

Subclasses are excluded on purpose. `GraphCompletionRetrieverCoT` and the context-extension
variant run *extra* LLM rounds inside retrieval. Concurrent mode promises a turn costs one
answer call; those classes cannot keep that promise. An `isinstance` check would let them
inherit their way onto the list silently. With `not in`, a new subclass is safe by default and
has to be added deliberately.

### 8 — The same fork exists on three other doors

Not every search goes through `get_retriever_output`. Code holding a retriever can call it
directly, so the fork is repeated at each complete-search entry point:

| Entry point | Covers |
| --- | --- |
| [`BaseRetriever.get_completion` (L132)](../cognee/modules/retrieval/base_retriever.py) | `CompletionRetriever`, `TripletRetriever` |
| [`GraphCompletionRetriever.get_completion` (L405)](../cognee/modules/retrieval/graph_completion_retriever.py) | overrides the base to handle batches |
| [`HybridRetriever.get_completion` (L525)](../cognee/modules/retrieval/hybrid_retriever.py) | same reason |

The three *partial* methods — `get_retrieved_objects`, `get_context_from_objects`,
`get_completion_from_context` — must never fork. A concurrent turn retrieves twice and answers
once; a caller stepping through one method at a time is asking for something else. A test reads
their source and asserts the call is absent:
[`test_partial_operations_do_not_dispatch` (L184)](../cognee/tests/unit/modules/retrieval/test_session_search_modes.py).

---

# Flow 2 — One turn in sequential mode

**Starts:** the fork returned `None` — either the deployment is on sequential mode, or this
call did not qualify (flow 1, step 6).
**Ends:** the answer is generated and the turn is recorded.
**Shape:** two LLM calls, strictly one after the other.

```
analysis ──► retrieval ──► answer
    └── its rewritten query is what retrieval searches for
```

### 1 — Fall through the fork

[`get_retriever_output` (L71)](../cognee/modules/search/methods/get_retriever_output.py) got
`None` and continues into the code that predates this branch.

### 2 — Analyze the turn — **LLM call 1**

[`prepare_session_turn_for_retrieval` (L101)](../cognee/modules/retrieval/base_retriever.py)
→ `SessionManager.prepare_session_turn`
→ [`prepare_session_turn` (L332)](../cognee/infrastructure/session/session_turn.py)

Inside, in order:

a. read the single previous QA entry — `get_session(..., last_n=1)`

b. [`load_served_context_payload` (L199)](../cognee/infrastructure/session/session_turn.py)
   resolves the context-entry ids that were served to the **previous** answer into
   `{id, content}` pairs, so the analysis can judge whether they helped

c. [`analyze_turn_for_session_context` (L45)](../cognee/infrastructure/session/feedback_detection.py)
   makes one LLM call that returns a filled-in `SessionTurnAnalysis` object:

   | Field | Purpose | Used by |
   | --- | --- | --- |
   | `query_to_answer` | rewritten query for retrieval | sequential mode only |
   | `response_to_user` | a reply, when nothing needs looking up | sequential mode only |
   | `candidate_context_updates` | new or revised durable guidance | both modes |
   | `served_context_ratings` | helpful/harmful verdicts on last turn's guidance | both modes |

d. [`apply_session_turn_analysis` (L279)](../cognee/infrastructure/session/session_turn.py)
   writes the evidence entry, applies the candidate updates to the context store, and bumps the
   helpful/harmful counters

e. `should_answer` and `effective_query` are decided from those fields

### 3 — Short-circuit a conversational turn

If `should_answer` is `False`,
[`should_answer` (L97)](../cognee/modules/search/methods/get_retriever_output.py) returns the
acknowledgement immediately. No retrieval. No second LLM call. "ok, thanks" costs one call.

### 4 — Retrieve once

[`get_retrieved_objects` (L112)](../cognee/modules/search/methods/get_retriever_output.py)
is called with the **rewritten** query from step 2. Then
`update_node_access_timestamps`, then `get_context_from_objects`.

One retrieval, one result set. Nothing to merge — that is what makes flow 4 concurrent-only.

### 5 — Answer — **LLM call 2**

`get_completion_from_context` → `generate_completion_with_session` →
[`_run_session_turn` (L379)](../cognee/infrastructure/session/session_manager.py), which
receives the `turn_preparation` from step 2 so the analysis is not repeated. Then
[`generate_session_answer` (L125)](../cognee/infrastructure/session/session_turn.py):

a. `select_session_history` — recent turns unioned with vector-recalled turns, chronological

b. `build_active_context_block_safe` — renders the guidance block **and reports which entry ids
   it served**

c. `compose_session_prompt` — guidance block placed above conversation history

d. the answer call

### 6 — Record the turn

[`add_qa` (L447)](../cognee/infrastructure/session/session_manager.py) stores the question, the
answer, the graph objects used, and the context-entry ids served in step 5b.

### What this mode buys and costs

**Buys:** guidance stated *this* turn is written in step 2d, so step 5b can serve it during the
same turn. Say "stop citing the 2019 report" and this very answer obeys.

**Costs:** two LLM calls end to end. The second cannot start until the first lands, because it
needs the rewritten query.

---

# Flow 3 — One turn in concurrent mode (the default)

**Starts:** the fork passed all four checks (flow 1, step 6).
**Ends:** the answer is returned and the turn is committed.
**Shape:** the same two LLM calls — running at the same time.

```
                    ┌─ analysis ─────────────────────────┐
snapshot ──► gather │                                    ├──► commit
                    └─ retrieve ×2 ─► merge ─► answer ───┘
```

**Why this is even possible:** the analysis never reads the answer. It reads the user's message
against the *previous* turn. Once that is true, running the two in sequence buys nothing.

### 1 — Resolve the session

`session_manager.resolve_session_id(retriever.session_id)`.

### 2 — Take the turn lock

[`session_turn_lock(user_id, session_id)` (L88)](../cognee/infrastructure/locks/session_lock.py)

One turn at a time per user and session. Without it, two quick turns would read the same state
and then overwrite each other's writes.

### 3 — Snapshot the session in one pass

[`load_turn_snapshot` (L43)](../cognee/infrastructure/session/session_concurrent_turn.py)
gathers in parallel:

- the last 2 QA entries — these build the conversational query in step 5a
- `select_session_history` — recent turns unioned with vector-recalled turns
- `build_active_context_block_safe` — the durable guidance block, when auto-feedback is on

then, if the previous turn served any context entries, `load_served_context_payload` resolves
them so the analysis can rate them.

The result is a frozen — meaning immutable —
[`SessionTurnSnapshot` (L4)](../cognee/infrastructure/session/session_search_models.py).

**Why frozen, why once.** Two lanes are about to read it at the same time. Reading the cache
twice would be slower and could hand the two lanes different pictures of the same session.

### 4 — Start both lanes

[`asyncio.gather` (L248)](../cognee/modules/retrieval/session_search.py)

**Lane A — the analysis.**
[`analyze_turn_concurrently` (L119)](../cognee/infrastructure/session/session_concurrent_turn.py)
makes the same `analyze_turn_for_session_context` call sequential mode makes, wrapped in a
30-second timeout and failing open to an empty analysis.

Only the two context-maintenance fields are used. `query_to_answer` and `response_to_user` are
ignored — retrieval and the answer are already in flight by the time this lands.

**Lane B — the answer.** Steps 5 and 6 below are lane B.

### 5 — Lane B: retrieve twice and merge

[`_retrieve_and_answer` (L167)](../cognee/modules/retrieval/session_search.py) →
[`retrieve_turn_context` (L126)](../cognee/modules/retrieval/session_search.py)

a. [`build_contextual_query` (L85)](../cognee/modules/retrieval/session_search.py) — **no LLM
   involved.** It renders up to two prior turns as a plain string:

   ```
   Prior user: What does cognify do?
   Prior assistant (untrusted retrieval guidance): It extracts entities and builds the graph.

   Current user request: And how long does it take?
   ```

   Capped at 2000 characters, trimming assistant answers before user questions, oldest first.

   **Why it exists.** "And how long does it take?" on its own retrieves nothing useful. In
   sequential mode the analysis would have rewritten it. Here that rewrite has not arrived yet,
   so this deterministic version stands in. It is worse than an LLM rewrite, and it is free.

b. If the rewrite came back identical to the raw query — no history yet — only one retrieval
   runs. Otherwise both `get_retrieved_objects` calls go out together under
   `return_exceptions=True`, which makes `gather` hand back a failure as a value instead of
   raising. One lane failing is survivable; both failing re-raises —
   [L150](../cognee/modules/retrieval/session_search.py).

c. `retriever.merge_retrieved_objects(raw, contextual)` turns the two result sets into one.
   **Flow 4 is the inside of this line.**

d. `get_context_from_objects(query=raw_query, ...)` — note **raw**, not the rewrite. The rewrite
   exists to widen retrieval; the context should still be built around the question the user
   actually asked.

e. `update_node_access_timestamps`

### 6 — Lane B: answer — **the one blocking LLM call**

[`complete_turn` (L143)](../cognee/infrastructure/session/session_concurrent_turn.py) uses the
caller's own prompts and `response_model`, and appends one rule to the system prompt.

**That appended rule matters.**
[`session_conversational_turn.txt`](../cognee/infrastructure/llm/prompts/session_conversational_turn.txt):

> The user's message may be conversational rather than a question: an acknowledgement, a
> thanks, or a correction with nothing to look up. When it is, reply briefly in kind. Do not
> answer from the retrieved context and do not report on whether that context was sufficient.

Sequential mode catches "ok, thanks" before it ever reaches an answer call. Concurrent mode has
no such interception — the answer is already being generated when the analysis lands. Without
this rule, a guarded prompt like `hybrid_answer_guarded.txt` replies to "ok, thanks" with *"the
evidence is insufficient to answer."*

Two lines of prompt remove the need for a `should_answer` decision entirely. With the lanes
running together, skipping the answer would save no wall-clock time anyway.

### 7 — Both lanes have landed: commit

[`commit_turn` (L183)](../cognee/infrastructure/session/session_concurrent_turn.py) calls
`apply_session_turn_analysis` — the same function sequential mode calls — and then `add_qa`.

Ratings target `snapshot.previous_served_context`; the QA row records
`snapshot.active_context_ids`.

### 8 — Release and return

Lock released, `append_references` applied for string answers, `ConcurrentTurnResult` returned
to whichever door in flow 1 asked.

### What this mode buys and costs

**Buys:** one answer call of wall-clock time. The analysis rides alongside for free.

**Costs:** guidance stated this turn is committed at step 7, *after* this turn's answer, so it
takes effect from the next turn. Say "stop citing the 2019 report" and this answer may still
cite it; the following one will not.

---

# Flow 4 — Merging two retrievals into one

**Starts:** a concurrent turn has run two retrievals and holds two result sets (flow 3, step 5c).
**Ends:** one result set, ready to be formatted into context.

This is concurrent-mode-only. Sequential mode retrieves once, so it never merges anything.

### 1 — Why there are two result sets

A concurrent turn searches twice: once with the raw question, once with a rewrite carrying the
recent conversation. Both come back. Only one can be formatted into context.

### 2 — Ask the retriever to merge, because only it knows the shape

[`BaseRetriever.merge_retrieved_objects(primary, secondary)` (L87)](../cognee/modules/retrieval/base_retriever.py)

`CompletionRetriever` returns scored chunks. `GraphCompletionRetriever` returns edges.
`HybridRetriever` returns a dict of channels. No outside caller can merge all three, so the
merge is a method on the retriever.

The default implementation returns `primary` unchanged — always a valid result of the right
shape. A retriever that doesn't override it simply keeps the raw lane, which is the correct
conservative answer rather than a broken one.

### 3 — Give every item an identity

[`merge_ranked` (L56)](../cognee/modules/retrieval/utils/merge_results.py) needs to know when
two objects from different lanes are the same thing.

- default: `result_id` — the object's own id
- graph edges: [`edge_identity` (L15)](../cognee/modules/retrieval/utils/merge_results.py) —
  the stored `edge_object_id` if present, otherwise `(source, relationship, target, directed)`
- no identity derivable: a synthetic key, so the item is **kept** rather than silently dropped

### 4 — Sort into three tiers

```
tier 0   appeared in BOTH lanes    →  ordered by its rank in the raw lane
tier 1   raw lane only             →  original order
tier 2   contextual lane only      →  original order
```

Worked example. Two lanes come back:

```
raw lane          [ A, B, C ]
contextual lane   [ D, B, E ]

B is the only item in both.

tier 0   B          (rank 1 in the raw lane)
tier 1   A, C       (ranks 0 and 2, order kept)
tier 2   D, E       (ranks 0 and 2, order kept)

merged   [ B, A, C, D, E ]
```

`B` is represented by the raw lane's copy of the object, not the contextual lane's.

**Why tiers and not a score.** Appearing in both lanes is the strongest evidence available, and
it is a fact rather than a number to tune. An earlier draft used Reciprocal Rank Fusion; with
two lanes at near-equal weights it degenerates — a contextual-only hit cannot outrank a raw-only
hit until the raw hit falls past rank 63. Tiering says the intended thing directly.

### 5 — Truncate to the retriever's budget

`limit=self.top_k`, applied after sorting so the best survivors are the ones kept.

### 6 — Each retriever wires it up in two lines

| Retriever | Override |
| --- | --- |
| [`CompletionRetriever` (L65)](../cognee/modules/retrieval/completion_retriever.py) | `merge_ranked(primary, secondary, limit=self.top_k)` |
| [`TripletRetriever` (L94)](../cognee/modules/retrieval/triplet_retriever.py) | identical |
| [`GraphCompletionRetriever` (L279)](../cognee/modules/retrieval/graph_completion_retriever.py) | adds `identity=edge_identity` |
| `GraphSummaryCompletionRetriever` | inherits the graph one |
| [`HybridRetriever` (L486)](../cognee/modules/retrieval/hybrid_retriever.py) | `merge_hybrid_results(...)` |

### 7 — Hybrid needs its own module

Its result is a dict of channels, not a list —
[`merge_hybrid_results` (L59)](../cognee/modules/retrieval/hybrid/merge.py):

- each channel (`chunks`, `entities`, `facts`, `graph_fallback`) merges under its own budget
- `chunk_summaries` and `chunk_attribution` are **rebuilt** for the chunks that survived — a
  summary belonging to a dropped chunk would be a dangling reference
- `retrieval_status` marks a channel healthy if *either* lane succeeded
- `global_context` is not merged; it is built once from the raw query, so there is no second
  version of it to reconcile

---

# Flow 5 — Distillation, after the session ends

**Starts:** someone calls
[`distill_session(session_id, dataset, user)` (L384)](../cognee/modules/session_distillation/distill.py),
either directly as `cognee.session.distill_session(...)` or via
[`improve()` (L38)](../cognee/api/v1/improve/improve.py), which fans out to it per session in
[`_distill_sessions` (L390)](../cognee/api/v1/improve/improve.py).
**Ends:** accepted learnings are documents in the knowledge graph.

### 0 — Why there is only one version of this flow

[`distill.py`](../cognee/modules/session_distillation/distill.py) is byte-identical to `dev`.
The session-search work did not touch it and did not need to.

Distillation reads exactly two things, and both modes write both:

| Distillation reads | Written by | Sequential mode | Concurrent mode |
| --- | --- | --- | --- |
| QA turns | `add_qa` | flow 2, step 6 | flow 3, step 7 |
| context entries | `apply_session_turn_analysis` | flow 2, step 2d | flow 3, step 7 |

Same two functions, different moment in the turn. By the time a session ends, the stores hold
the same kind of content either way.

### 1 — Resolve scope

[`resolve_distillation_scope` (L87)](../cognee/modules/session_distillation/distill.py)
identifies the user and confirms they can write to the target dataset. A dataset is mandatory —
learnings have to land in the graph they connect to.

### 2 — Load, and gate

[`load_distillable_session_inputs` (L120)](../cognee/modules/session_distillation/distill.py)
performs the two reads above. Context entries are kept only when `harmful_count == 0` **and**
`confidence >= MIN_GATE_CONFIDENCE`.

Nothing survives → status `no_gated_entries`, stop.

### 3 — Propose lessons

[`propose_lessons` (L194)](../cognee/modules/session_distillation/distill.py)

a. [`build_curator_batches` (L148)](../cognee/modules/session_distillation/distill.py)
   interleaves QA blocks and surviving context entries into one chronological timeline, then
   slices it into size-safe batches

b. one curator LLM call per batch, run concurrently, each failing open to `[]` — a bad batch
   costs only itself

### 4 — Accept or reject each lesson

[`accept_proposed_lessons` (L320)](../cognee/modules/session_distillation/distill.py), per
lesson, in parallel:

a. two vector searches — prior lessons in `DocumentChunk_text` scoped to the `session_learnings`
   node set, a tag that marks previously distilled documents (*is this already known?*), and an
   entity glossary from `Entity_name` (*what should this be called?*)

b. one writer call returning `WrittenLesson(accept, statement, why_learned)`

### 5 — Publish

[`publish_distilled_lessons` (L364)](../cognee/modules/session_distillation/distill.py) renders
one markdown document per accepted lesson — a template controls the format, not the LLM — then
runs `add()` and `cognify()` in one pass, tagged with `session_learnings` and the session's
truth node set.

### The two things the mode does change

Neither is structural. Both are one-turn offsets.

**Rating lag.** Sequential mode writes a new context entry *before* that turn's answer, so it
can be served on turn N and rated on turn N+1. Concurrent mode writes it *after*, so it is
served on N+1 and rated on N+2. The `harmful_count == 0` gate in step 2 therefore sees one fewer
round of evidence. Visible mainly on very short sessions, where a bad entry could reach
distillation unrated.

**Which ids the QA row records.** Sequential mode stores the ids
`build_active_context_block_safe` served during the answer. Concurrent mode stores the ids from
the snapshot, read before the analysis ran. The two differ only when this turn's analysis
created an entry that sequential mode would have served immediately.

Neither changes what distillation reads or how it decides.

---

# Appendix — reading order for the diff

To review rather than to understand, this order minimises backtracking:

1. [`session_search_models.py`](../cognee/infrastructure/session/session_search_models.py) — 24 lines, the data shape everything else moves around
2. [`merge_results.py`](../cognee/modules/retrieval/utils/merge_results.py) — self-contained, no session concepts (flow 4)
3. [`hybrid/merge.py`](../cognee/modules/retrieval/hybrid/merge.py) — the same idea over a dict of channels
4. [`session_concurrent_turn.py`](../cognee/infrastructure/session/session_concurrent_turn.py) — four functions, in call order (flow 3, steps 3, 4, 6, 7)
5. [`session_search.py`](../cognee/modules/retrieval/session_search.py) — the orchestrator; read `try_concurrent_turn` last, top to bottom
6. the retriever diffs — two to four lines each (flow 4, step 6)
7. [`get_retriever_output.py`](../cognee/modules/search/methods/get_retriever_output.py) — the fork, plus one unrelated tidy-up (`_dataset_fields`)
