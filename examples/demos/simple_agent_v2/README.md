# simple_agent_v2 — Demo Overview

## The Problem Setup

A SaaS company receives upgrade request emails. Only the free tier can be approved automatically — everything else is rejected.

Two agents handle each email: **Agent A (Sales rep)** proposes a package, **Agent B (Compliance checker)** approves or rejects it with a reason. If rejected, Agent A retries with a different proposal.

**Without memory**, Agent A repeats the same failing proposals on every new email — it has no idea what was rejected before.
**With memory**, Agent B's rejections are saved to Cognee. Agent A queries that history before proposing, and skips offers it knows will fail.

Same agents, same emails, same policy — one run has memory, the other doesn't.

---

This folder shows **how to add memory to AI agents using Cognee**.

The core idea is simple: wrap any async agent function with a decorator that optionally queries Cognee's knowledge graph before the function runs, and optionally saves the result back into the graph afterward.

---

## What's Inside

```
simple_agent_v2/
├── acreate_structured_output_usage_example.py   # Minimal working demo
├── agentic_context_trace/                       # The memory decorator
│   ├── agentic_root.py                          # @agentic_trace_root decorator
│   └── prompt_trace_context.py                  # AgentContextTrace data model
├── agentic_trace_persistance/                   # Saves traces back to Cognee
│   └── parsistagent_trace_pipeline.py
└── memory-vs-nomemory-agents/                   # Side-by-side comparison demo
    ├── common.py                                # Shared agent logic
    ├── memory_impl.py                           # Same agents WITH memory
    ├── nomemory_impl.py                         # Same agents WITHOUT memory
    ├── run_demo_memory.py                       # Entry point (memory)
    └── run_demo_nomemory.py                     # Entry point (no memory)
```

---

## The Core Concept: `@agentic_trace_root`

This decorator is the building block. Add it to any `async` function:

```python
from examples.demos.simple_agent_v2.agentic_context_trace import agentic_trace_root

@agentic_trace_root(
    with_memory=True,      # query Cognee before running
    save_traces=True,      # save input + output back to Cognee
    task_query="What do we know about this user?",
)
async def my_agent(payload: dict) -> dict:
    # your LLM logic here
    ...
```

**What it does at runtime:**
1. Creates an `AgentContextTrace` object capturing the function name, parameters, and decorator settings.
2. If `with_memory=True` — runs a `GRAPH_COMPLETION` search against Cognee using `task_query` and stores the result in `trace.memory_context`. Your agent function can then use this context.
3. Runs your function.
4. If `save_traces=True` — saves the trace (inputs + output) as a `DataPoint` into Cognee's graph so future calls can learn from it.

The trace is available inside the call via `get_current_agent_context_trace()` if any nested code needs to read the memory context.

### How memory gets into the prompt — step by step

This is the part that's easy to miss, because no agent function explicitly handles it. Here is the full sequence when you call a `with_memory=True` decorated function:

**Step 1 — Decorator runs before your function**

The decorator creates a trace object and immediately fires a Cognee search:

```python
# inside the decorator wrapper, before calling your function
await trace.get_memory_context(task_query)
```

This calls `cognee.search(query_type=SearchType.GRAPH_COMPLETION, top_k=20)` against the knowledge graph. Whatever Cognee finds is stored as a plain string in `trace.memory_context`. This search happens **once per function call**, not once per LLM call.

**Step 2 — Trace is placed on a ContextVar**

Python has a built-in mechanism called `ContextVar` — think of it as a thread-local sticky note that any code running in the same async context can read, without you passing it explicitly as a parameter.

```python
token = _agent_context_trace_var.set(trace)  # stick the trace on the context
result = await fn(*args, **kwargs)            # now run your function
_agent_context_trace_var.reset(token)        # clean up afterward
```

The sticky note is invisible to your function, but it's there.

**Step 3 — LLMGateway reads the sticky note on every LLM call**

Every LLM call in this codebase goes through `LLMGateway.acreate_structured_output`. Before forwarding the call to the LLM, it checks:

```python
context_trace = get_current_agent_context_trace()   # read the sticky note
if context_trace is not None and context_trace.with_memory:
    memory_context = context_trace.memory_context
    text_input = f"Additional Memory Context: {memory_context} Original input: \n\n{text_input}"
```

So the memory context is **silently prepended to `text_input`** for every LLM call made anywhere inside the decorated function — including inside nested functions like `propose_offer`. The agent functions themselves never need to know memory exists.

**Step 4 — Trace is optionally saved back to Cognee**

After your function returns, if `save_traces=True`, the full trace (inputs + output) is saved as a `DataPoint` in Cognee's graph. This is how the system builds up knowledge over time — the output of one call becomes searchable context for future calls.

**The complete timeline for one `_subagent_propose_offer` call:**

```
@agentic_trace_root wrapper starts
        │
        ├─► cognee.search("List ALL method return values ... Focus on feedbacks only")
        │         └─► returns past feedback strings → stored in trace.memory_context
        │
        ├─► _subagent_propose_offer() runs
        │         └─► propose_offer() runs
        │                   └─► LLMGateway.acreate_structured_output()
        │                             ├─► reads trace.memory_context from ContextVar
        │                             ├─► prepends it to text_input
        │                             └─► sends enriched prompt to LLM → returns ProposalOutput
        │
        └─► (save_traces=False here, so nothing is saved)
```

**Why this design?**

Separating memory retrieval from the agent functions means you can add or remove memory from any function by just changing its decorator — no changes needed inside the function itself. The `LLMGateway` acts as the single injection point, so memory enrichment is consistent everywhere automatically.

---

## Demo 1: Minimal Example

**File:** `acreate_structured_output_usage_example.py`

Adds one sentence to Cognee, then asks "What does Cognee do?" — once with memory enabled, once without.

```bash
uv run python examples/demos/simple_agent_v2/acreate_structured_output_usage_example.py
```

The memory-enabled call retrieves graph context before answering. The no-memory call answers cold.

---

## Demo 2: Memory vs No-Memory Agents

**Folder:** `memory-vs-nomemory-agents/`

A realistic multi-agent workflow: a stream of emails from students requesting software packages. Two sub-agents work together:

- **Agent A (ProposeOffer)** — reads the email and proposes a package tier.
- **Agent B (CheckEligibility)** — evaluates the proposal against policy (only `OFFER_FREE` passes).
- **Controller** — decides which agent runs next using an LLM call.

The loop retries until eligibility passes or the retry limit is hit.

**Memory version** wraps `ProposeOffer` with `@agentic_trace_root(with_memory=True)`. After the first few emails are processed and traces are saved, the proposer can query past feedback to avoid repeating rejected offers.

**No-memory version** runs the same logic but each email starts completely fresh — no access to what happened before.

```bash
# Memory-enabled run
uv run python examples/demos/simple_agent_v2/memory-vs-nomemory-agents/run_demo_memory.py

# No-memory run
uv run python examples/demos/simple_agent_v2/memory-vs-nomemory-agents/run_demo_nomemory.py
```

---

## Key Classes

| Class / Function | File | Purpose |
|---|---|---|
| `agentic_trace_root` | `agentic_root.py` | Decorator — wraps async functions with memory + tracing |
| `AgentContextTrace` | `prompt_trace_context.py` | Pydantic `DataPoint` that holds one agent call's full trace |
| `persist_agent_trace_default_pipeline` | `parsistagent_trace_pipeline.py` | Saves a trace into Cognee's graph via `add_data_points` |
| `get_current_agent_context_trace` | `agentic_root.py` | Retrieves the active trace from Python's `contextvars` |

---

## Prerequisites

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Set your LLM API key in .env
LLM_API_KEY="your_openai_api_key"
```

Cognee defaults to local SQLite + LanceDB + Kuzu — no external databases needed to run these demos.

---

## Appendix: Agent Inputs and Outputs

### Demo 1 — `acreate_structured_output_usage_example.py`

Both functions share the same downstream LLM call. The only difference is whether the decorator fetches memory context first.

#### `with_memory_method`

| | |
|---|---|
| **Decorator** | `@agentic_trace_root(with_memory=True, save_traces=True, task_query="How are agents related to Cognee")` |
| **Input** | `query: str` — default `"What does cognee do?"` |
| **Before running** | Queries Cognee graph with `task_query`, result stored in `trace.memory_context` |
| **Output** | `str` — LLM answer to the query |
| **After running** | Saves the full trace (params + answer) as a `DataPoint` in Cognee |

#### `without_memory_method`

| | |
|---|---|
| **Decorator** | `@agentic_trace_root(with_memory=False)` |
| **Input** | `query: str` — default `"What does cognee do?"` |
| **Before running** | Nothing — no memory lookup |
| **Output** | `str` — LLM answer to the query (cold, no graph context) |
| **After running** | Nothing — traces not saved |

---

### Demo 2 — `memory-vs-nomemory-agents/`

Three agents collaborate in a loop per email. The **Controller** orchestrates; the two **sub-agents** do the actual work.

#### Controller (`controller_decide_tool`)

Decides which tool to call next. Runs once per loop iteration.

**Input:**

| Field | Type | Description |
|---|---|---|
| `email_id` | `str` | ID of the email being processed |
| `loop_iteration` | `int` | Current iteration count (max 32) |
| `has_proposal` | `bool` | Whether Agent A has produced a proposal this cycle |
| `has_check` | `bool` | Whether Agent B has run on the current proposal |
| `eligibility_decision` | `"YES" \| "NO" \| None` | Latest decision from Agent B |
| `retry_cycle` | `int` | How many times the proposal was rejected so far |
| `max_retry_cycles` | `int` | Upper limit on retries (8) |

**Output — `NextToolDecision`:**

| Field | Type | Description |
|---|---|---|
| `thought` | `str` | 1-2 sentence explanation of why this tool is selected |
| `tool_name` | `ToolName` | One of `ProposeOffer`, `CheckEligibility`, `RetryOrFinish` |
| `continue_loop` | `bool` | `False` stops the email loop (only set after `YES` eligibility) |
| `stop_reason` | `str \| None` | Optional label when stopping, e.g. `"ACCEPTED"` |

**Stage rules enforced by the controller:**
- Stage `PROPOSE` → must call `ProposeOffer`
- Stage `CHECK` → must call `CheckEligibility`
- Stage `RETRY_OR_FINISH` → must call `RetryOrFinish` (resets state if `NO`, stops if `YES`)

---

#### Agent A — `ProposeOffer` (`propose_offer`)

Reads an email and proposes one package tier for the user.

**Input dict:**

| Field | Type | Description |
|---|---|---|
| `email_id` | `str` | Email identifier |
| `email_text` | `str` | Raw email content |
| `feedback_history` | `list[str]` | Eligibility feedback from all previous rejections this email |
| `proposal_history` | `list[str]` | Package names proposed in previous cycles this email |
| `rejected_offers` | `list[str]` | Set of packages already tried and rejected |

**Output — `ProposalOutput`:**

| Field | Type | Description |
|---|---|---|
| `location` | `str` | Where the user is located (extracted from email) |
| `user_category` | `str` | User type, e.g. `"student"` |
| `requested_service_tier` | `str` | What the user asked for |
| `proposed_action` | `str` | One of `OFFER_FREE`, `OFFER_STARTER`, `OFFER_PLUS`, `OFFER_PRO`, `OFFER_TEAM`, `OFFER_ENTERPRISE` |
| `rationale` | `str` | One-sentence reason for the choice (max 120 chars) |

**Memory behaviour (memory version only):**
Wrapped with `@agentic_trace_root(with_memory=True, task_query="List ALL method return values ... Focus on the feedbacks only.")`. Before the function runs, the decorator queries Cognee and stores the result in `trace.memory_context`. Then `LLMGateway.acreate_structured_output` automatically prepends it to `text_input` for every LLM call made within the decorated scope:

```python
text_input = f"Additional Memory Context: {memory_context} Original input: \n\n{text_input}"
```

`propose_offer` itself does not need to know about memory at all — the injection is transparent. It works via Python's `ContextVar`: the decorator sets the trace on the current async context, and `LLMGateway` reads it on every call. This means if a decorated function made multiple LLM calls internally, all of them would receive the memory context.

---

#### Agent B — `CheckEligibility` (`check_eligibility`)

Evaluates a proposal against policy. In this demo the policy is fixed: only `OFFER_FREE` is eligible.

**Input dict:**

| Field | Type | Description |
|---|---|---|
| `proposal` | `dict` | A serialised `ProposalOutput` from Agent A |

**Output — `EligibilityOutput`:**

| Field | Type | Description |
|---|---|---|
| `decision` | `"YES" \| "NO"` | Whether the proposed package is allowed |
| `feedback` | `str` | One original sentence explaining why — fed back into Agent A on the next retry |

**Trace behaviour (memory version only):**
Wrapped with `@agentic_trace_root(with_memory=False, save_traces=True)`. It does not query memory but **does** save its output (decision + feedback) as a trace into Cognee, making that feedback available to Agent A in subsequent emails.

---

### How memory flows across emails

```
Email N
  └─► Agent A proposes (no memory saved yet, Cognee graph is empty)
  └─► Agent B checks → decision: NO, feedback: "Only free tier is eligible"
        └─► save_traces=True → feedback saved as DataPoint in Cognee graph

Email N+1
  └─► Agent A's decorator fires a Cognee search BEFORE calling propose_offer
        └─► Cognee returns: "Only free tier is eligible" (from Email N's trace)
        └─► LLMGateway prepends this to Agent A's prompt
        └─► Agent A now knows not to propose paid tiers → proposes OFFER_FREE
  └─► Agent B checks → decision: YES → loop ends
```

Without memory, Agent A starts blind on every email and may repeat the same failing proposals indefinitely until the retry limit is hit. With memory, it learns from past eligibility feedback across emails.
