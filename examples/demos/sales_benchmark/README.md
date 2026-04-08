# Sales Benchmark: No Memory vs Context Stuffing vs Cognee Graph Memory

A quantitative benchmark comparing three approaches to AI agent memory using a simulated sales scenario.

## What it does

A sales agent sells Cognee features to 198 customer leads across 6 archetypes. Each lead's opening message is deliberately ambiguous -- it doesn't reveal which feature they actually need. The agent must figure it out.

Three modes are compared:

| Mode | How it works |
|---|---|
| **No memory** | Agent guesses based only on the customer's message |
| **Context stuffing** | All past conversation transcripts are stuffed into the prompt |
| **Graph memory** | Past conversations are stored as structured graph nodes. The `@cognee.agent_memory` decorator retrieves relevant traces and auto-injects them into the agent's prompt |

Outcomes are **deterministic** -- the right feature = win, wrong feature on last round = loss. This eliminates LLM randomness from win/loss decisions.

## Quick start

```bash
# From the repo root
python -m examples.demos.sales_benchmark.run_benchmark
```

Requires a working Cognee setup with `LLM_API_KEY` configured in `.env`.

## Results (198 leads)

| Metric | No-Memory | Context | Graph |
|---|---|---|---|
| Win rate | 90% | 91% | **97%** |
| R1 feature accuracy | 49% | 60% | **78%** |
| Avg rounds | 1.5 | 1.4 | **1.2** |
| Total tokens | 353K | 928K | **597K** |

Graph memory gets the right feature on the first pitch 78% of the time (vs 49% blind), while using 36% fewer tokens than context stuffing.

## How graph memory works

Each completed conversation is stored as a `SalesTraceNode` with edges to shared nodes:

```
SalesTrace --customer_problem--> CustomerProblem:scaling_ai
SalesTrace --features_pitched--> Feature:retrieval
SalesTrace --features_pitched--> Feature:access_control
SalesTrace --winning_feature--> Feature:access_control
SalesTrace --winning_angle--> Angle:compliance
SalesTrace --outcome--> Outcome:CLOSED_WON
```

Multiple traces sharing the same nodes create traversable clusters. When a new lead arrives, the decorator searches the graph using the customer's opening message and retrieves relevant past traces.

The key decorator usage:

```python
@cognee.agent_memory(
    with_memory=True,
    save_traces=False,
    memory_query_from_method="lead_intro",
    memory_only_context=True,
    dataset_name="sales_benchmark_traces",
)
async def sales_agent(conversation_history, lead_intro, round_num):
    return await sales_agent_turn(conversation_history, lead_intro, round_num)
```

## File structure

| File | Purpose |
|---|---|
| `run_benchmark.py` | Entry point -- runs all 3 modes and prints comparison |
| `agents.py` | Core agent logic shared by all modes |
| `catalog.py` | Cognee feature catalog (the product being sold) |
| `leads.py` | 198 customer leads across 6 archetypes |
| `models.py` | Pydantic models for LLM output + DataPoint graph nodes |
| `memory_impl.py` | Graph memory mode using `@cognee.agent_memory` |
| `context_impl.py` | Context stuffing mode |
| `nomemory_impl.py` | No-memory baseline (parallelized) |
| `metrics.py` | Metrics collection and comparison reporting |

## Configuration

- `TARGET_LEADS` in `leads.py` -- number of leads to run (default: 198)
- `MAX_ROUNDS` in `agents.py` -- max conversation rounds per lead (default: 2)
- `CONCURRENCY` in `nomemory_impl.py` -- parallel lead count for no-memory mode (default: 10)
