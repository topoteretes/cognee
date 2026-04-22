# Shared-Memory Cross-Domain Demo

Three agents analyze three apparently-distinct observational domains that
secretly reduce to the same mathematical structure (a random walk with
Gaussian increments, variance linear in time, independent steps). With a
shared cognee graph, later agents retrieve earlier agents' findings via
**structural tag similarity** — despite zero natural-language overlap
between their corpora — and cite those memory passages alongside their
own corpus IDs.

## Claim

The **SHARED** arm (one shared cognee dataset; each agent retrieves
prior findings via `cognee.search`) is compared against:

- **ISOLATED** — per-agent dataset; no prior context.
- **CONCAT** — per-agent dataset; prior agents' findings passed as raw
  concatenated text in the prompt.

The real claim is **SHARED ≥ CONCAT on correctness at a much smaller
token budget**. A SHARED prior-context passage is typically a ~250-char
graph-synthesized summary; a CONCAT prior-context block is ~2000 chars
of raw findings text. If SHARED matches or beats CONCAT, structured
retrieval demonstrably compresses prior work.

## Layout

```
shared_memory_crosdomain/
├── README.md
├── vocabulary.py      # 15 fixed structural tags (shared reduction vocabulary)
├── data_models.py     # StructuralFinding(DataPoint), tags indexed for embedding
├── corpora/
│   ├── agent_1_particle.md   # masked snippets — Type-P grains in Fluid-W
│   ├── agent_2_material.md   # masked snippets — Substance-S in Medium-M
│   └── agent_3_market.md     # masked snippets — Asset-Q across market ticks
├── agents.py          # run_agent(): retrieve → read corpus → derive → persist → answer
├── rubric.py          # blind LLM judge: correctness 0..8 + binary abstraction
├── run_demo.py        # orchestrator: one arm × N seeds → results/*.json
└── metrics.py         # aggregate results/*.json into summary tables
```

## Running

### Environment

`run_demo.py` loads `.env` from `$COGNEE_ENV_PATH` (default
`/Users/veljko/coding/cognee/.env` — override if you run this elsewhere).
Point it at any cognee-compatible `.env` with an LLM API key and
embedding provider configured.

Two env defaults are applied at import time:

- `EMBEDDING_DIMENSIONS=1536` — required for `text-embedding-3-small`
  (cognee's default is `3072` for `text-embedding-3-large`).
- `COGNEE_SKIP_CONNECTION_TEST=true` — cognee's 30 s embedding probe is
  too tight for some OpenAI endpoints; the actual embedding calls work.

cognee state is isolated under `$SHARED_MEM_DEMO_STATE` (default
`~/.cache/shared_mem_demo/`) so this demo does **not** touch the user's
other cognee projects.

### Single seeds (smoke test)

One seed per arm, ~2 minutes per seed:

```bash
python examples/guides/shared_memory_crosdomain/run_demo.py --arm shared   --seeds 1
python examples/guides/shared_memory_crosdomain/run_demo.py --arm isolated --seeds 1
python examples/guides/shared_memory_crosdomain/run_demo.py --arm concat   --seeds 1
python examples/guides/shared_memory_crosdomain/metrics.py
```

### Full sweep

5 seeds × 3 arms = 15 runs, ~45 minutes on `gpt-oss-120b` via Baseten
plus `text-embedding-3-small` via OpenAI:

```bash
for arm in shared isolated concat; do
  python examples/guides/shared_memory_crosdomain/run_demo.py --arm "$arm" --seeds 5
done
python examples/guides/shared_memory_crosdomain/metrics.py
```

Per-run JSONs land under `results/` (gitignored). `metrics.py` reads
the directory and prints four panels to stdout.

## Output panels

```
Correctness (0-8) — mean ± stddev across seeds
  arm        A1             A2             A3            n

Secondary metrics — mean across seeds
  arm        metric                      A1       A2       A3
             LLM calls                   …        …        …
             mem cites (findings)        …        …        …
             mem cites (final)           …        …        …
             prior-context chars         …        …        …

A3 unifies-domains rate (final answer)
  arm        hits/n  (pct%)

Paired correctness deltas — SHARED minus baseline (same seed)
  SHARED − ISOLATED  A1 Δ  A2 Δ  A3 Δ  (n=…)
  SHARED − CONCAT    A1 Δ  A2 Δ  A3 Δ  (n=…)
```

## How the mechanism works

1. **Shared tag vocabulary** (15 descriptors in `vocabulary.py`) is
   given to every agent verbatim. Agents tag findings using only these
   strings.
2. `StructuralFinding(DataPoint)` declares `description` and
   `structural_tags` as `index_fields`, so both are embedded by the
   vector engine. The tag string is stored comma-joined (cognee's
   vector indexers only accept string index fields).
3. Each finding is persisted via `cognee.add(..., node_set=[f"agent_{i}"])`
   + `cognee.cognify(graph_model=StructuralFinding, custom_prompt=...)`.
   The `node_set` propagates to `source_node_set` on every extracted
   node, making per-agent provenance recoverable.
4. In the SHARED arm, Agent 2 queries the graph with a natural-language
   prompt that embeds the tag vocabulary. Agent 1's findings surface
   via **tag embedding** overlap — not natural-language overlap — and
   are cited as `[mem:0]`, `[mem:1]`, … alongside corpus IDs.

## Known caveats

1. **Obfuscation is imperfect.** Baseline LLMs identify the underlying
   processes (Brownian motion / diffusion / geometric Brownian motion)
   from any corpus alone, even with masked domain nouns and qualitative
   statistical descriptors. The **citation requirement** is the real
   defense: every claim must cite a corpus snippet or memory passage,
   and uncited claims are rejected at validation time. The cross-
   pollination signal is measured on actual citation structure, not on
   whether pretraining "knew" the answer.

2. **Agent 3 is not explicitly asked to unify the three domains.** The
   `score_abstraction` rubric measures whether A3's answer happens to
   unify; the prompt does not instruct it to. A SHARED-arm A3 that
   unifies is a real cross-pollination signal; an absence of
   unification is a ceiling effect rather than a bug.

3. **Tokens are not tracked.** `metrics.py` reports LLM call counts, not
   token totals. Adding token accounting would require a litellm
   callback; the per-arm *relative* cost can be inferred from call
   counts (SHARED adds one search call per agent) and prior-context
   length.

4. **Seeds are replicas, not RNG seeds.** LLM temperature is not pinned;
   each seed is an independent replica. `--seeds 5` is the minimum for
   a useful mean ± stddev — bump it for tighter error bars.

## Headline results

Populated after the full sweep runs. See `results/` for per-run
artifacts (gitignored).
