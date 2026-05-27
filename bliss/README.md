# Theory Meets Practice #4: Explainable Scientific Discovery with AI Memory

Teaching demo of how we used Cognee for biomedical hypothesis work at Bayer. The research pipeline ingests PubMed literature, extracts entities and hypotheses (each split into premise and conclusion), stores them in a knowledge graph, assesses new candidates — mainly **credence** and **novelty** — and generates further hypotheses from graph communities and frontier regions.

This folder covers ingest, enrichment, and scoring only: four synthetic papers and three Cognee steps (**remember → improve → retrieve**). Hypothesis generation from the graph is a separate mechanism in the full pipeline and is not implemented here. No MeSH or full metric stack.

| Research pipeline | Demo |
|-------------------|------|
| Ingest papers → entities + hypotheses | `remember()` → `Paper`, `Entity`, `Hypothesis` |
| Enrich graph (decompositions, vector index) | `improve()` → `HypothesisPremise` / `HypothesisConclusion` |
| Score a candidate (credence, novelty, …) | `bliss_retriever.py` + `scores.py` (feasibility ≈ credence, novelty) |
| Generate hypotheses from graph structure | not covered |

## Prerequisites

Copy `.env.template` to `.env` and set `LLM_API_KEY` (embeddings default to OpenAI if not configured separately).

Run from the repo root:

```bash
uv run python bliss/bliss_run.py
```

Use `--no-verbose` for completion output only.

## Pipeline

```
papers.json
    │
    ▼ remember()          bliss_remember.py
    │  Paper → Entity, Hypothesis nodes
    ▼
    improve()             bliss_improve.py
    │  Hypothesis → HypothesisPremise, HypothesisConclusion
    │  edges: has_premise, has_conclusion
    ▼
    bliss_retriever.py + scores.py
       decompose candidate → vector search premises → score → explain
```

| Step | Cognee API | What happens |
|------|------------|--------------|
| Remember | `cognee.remember()` | Four toy papers land in `main_dataset`; LLM extracts `Paper`, `Entity`, `Hypothesis` via custom `Paper` graph model. |
| Improve | `cognee.improve()` | Custom memify tasks read all `Hypothesis` nodes, LLM-decompose each, store premise/conclusion nodes and wire edges. |
| Retrieve | custom retriever | Decompose the query, find similar stored premises, compute feasibility/novelty, LLM explains the scores. |

## Files

| File | Role |
|------|------|
| `bliss_run.py` | End-to-end demo: forget → remember → improve → retrieve |
| `bliss_remember.py` | `Paper` / `Entity` / `Hypothesis` models + `remember()` |
| `bliss_improve.py` | Extraction + enrichment tasks for `improve()` |
| `decompose.py` | Shared LLM premise/conclusion decomposition |
| `scores.py` | Premise-anchored feasibility and novelty (numpy) |
| `bliss_retriever.py` | Fetch, context, and completion over scored matches |
| `data/papers.json` | Four synthetic papers (Problems A/B, modules alpha–delta) |

## Scores

**Feasibility** is the demo name for Bayer **credence**; both live in `scores.py`.

### Setup (same for both scores)

1. Decompose the candidate hypothesis into premise text and conclusion text (LLM).
2. Embed both strings and normalize to unit vectors: `p` (premise), `c` (conclusion).
3. Search the premise vector index for the top-k stored premises most similar to `p`.
4. For each hit, load the paired conclusion from the graph → match pairs `(p_i, c_i)`.
5. Embed and normalize each matched premise and conclusion text → vectors `p_i`, `c_i`.

If step 3 finds nothing, both scores are `NaN`.

### Feasibility

*Question: among premises like mine, do the literature conclusions agree with my conclusion?*

1. For each match `i`, score how similar the stored premise is to the candidate: `weight_i = dot(p, p_i)`.
2. Form a single “typical conclusion” vector by weighting each `c_i` by its premise similarity and summing.
3. Normalize that sum to unit length.
4. Compare it to the candidate conclusion: `feasibility = dot(that_vector, c)`.

```
for each match i:
    weight_i = dot(p, p_i)
weighted_conclusion = sum(weight_i * c_i for each i)
feasibility = dot(normalize(weighted_conclusion), c)
```

High → your conclusion aligns with what similar premises usually imply. Low → similar premises in memory point elsewhere.

### Novelty

*Question: how different am I from the closest thing already in memory?*

1. For each match `i`, measure premise distance: `1 - dot(p, p_i)` (0 = identical direction, 2 = opposite).
2. Same for conclusion: `1 - dot(c, c_i)`.
3. Combine into one distance per match: average of premise and conclusion distance.
4. Take the **maximum** over matches → `novelty`.

```
for each match i:
    premise_dist = 1 - dot(p, p_i)
    conclusion_dist = 1 - dot(c, c_i)
    distance_i = 0.5 * premise_dist + 0.5 * conclusion_dist
novelty = max(distance_i)
```

High → even your nearest neighbor in memory is still far in premise, conclusion, or both. Low → you closely resemble an existing pair.

`bliss_retriever.py` passes the scores and matched pairs to an LLM for a short explanation.

## Run steps individually

```bash
uv run python bliss/bliss_remember.py
uv run python bliss/bliss_improve.py
uv run python bliss/bliss_retriever.py
uv run python bliss/scores.py
```

## Default candidate

`bliss_run.py` scores:

> Problem A is solved by routing inputs through beta, then gamma, then delta.

Against four stored papers — two claim Problem A needs only alpha, two describe Problem B pipelines — so feasibility and novelty should reflect overlap with the alpha-centric stored hypotheses.
