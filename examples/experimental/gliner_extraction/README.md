# GLiNER2 LLM-free extraction for cognee (experimental)

Replace cognify's LLM stages with [GLiNER2](https://github.com/fastino-ai/GLiNER2) —
a 205M-parameter, CPU-first encoder model that does entity extraction, relation
extraction, classification, and structured extraction in a single forward pass.
No API calls, no rate limits, deterministic, ~2 GB RAM.

Tracking issue: COG-6188.

## Install

```bash
pip install cognee[gliner]        # or: pip install gliner2[local]
```

## Two integration levels

### 1. Extraction hook only (rest of cognify unchanged)

`extract_graph_from_data` already accepts a `calculate_chunk_graphs` callable
that replaces LLM extraction. No cognee changes needed:

```python
from gliner2 import GLiNER2
from gliner_graph_extractor import gliner_chunk_graph_calculator

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
await cognee.cognify(
    datasets=["my_dataset"],
    calculate_chunk_graphs=gliner_chunk_graph_calculator(extractor),
)
```

LLM still writes chunk summaries (they run in parallel with extraction).

### 2. Whole-cognify swap — zero LLM calls

`gliner_cognify()` mirrors cognify's default task list with both LLM stages
replaced. One batched GLiNER forward pass per chunk batch feeds **both** the
knowledge graph and extractive `TextSummary` datapoints:

```python
from gliner_cognify import gliner_cognify

await cognee.add(data, dataset_name="my_dataset")
await gliner_cognify(datasets=["my_dataset"], extractor=extractor)
```

| Stage | Default cognify | gliner_cognify |
|---|---|---|
| classify_documents / chunking | no LLM | unchanged |
| graph extraction | LLM (Instructor) | GLiNER2 entities + relations |
| chunk summaries | LLM | GLiNER2 extractive summaries |
| add_data_points | embeddings | unchanged (embeddings only) |

Note: GLiNER2 is encoder-only — summaries are *extractive* (most entity-dense
sentences + an entity digest), not abstractive. Answer generation at query
time (`GRAPH_COMPLETION`) still needs an LLM; `CHUNKS`/`SUMMARIES`/`CHUNKS_LEXICAL`
search stays fully LLM-free.

## Where the schema comes from (OWL + LLM discovery)

GLiNER is schema-driven: it only extracts the types you name. Instead of the
hardcoded defaults, derive the schema from an OWL ontology, and optionally
let ONE per-dataset LLM call propose types the ontology doesn't cover:

```python
from ontology_schema import gliner_schema_from_ontology, discover_additional_types

# OWL classes -> entity types, OWL object properties -> relation types
# (rdfs:label as name, rdfs:comment as the description GLiNER matches on)
entity_types, relation_types = gliner_schema_from_ontology("military.owl")

# Optional: one LLM call over a sample proposes labels NOT in the ontology
extra_e, extra_r = await discover_additional_types(sample_texts, entity_types, relation_types)
entity_types |= extra_e
relation_types |= extra_r

await gliner_cognify(datasets=["my_dataset"], extractor=extractor,
                     entity_types=entity_types, relation_types=relation_types)
```

This keeps open-endedness at per-dataset cost while extraction itself stays
LLM-free per chunk. Additionally, when `ONTOLOGY_FILE_PATH` is set, cognee's
existing ontology resolver canonicalizes and enriches the extracted graph
against the same OWL — that mechanism applies to GLiNER output unchanged.

## Files

- `gliner_graph_extractor.py` — the `calculate_chunk_graphs` hook; maps GLiNER
  output (batched, span-aware) onto cognee's `KnowledgeGraph` model.
- `gliner_cognify.py` — the full pipeline swap (`gliner_extract_and_summarize`
  task + `gliner_cognify()` runner).
- `ontology_schema.py` — OWL → GLiNER schema derivation + per-dataset LLM
  discovery of types the ontology doesn't cover.
- `demo_gliner_cognify.py` — end-to-end demo that runs with a deliberately
  broken LLM key as proof of zero LLM calls.
- `benchmark_book.py` — whole-book benchmark (`python benchmark_book.py book.pdf`).

## Measured performance (M-series laptop, CPU only)

War & Peace, 2,043-page PDF, chunk_size=512:

| Metric | Value |
|---|---|
| Total wall clock (add + cognify) | ~22 min |
| Chunks / summaries | 3,140 / 3,140 |
| Unique entities | 6,933 |
| Edges | 48,159 |
| GLiNER share of cognify time | ~54% (rest: embedding round-trips + graph writes) |
| Marginal cost | ~$0.10 (embeddings only) |
| GLiNER throughput | ~30 chunks/s per forward pass; 70 ms/text sequential, 23 ms/text batched (32 short texts) |

Scaling levers: worker processes (~2 GB each, linear), CUDA GPU
(`map_location="cuda"`, `quantize=True`), local embeddings, larger pipeline
batch delivery. Apple MPS tested slower than batched CPU for short inputs.

## Trade-offs vs LLM extraction

- Fixed label set (entity/relation types are configured, not open-ended);
  descriptions improve accuracy but there's no free-form ontology discovery.
- No abstractive summaries.
- In exchange: ~2 orders of magnitude higher per-worker throughput,
  deterministic output, zero token cost, fully local/private option.
