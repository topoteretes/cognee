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

## The default pipeline: cached ontology + GLiNER

`gliner_cognify()` with no explicit schema runs the full architecture by
default (`auto_schema=True`):

```
                 COGNEE DATASET
                       │
               cached ontology?
                  /           \
               yes             no
                │               │
          (no LLM call)   ONE LLM ontology-
                │         discovery call over
                │         the first batch
                └───────┬───────┘
                        ▼
                  ontology vN  (versioned JSON per dataset)
                        ▼
                    GLiNER2  ← all chunks, batched, LLM-free
                        ▼
          low-density batch? → one discovery call → ontology vN+1
```

- First ingestion of a dataset: one LLM call invents the ontology from the
  data (verified: a medical sample produced `clinical_event`, `diagnosis`,
  `medication`, `healthcare_facility`, …), saved as
  `.gliner_schema_cache/<dataset>.json` v1.
- Every later ingestion: the cached ontology loads with **zero LLM calls**.
- Density-triggered expansion (below) bumps the cached version, so the
  ontology evolves while the LLM stays out of the hot path.
- No LLM configured at all: falls back to generic default labels with a loud
  warning (or pass `entity_types`/`relation_types` explicitly).

The discovery call is constrained (read samples, emit 10–30 labels), so a
small local model is enough. Verified with **qwen3:4b via Ollama** (~2.5 GB,
`ollama pull qwen3:4b`):

```bash
LLM_PROVIDER=ollama LLM_MODEL=qwen3:4b \
LLM_ENDPOINT=http://localhost:11434/v1 LLM_API_KEY=ollama \
python your_ingestion.py
```

That makes the whole pipeline local: qwen3:4b invents the schema once
(~2 min — it is a thinking model; once per dataset, not per document),
GLiNER2 extracts everything, no document leaves the machine.

## Model downloads

The GLiNER2 checkpoint (~450 MB for base) downloads from HuggingFace on
first use and is cached under `~/.cache/huggingface`. `gliner_cognify` and
`GLiNERWorkerPool` call `ensure_model_available()` up front, which prints a
visible notice before downloading — and pre-downloads in the parent process
so N workers never race the download. Ollama models are pulled explicitly
(`ollama pull qwen3:4b`, with its own progress output).

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

### Adaptive resampling (per-batch density monitoring)

A single up-front sample can under-cover heterogeneous datasets. The
`AdaptiveSchemaTuner` uses GLiNER's own output as the coverage signal:
it tracks entity density (mentions per 1k chars) per batch, and when a
batch drops below `trigger_ratio` x the running average, it samples that
batch's lowest-density chunks, makes ONE discovery LLM call, merges the new
labels, and re-extracts the batch with the expanded schema. LLM cost is
capped by `max_discoveries`; `report()` gives a post-run coverage summary.

```python
from adaptive_schema import AdaptiveSchemaTuner

tuner = AdaptiveSchemaTuner(entity_types, relation_types)
await gliner_cognify(datasets=["my_dataset"], extractor=extractor, schema_tuner=tuner)
print(tuner.report())   # densities per batch, what was discovered and when
```

Verified on synthetic two-domain data: military batches ran at density ~47,
a medical batch scored 0.0 under the military schema, triggered one
discovery call (+8 entity, +8 relation types), and the re-extraction found
`medication`/`dosage`/`condition` spans immediately.

## Files

- `gliner_graph_extractor.py` — the `calculate_chunk_graphs` hook; maps GLiNER
  output (batched, span-aware) onto cognee's `KnowledgeGraph` model.
- `gliner_cognify.py` — the full pipeline swap (`gliner_extract_and_summarize`
  task + `gliner_cognify()` runner).
- `ontology_schema.py` — OWL → GLiNER schema derivation + per-dataset LLM
  discovery of types the ontology doesn't cover.
- `adaptive_schema.py` — per-batch entity-density monitoring with
  density-triggered schema expansion and re-extraction.
- `demo_gliner_cognify.py` — end-to-end demo that runs with a deliberately
  broken LLM key as proof of zero LLM calls.
- `benchmark_book.py` — whole-book benchmark (`python benchmark_book.py book.pdf`).

## Measured performance (M-series laptop, CPU only)

War & Peace, 2,043-page PDF, chunk_size=512. Baseline = single process, no
packing, serial batches. Optimized = `workers=3` + packing + overlapped
storage (`perf:` commit):

| Metric | Baseline | + workers/pack/overlap | + storage_depth=2 |
|---|---|---|---|
| cognify wall clock | 1,290 s (21.5 min) | 911 s (15.2 min) | **794 s (13.2 min), 1.63x** |
| batch cycle | 12.7 s (serial) | 9.0 s (storage-bound) | ~8.0 s |
| Chunks / summaries | 3,140 / 3,140 | 3,140 / 3,140 | 3,140 / 3,140 |
| Unique entities | 6,933 | **8,151** (+18% — packing gives GLiNER more context) | 8,151 |
| Edges | 48,159 | 50,467 | 50,471 |
| Marginal cost | ~$0.10 (embeddings only) | same | same |

Config for the best run: `workers=3, storage_depth=2, chunks_per_batch=256`,
OpenAI embeddings.

Caveat learned the hard way: chunk sizes are NOT comparable across embedding
providers. cognee's chunker counts tokens word-by-word, and `chunk_by_word`
attaches the trailing space to each word. The TikToken adapter counts that
trailing space as a token (~2 counted tokens per word), while HuggingFace
tokenizers drop whitespace (~1 per word) — on whole text the two tokenizers
agree within 0.2%, so the ~2x gap is purely an artifact of per-word counting.
Net effect: with OpenAI embeddings a `chunk_size=512` chunk closes at ~250
real tokens; with `HUGGINGFACE_TOKENIZER` set (Ollama) the same setting
yields ~2x bigger chunks, which broke packing (chunks exceeded
`pack_target_chars`) and degraded GLiNER recall in our Ollama run. Halve
`chunk_size` (or fix the upstream counting) when using HF tokenizers.

After these changes extraction is fully hidden behind storage: the critical
path is now `add_data_points` (OpenAI embedding round-trips + single-threaded
graph writes, slowed further by CPU contention with the workers). The next
meaningful levers are therefore local/in-process embeddings and cheaper
summary indexing — not more extraction speed. On GPU hardware
(`map_location="cuda"`, `quantize=True`, FlashDeberta) extraction itself has
another 5–20x of headroom. Apple MPS tested slower than batched CPU.

## Trade-offs vs LLM extraction

- Fixed label set (entity/relation types are configured, not open-ended);
  descriptions improve accuracy but there's no free-form ontology discovery.
- No abstractive summaries.
- In exchange: ~2 orders of magnitude higher per-worker throughput,
  deterministic output, zero token cost, fully local/private option.
