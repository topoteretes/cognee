# Cognify and Search Lifecycle Architecture

This document provides visual pipeline flowcharts and a technical reference for Cognee's end-to-end memory lifecycle: how raw input transitions into knowledge graph structures and vector embeddings during `cognify()`, how hybrid retrieval traverses both during `search()` / `recall()`, and how pruning methods (`prune_data()`, `prune_system()`) manage data lifecycle and cleanup.

Source of truth:
- Cognify: `cognee/api/v1/cognify/cognify.py`
- Search & Recall: `cognee/api/v1/search/search.py` and `cognee/api/v1/recall/recall.py`
- Prune & Deletion: `cognee/modules/data/deletion/`

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion (add / remember)"]
        RawInput["Raw Data (Text, Documents, Connectors)"] --> Ingest["cognee.add() / cognee.remember()"]
        Ingest --> RawStorage[("File & Metadata Storage")]
    end

    subgraph Cognify ["2. Cognify Pipeline (cognify)"]
        RawStorage --> Classify["Classify & Ingest Documents"]
        Classify --> Chunking["Text Chunking (TextChunker)"]
        Chunking --> Extractor{"Graph Extractor"}
        Extractor -->|LLM Provider| LLMExtract["LLM Entity & Edge Extraction"]
        Extractor -->|Local / No-LLM| GLiNERExtract["GLiNER Local Extraction"]
        LLMExtract --> GraphCons["Graph Node & Edge Resolution"]
        GLiNERExtract --> GraphCons
        GraphCons --> GraphDB[("Graph DB (Kùzu / Neo4j)")]
        Chunking --> EmbedChunks["Vector Embedding (Chunks & Triplets)"]
        EmbedChunks --> VectorDB[("Vector DB (LanceDB / PGVector)")]
        GraphCons -.-> Improve["Post-Cognify: improve() (Index & Summaries)"]
    end

    subgraph Retrieval ["3. Retrieval & Search (recall / search)"]
        UserQuery["Query / User Prompt"] --> QueryRouter{"Query Router"}
        QueryRouter -->|auto_route| RouteDecision["Rule-based Strategy (HYBRID, GRAPH, CHUNKS...)"]
        QueryRouter -->|query_type| ExplicitRoute["Pinned SearchType"]
        RouteDecision --> HybridSearch["Hybrid Search Executor"]
        ExplicitRoute --> HybridSearch
        HybridSearch -->|Vector Search| VectorDB
        HybridSearch -->|Graph Traversal| GraphDB
        HybridSearch --> ContextAssembly["Context Assembly & Citations"]
        ContextAssembly --> Synthesis["LLM Response Synthesis / response_model"]
    end

    subgraph PruneLifecycle ["4. Data Pruning & Lifecycle"]
        PruneData["cognee.prune.prune_data()"] -->|Wipe| RawStorage
        PruneSystem["cognee.prune.prune_system()"] -->|Destroy| GraphDB
        PruneSystem -->|Destroy| VectorDB
        PruneSystem -->|Drop| RawStorage
    end
```

---

## 2. Ingestion & Cognify Lifecycle

`cognify()` transforms raw documents into two synchronized representations: a **Knowledge Graph** (entities, relationships, and attributes) and **Vector Indices** (dense chunk embeddings and triplet embeddings).

```mermaid
sequenceDiagram
    autonumber
    participant App as Application / User
    participant Pipeline as Cognify Pipeline
    participant Chunker as TextChunker
    participant Extractor as Extractor (LLM / GLiNER)
    participant Graph as Graph Database
    participant Vector as Vector Database
    participant Relational as Relational / Metadata

    App->>Pipeline: cognify(datasets=[...])
    Pipeline->>Relational: Fetch unindexed Data items & classify
    Pipeline->>Chunker: Split documents into bounded token chunks
    Chunker-->>Pipeline: Yield Chunks (with provenance offsets)

    par Parallel Processing
        Pipeline->>Extractor: Extract entities, relations, dates, and summaries
        Extractor-->>Pipeline: Extracted KnowledgeGraph (Nodes & Edges)
        Pipeline->>Graph: Upsert DataPoint nodes, typed edges, and timestamps
    and
        Pipeline->>Vector: Generate embeddings for text chunks & triplets
        Pipeline->>Vector: Upsert embeddings into collections (index_chunks_for_search)
    end

    Pipeline->>Relational: Record pipeline run status and provenance
    Pipeline-->>App: CognifyResult (nodes count, edges count, execution spans)
```

### Key Stages in Cognify

1. **Document Classification & Chunking:**
   - Unprocessed documents from `cognee.add()` are classified by MIME type.
   - `TextChunker` breaks text into chunks sized according to model limits and configured overlap parameters.
2. **Entity & Relation Extraction:**
   - Either through configured LLMs (structured output prompt) or local zero-shot NER models (GLiNER).
   - Resolves canonical names and merges identities across chunks.
   - Extracts explicit temporal markers as `Timestamp` nodes for temporal reasoning.
3. **Dual Persistence:**
   - **Graph Store:** Nodes (entities, concepts) and Edges (relations) are written to Kùzu, Neo4j, or configured graph backend.
   - **Vector Store:** Dense embeddings of raw chunks and graph triplets are stored in LanceDB, PGVector, or Turso.
4. **Enrichment (`improve()`):**
   - Optionally computes community summaries and populates dataset routing indexes.

---

## 3. Search and Retrieval Lifecycle

`recall()` (or low-level `search()`) executes hybrid retrieval across vector and graph storage to assemble comprehensive context.

```mermaid
flowchart LR
    subgraph QueryFlow ["Query Routing & Execution"]
        Q["User Query"] --> QR{"Auto Query Router"}
        QR -->|Quoted phrase| C_LEX["CHUNKS_LEXICAL"]
        QR -->|Where is X defined?| CODE["CODE / CodeRetriever"]
        QR -->|How many / counts| BROAD["BROAD"]
        QR -->|Absolute date / in 2024| TEMP["TEMPORAL"]
        QR -->|General question| HC["HYBRID_COMPLETION"]
    end

    subgraph RetrievalStage ["Hybrid Search Execution"]
        HC --> VS["Vector Similarity Search"]
        HC --> GT["Graph Subgraph Traversal"]
        VS --> VecHits["Top-K Matching Chunks"]
        GT --> GraphHits["Entity Neighborhoods & Paths"]
    end

    subgraph ContextFusion ["Fusion & Generation"]
        VecHits --> Fuse["Context Assembly & Reranking"]
        GraphHits --> Fuse
        Fuse --> LLM["LLM Answer Synthesis"]
        LLM --> Out["SearchResult / Structured Model"]
    end
```

### Retrieval Steps

1. **Query Routing:**
   - If `query_type` is omitted and `auto_route=True`, regex heuristics pick the best retrieval strategy without LLM cost.
2. **Dataset Routing:**
   - In multi-dataset setups, queries are matched against dataset routing index roots, avoiding full-database scans on unrelated memory.
3. **Hybrid Lookup:**
   - Vector search locates closest text chunks.
   - Graph engine traverses relationships connected to matching entity nodes.
4. **Context Fusion:**
   - Chunks, graph edges, and session memories are merged, formatted with citations, and handed to the completion model.

---

## 4. Data Lifecycle and Pruning APIs

Cognee provides distinct pruning methods depending on the required level of data deletion.

> [!WARNING]
> Pruning operations permanently remove data and cannot be undone. Always ensure necessary backups exist before invoking system-wide pruning methods.

### Comparison of Deletion Methods

| Method | Scope | Affected Stores | Use Case |
|---|---|---|---|
| `cognee.forget(dataset=..., document=...)` | Document or Dataset | Graph nodes, vector chunks, metadata | Safe user/session data removal |
| `cognee.prune.prune_data()` | Raw data files & caches | File storage root, cache engines | Free up disk space without destroying graph memory |
| `cognee.prune.prune_system()` | Entire system | Graph DB, Vector DB, Relational DB, Caches | Complete development / testing environment reset |

### Programmatic Pruning Examples

#### 1. Prune Data Storage
Removes uploaded raw files and clears memory caches, leaving the indexed knowledge graph and vector stores intact:

```python
import cognee

# Clear uploaded file assets and storage caches
await cognee.prune.prune_data()
```

#### 2. Prune System (Destructive Reset)
Wipes all graph databases, vector collections, relational tables, and cached data:

```python
import cognee

# Complete wipe of all cognee storage engines
await cognee.prune.prune_system(
    graph=True,
    vector=True,
    metadata=True,
    cache=True,
)
```

### CLI Data Lifecycle Commands

Cognee CLI exposes pruning and deletion operations from the terminal:

```bash
# Forget all memory across all datasets
cognee-cli forget --all

# Forget a specific dataset
cognee-cli forget -d my_dataset

# Inspect dataset and memory status
cognee-cli datasets list
```
