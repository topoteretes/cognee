# Auto-Restricted Ontology

Legend: 🟦 **PARALLEL** &nbsp;|&nbsp; 🟧 **SEQUENTIAL (lock held — only one batch at a time)** &nbsp;|&nbsp; 🟥 mutable state

---

## Approach 1 — `AUTO_RESTRICTED`
Per-chunk discovery, then a 2-call resolve under a lock. **Cost per batch: N + 2 LLM calls.**

### Single-batch flow

```mermaid
flowchart TB
    A([Batch: N chunks]):::input

    subgraph S1[" 🟦 Step 1 — Discover · PARALLEL (no lock) · N LLM calls "]
        direction LR
        C1[LLM<br/>chunk 1]
        C2[LLM<br/>chunk 2]
        Cn[LLM<br/>chunk N]
    end

    A --> S1
    S1 --> P[Prior canonical + N per-chunk drafts]
    P --> FD[Flatten + dedupe<br/>lowercase / snake_case<br/>NO LLM]
    FD --> RT[raw_types list]
    FD --> RR[raw_relations list]

    subgraph S2[" 🟧 Step 2 — Resolve · SEQUENTIAL across batches (lock held) · 2 LLM calls in parallel "]
        direction LR
        R1["LLM call A — ENTITY TYPE resolution<br/>cluster synonyms · first = canonical<br/>e.g. city / town / municipality → city<br/>drop instances (Eiffel Tower)"]
        R2["LLM call B — RELATION TYPE resolution<br/>cluster synonyms · present-tense canonical<br/>e.g. located_in / is_in / situated_in → located_in<br/>reject past-tense / one-off verbs<br/>(painted, flew, broke) → rejected list"]
    end

    RT --> R1
    RR --> R2
    R1 --> M[Pick first item of each cluster as canonical<br/>drop relation clusters with any rejected member]
    R2 --> M
    M --> U[(self.canonical)]:::state

    subgraph S3[" 🟦 Step 3 — Extract · PARALLEL · N LLM calls "]
        direction LR
        E[Restricted KnowledgeGraph<br/>+ allowlist prompt]
        E --> X1[extract<br/>chunk 1]
        E --> X2[extract<br/>chunk 2]
        E --> Xn[extract<br/>chunk N]
    end

    U --> S3
    S3 --> O([KnowledgeGraph list]):::output

    classDef input fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
    classDef state fill:#fee2e2,stroke:#991b1b,color:#000
    style S1 fill:#eff6ff,stroke:#3b82f6,color:#000
    style S2 fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#000
    style S3 fill:#eff6ff,stroke:#3b82f6,color:#000
```

### Across batches (2 batches running concurrently)

```mermaid
flowchart LR
    subgraph BA["Batch A"]
        direction LR
        A1["🟦 Discover N calls"]:::par --> A2["🟧 Resolve"]:::seq --> A3["🟦 Extract N calls"]:::par
    end
    subgraph BB["Batch B"]
        direction LR
        B1["🟦 Discover N calls"]:::par --> B2["🟧 Resolve (waits for A)"]:::seq --> B3["🟦 Extract N calls"]:::par
    end

    A1 -.->|concurrent with| B1
    A2 ==>|lock| B2

    classDef par fill:#dbeafe,stroke:#1e40af,color:#000
    classDef seq fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#000
```

> Only **resolve** is serialized. Discovery and extraction of multiple batches overlap freely.

---

## Approach 2 — `AUTO_RESTRICTED_ITERATIVE`
One discovery call per batch, prior canonical injected as context. **Cost per batch: 1 LLM call.**

### Single-batch flow

```mermaid
flowchart TB
    A([Batch: N chunks]):::input

    subgraph S1[" 🟧 Step 1 — Discover · SEQUENTIAL across batches (lock held entire step) · 1 LLM call "]
        direction TB
        U1[(Read prior canonical)]:::state
        L[LLM: single call<br/>sees N chunks + prior canonical<br/>prompt: reuse existing names]
        U2[(Update canonical)]:::state
        U1 --> L --> U2
    end

    A --> S1

    subgraph S2[" 🟦 Step 2 — Extract · PARALLEL · N LLM calls "]
        direction LR
        E[Restricted KnowledgeGraph<br/>+ allowlist prompt]
        E --> X1[extract<br/>chunk 1]
        E --> X2[extract<br/>chunk 2]
        E --> Xn[extract<br/>chunk N]
    end

    S1 --> S2
    S2 --> O([KnowledgeGraph list]):::output

    classDef input fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
    classDef state fill:#fee2e2,stroke:#991b1b,color:#000
    style S1 fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#000
    style S2 fill:#eff6ff,stroke:#3b82f6,color:#000
```

### Across batches (2 batches running concurrently)

```mermaid
flowchart LR
    subgraph BA["Batch A"]
        direction LR
        A1["🟧 Discover"]:::seq --> A2["🟦 Extract N calls"]:::par
    end
    subgraph BB["Batch B"]
        direction LR
        B1["🟧 Discover (waits for A)"]:::seq --> B2["🟦 Extract N calls"]:::par
    end

    A1 ==>|lock| B1

    classDef par fill:#dbeafe,stroke:#1e40af,color:#000
    classDef seq fill:#fde68a,stroke:#d97706,stroke-width:2px,color:#000
```

> The **entire discovery** is serialized. Batch B can't start discovery until Batch A finishes its single LLM call. Extraction across batches still overlaps.
