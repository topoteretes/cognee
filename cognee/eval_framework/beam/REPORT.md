**Note.** This is a preliminary report based on the current BEAM setup and available results. Additional details will follow.

# Cognee on BEAM - Preliminary Report

## 1. Introduction

This report documents how we ran BEAM on open-source Cognee: standard ingestion and retrieval primitives, BEAM-specific formatting, and BEAM's official evaluation pipeline. It complements the companion blog with implementation and reproducibility details.

We developed the methodology on the 100K benchmark before evaluating a single 10M conversation. Where relevant, we distinguish observations that are well supported by repeated experiments from those based on the single 10M evaluation.

## 2. BEAM

### 2.1 Benchmark Summary

BEAM is a benchmark for evaluating long-context conversational memory systems. It consists of synthetic multi-session user-assistant conversations, probing questions, gold answers, and rubric-based evaluation performed by an LLM judge.

Questions target ten memory abilities: information extraction, multi-hop reasoning, knowledge updates, temporal reasoning, summarization, preference following, abstention, contradiction resolution, event ordering, and instruction following. BEAM documentation also refers to multi-hop reasoning as multi-session reasoning.

This report evaluates Cognee on the **100K** and **10M** context scales. Unless otherwise noted, reported 100K results correspond to the **second 100K conversation** in the dataset, while 10M results correspond to the **first 10M conversation**.

### 2.2 Reference Implementation

This evaluation follows the publicly available BEAM implementation.

- Paper: https://arxiv.org/abs/2510.27246
- Repository: https://github.com/mohammadtavakoli78/BEAM
- Datasets:
  - https://huggingface.co/datasets/Mohammadta/BEAM
  - https://huggingface.co/datasets/Mohammadta/BEAM-10M
- Evaluation scripts: upstream repository (linked above)

## 3. Methodology

We evaluated Cognee on BEAM in two stages. First, each conversation was ingested into Cognee memory. Second, BEAM probing questions were answered using Cognee retrievers and scored with BEAM's official evaluation pipeline.

Ingestion used open-source Cognee memory primitives. We did not build a benchmark-specific graph schema, ontology, or memory architecture. The BEAM-specific code handles dataset formatting, turn-preserving JSON-list files, ingestion orchestration, and evaluation artifacts. Retrieval, prompting, and routing were tuned for BEAM comparability, as described below.

### 3.1 Experimental Design

At **100K**, we tuned on two conversations, checked transfer on two others, and reported on the **second 100K conversation** in the dataset, which was held out from that tune-and-transfer setup.

At **10M**, we ingested a single conversation once. We started from settings selected during 100K work, adjusted retriever parameters after those settings underperformed at 10M, and independently reran strong routing configurations to reduce evaluation variance. We did not perform a second full 10M ingestion.

Multi-turn agentic retrieval improved scores further, but we exclude it from the reported benchmark figures. It changes the retrieval setting substantially and is outside the scope of the main comparison in this report.

### 3.2 Data Ingestion

**Source data.** Conversations come from [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) and [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M). Ingestion used standard Cognee **`add`** and **`cognify`** primitives. Graph and vector extraction follow the normal **`cognify`** path. A single fixed model was used for ingestion across all reported runs; the specific model will be reported in a later revision.

**Document and chunk mapping.** In BEAM, conversations below 10M are split into **batches**, where each batch is one continuous user-assistant dialogue. At 10M, the conversation is split into **plans**, and each plan is split into batches in the same way. We mapped each batch to one Cognee document and each user-assistant turn to one chunk. Turn order, timestamps, and session metadata were preserved where BEAM provides them. BEAM batch boundaries were treated as sufficient local units; we did not use overlap-based chunking. The JSON-list chunker used for this path is intentionally minimal: preprocessing already defines the turn units, and ingestion reads one list item as one chunk.

**Run mode.** The reported 10M ingestion was executed with Cognee distributed mode on Modal. The current public code includes the same preprocessing representation and a local ingestion path. The BEAM-specific distributed orchestration used for the reported 10M run is not fully organized in this version, but the JSON-list session representation can be wired into Cognee's standard Modal distributed execution.

**Preprocessing.** Less than **0.2%** of turns required cleanup before ingestion, mostly long assistant turns. Noise included repeated random strings, nonsensical text, and leaked conversation-plan text. Those turns were passed through a small LLM compression step using a minimal, non-descript prompt with no benchmark-specific tuning.

Cleanup addressed obvious corpus noise only. We did not remove turns because Cognee failed a question, rewrite content to ease retrieval, or tune cleanup around question-specific failures.

**Memory structures.** Ingestion ran with default settings, including the **global context index**: a tree-like summary structure with a root summary and more local summaries below. It surfaces temporal links and broader conversation themes, supplementing chunk, graph, and vector retrieval rather than replacing them.

BEAM batches also map naturally to Cognee QA sessions. Session learnings — memory items inferred from the session as a whole, not copied turn-by-turn — were written into memory during ingestion. In production, session knowledge distillation is part of **`improve`**; here, it was run during ingestion so completed BEAM batches could contribute session-level memory. Session learnings are stored as a separate memory class and can be included or excluded through retrieval configuration. Distillation deduplicates against similar existing learnings and rejects repetitive candidates.

Live session memory was omitted because BEAM assistant turns are fixed; Cognee did not generate them.

**Deliberate non-changes.** We did not apply custom graph models, domain ontologies, or bespoke ingestion pipelines. The purpose was to evaluate how standard Cognee extraction behaves on a long-conversation benchmark, not how far a fully benchmark-tuned ingestion stack could be pushed.

### 3.3 Question Answering and Retrieval

Each probing question was answered using standard Cognee retrievers, followed by a reader LLM response. To stay directionally comparable with other published BEAM runs, we used a separate answer prompt per question type and optionally routed retriever choice and parameters by question type.

| Component | What varied | What stayed fixed |
| --- | --- | --- |
| Retriever | Hybrid, graph-completion, decomposition; raw chunk in search space | Standard Cognee retriever implementations |
| Prompt | Per question type | No BEAM-specific graph builder or custom memory architecture |
| Retrieval depth | Graph, chunk, and vector result counts; vector search breadth | Standard Cognee retrieval options |
| Evaluation | BEAM questions and rubric scoring | BEAM upstream eval code |
| Data pipeline | BEAM formatting and paired-turn chunking | Default Cognee ingestion and memory components |

**One-shot retrieval.** We searched over standard retrievers. **Hybrid** retrieval dominated the strongest settings. **Graph-completion** and **decomposition** were selected for question types that depend on connected evidence or intermediate reasoning. **Raw chunk** retrieval was included in the search space but was generally weaker.

**Question-type routing.** Routing applies type-specific retriever and depth settings instead of a single global configuration. It raised 100K scores above **0.8** and produced the best **0.67** run at 10M.

**Hyperparameter search.** Before the reported 100K run, we ingested several conversations to learn the benchmark, tuned prompts and retriever settings on **two** conversations, and verified transfer on **two** others. The headline 100K score is on the **second 100K conversation**, held out from that tune-and-transfer set.

At 10M, we started from 100K-selected settings, then adjusted retriever parameters and routing on the 10M ingestion after those settings underperformed. We did not iteratively tune against the final reported aggregate score (see §3.1 for the variance-reduction reruns on strong routing configurations).

The exact retriever configurations, prompt variants, and per-question-type routing choices are not fully listed in this version of the report.

### 3.4 Evaluation Protocol

Answers were scored with BEAM's official rubric-based LLM evaluation logic, ported into Cognee's evaluation framework so it can run with the sweep tooling. The primary metric is the **average rubric score** across evaluated questions. BEAM auxiliary metrics, such as Kendall's tau, are retained where applicable.

The same model generated answers from retrieved context at 100K and 10M, and a separate model judged those answers against BEAM's rubric. Concrete model names will be reported in a later revision.

Answer prompts varied by BEAM question type. This matches common practice in the community and matters because rubric wording strongly shapes the format a scored answer is expected to take.

Reported settings were evaluated multiple times over the same ingestion and averaged where applicable. Failed questions were not retried. At 100K, the score stabilized over **four** repeated QA-and-evaluation rounds on a fixed ingestion. LLM-as-judge scoring adds noise throughout, especially when margins between systems are small.

## 4. Results

### 4.1 Summary

| Scale | Configuration | Score | Comparison baseline |
| --- | --- | --- | --- |
| 100K | One-shot + per-type prompts | **0.79** | 0.735 |
| 100K | Question-type routing | **>0.8** | 0.735 |
| 10M | Best routing run | **0.67** | 0.64 |

*Baseline scores are from our own reproduction runs and match the figures already shared in the companion blog post. The underlying run artifacts and full methodology will be published alongside a later revision of this report.*

### 4.2 100K

The 100K setup was used for methodology development, transfer checks, and the strongest repeated evaluation. A one-shot retrieval strategy with per-type prompts reached **0.79** on the held-out second 100K conversation. Question-type routing consistently improved results and produced scores above **0.8**.

The 100K setting is the better-supported part of the evaluation. It includes multiple ingestions, tuning conversations, transfer checks, and repeated QA-and-evaluation rounds against a fixed ingestion.

### 4.3 10M

Applying 100K-selected settings unchanged to the 10M conversation produced a lower score, as expected. Subsequent parameter sweeps showed that different retriever values emphasized different graph signals, and no single global parameter set matched the stronger 100K configurations across all question types.

Question-type routing recovered much of the gap and produced the best observed **0.67** score (independently rerun to reduce variance — see §3.1). Because the 10M result is based on a single ingestion, we report it as a strong directional result rather than a fully optimized benchmark conclusion.

### 4.4 Per-Ability Breakdown

*Per-question-type score breakdown will be included in a later revision of this report.*

## 5. Discussion

This discussion should be read as an interpretation of the currently reported runs, not a complete ablation or final benchmark study.

### 5.1 Observations About Cognee

**Standard Cognee memory primitives were sufficient for competitive results.** The reported runs use standard Cognee ingestion, memory structures, and retrievers. We did not introduce a BEAM-specific graph builder, ontology, or memory architecture.

**Routing mattered.** Question-type routing consistently outperformed a single global retriever configuration at both 100K and 10M. This suggests that different BEAM question types place meaningfully different demands on retrieval depth, graph traversal, and evidence selection.

**Scale changed retrieval behavior.** At 10M, retriever parameter sensitivity became more visible. Different parameter values emphasized different graph signals, and 100K-optimal settings did not transfer directly as a single global configuration.

**Global context and session memory were relevant, but not isolated.** The global context index provided a higher-level navigation layer over long conversations, and session learnings captured conversation-level updates and preferences relevant to knowledge-update and preference-following questions. We did not run formal ablations with these components disabled, so their standalone contribution is not reported.

**Preprocessing was not a major driver.** Cleanup affected less than **0.2%** of turns and targeted obvious corpus noise. We found no evidence that preprocessing materially moved aggregate scores; its purpose was to prevent corrupted turns from polluting ingestion.

### 5.2 Benchmark Considerations

**Corpus noise.** BEAM conversations are synthetic but not clean. A small fraction of turns contain repetitive strings, nonsensical text, or leaked planning content.

**Rubric-format coupling.** Rubric wording affects the form and format expected in a scored answer. Per-type prompts adapt to that structure and improve comparability with other community runs, but they also introduce benchmark-specific tuning.

**Gold turn-reference errors.** BEAM gold answers cite supporting conversation turns, but those references can contain errors at this scale.

**LLM-judge variance.** Final scores depend on LLM judges applying rubrics. This adds noise, especially when score differences between systems are small.

**Comparison limits.** Baseline setups are not fully documented. We compare in LLM class and BEAM protocol, not as a strict apples-to-apples match. BEAM also does not cover every production agent workload, such as tool use, APIs, or cross-system feedback loops.

## 6. Conclusion

Open-source Cognee — standard **`add`** / **`cognify`** ingestion, session distillation at ingest time, global context index, and standard retrievers with question-type routing — reached **0.79** / **>0.8** at 100K and **0.67** at 10M on BEAM without a benchmark-specific memory architecture.

These scores are a **directional signal**, not proof of general memory quality or production readiness. We tuned prompts and retrieval for BEAM comparability, excluded agentic multi-turn retrieval from the reported numbers, ran one 10M ingestion, and do not claim that BEAM represents all real-world agent memory workloads.

The strongest supported claim is narrower: Cognee's standard memory pipeline and standard retrievers perform competitively on BEAM, and the system appears to scale meaningfully from 100K to 10M contexts when retrieval is adapted by question type.

## Appendix A. Reproducibility Notes

This report is published before all supporting artifacts have been organized into a final public bundle. The current code exposes the main reproducibility entry points for preprocessing, local ingestion, and BEAM evaluation. Exact configuration files, detailed metric tables, and distributed run metadata are not fully documented in this version.

The public entry points are:

- `cognee.eval_framework.beam.preprocessing.preprocess`
- `cognee.eval_framework.beam.local_ingest`
- `cognee.eval_framework.beam.eval.run_sweep`

## Appendix B. Running An Ingestion

First preprocess BEAM conversations into audited JSON and ingestion-ready JSON-list session files:

```bash
uv run python -m cognee.eval_framework.beam.preprocessing.preprocess \
  --dataset beam \
  --splits 100K \
  --max-conversations 1 \
  --output-dir temp/beam_preprocessed_documents
```

To execute LLM compression for over-limit turns, add `--execute-compressions`. Without it, preprocessing runs as a dry-run audit.

Then ingest one generated conversation folder:

```bash
uv run python -m cognee.eval_framework.beam.local_ingest \
  temp/beam_preprocessed_documents/100k/conversation_000000_id_<conversation_id> \
  --dataset-name beam_100k_local \
  --prune-first
```

The ingestion command writes an `ingestion_report.json` in the run directory. It uses standard Cognee ingestion and memory components, with JSON-list session files preserving BEAM turn boundaries.

## Appendix C. Running Evaluation

After ingestion, run a BEAM sweep against the existing Cognee corpus:

```bash
uv run python -m cognee.eval_framework.beam.eval.run_sweep \
  --split 100K \
  --conversation-index 0 \
  --config-json-path path/to/sweep_config.json
```

For BEAM-10M, pass exported questions with `--questions-path`. The reported 10M ingestion used distributed execution on Modal; this version of the report does not include fully organized distributed run instructions.

## Appendix D. Supporting Artifacts

The following artifacts are useful for reproducing or extending the report. Some are available through the current code paths; others are not fully organized in this version.

- Preprocessing reports and manifests.
- Ingestion reports.
- Sweep configuration files.
- Question files used with `--questions-path`.
- Raw answer, metric, and aggregate metric outputs.
- Per-question-type summaries.
- Distributed run metadata for large-scale runs.
