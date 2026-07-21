*Preliminary version*

# Cognee on BEAM - Technical Evaluation Report

## 1. Introduction

This report documents our evaluation of Cognee, an open-source AI memory platform, on BEAM. We used standard Cognee memory components with BEAM-specific data formatting, answer prompts, retrieval configuration, and evaluation tooling. We kept the effort deliberately bounded: our goal was to evaluate the existing pipeline, compare retrieval behavior across the tested context scales, and make the resulting setup auditable.

Cognee reached **0.79** on the primary 100K evaluation and **0.67** in an exploratory 10M scale check. The 100K work established a repeatable fixed-retrieval configuration. At 10M, question-type routing improved on that starting point.

The 100K result was evaluated repeatedly on a held-out conversation. For 10M, we used one ingestion and selected the retrieval settings on the same questions used for the reported score, rather than a separate held-out set. We therefore treat the result as exploratory.

## 2. Preliminaries

This section gives the background the rest of the report builds on: what BEAM evaluates, and the Cognee concepts used throughout. Readers already familiar with Cognee can skip §2.2.

### 2.1 BEAM

BEAM is a benchmark for evaluating long-context conversational memory systems. It consists of synthetic multi-session user-assistant conversations, probing questions, gold answers, and rubric-based evaluation performed by an LLM judge.

Questions target ten memory abilities: information extraction, multi-hop reasoning, knowledge updates, temporal reasoning, summarization, preference following, abstention, contradiction resolution, event ordering, and instruction following. BEAM documentation also refers to multi-hop reasoning as multi-session reasoning.

**Reference implementation.** This evaluation follows the publicly available [BEAM paper](https://arxiv.org/abs/2510.27246) and [upstream implementation](https://github.com/mohammadtavakoli78/BEAM). The source datasets are linked in §3.2.

### 2.2 Cognee

Cognee is an open-source memory platform with two ingestion primitives: **`add`** registers source data as documents, and **`cognify`** processes those documents into memory. During `cognify`, documents are split into **chunks**, an LLM extracts entities and relationships into a **knowledge graph**, summaries are generated, and the results are embedded for vector search. One ingestion therefore produces several views of the same conversation: its text, entities and relationships, and summaries.

Two further components support long conversations. The **global context index** organizes local summaries into a tree with a single root, allowing retrieval to navigate themes and temporal links that may not be visible in one chunk. **Session** processing distills knowledge updates, user preferences, and other durable information from completed conversations into the same memory.

At question time, a **retriever** selects context from one or more of these views. **Chunk-vector (RAG-style)** retrieval searches conversation chunks, **graph retrieval** uses the entity-relationship graph, **summary-based** retrieval uses summaries, and **hybrid** retrieval combines these signals. Retrieval settings control how much context each channel contributes. A reader LLM then answers from the retrieved context under an **answer prompt**. Appendix D describes the specific strategies and settings considered in this evaluation.

## 3. Methodology

We evaluated Cognee on BEAM in two stages. First, each conversation was ingested into Cognee memory. Second, BEAM probing questions were answered using Cognee retrievers and scored with BEAM's rubric-based evaluation logic.

The BEAM-specific code handles dataset formatting, the turn-preserving representation, ingestion orchestration, prompts, and evaluation artifacts. Cognee handles ingestion and memory construction. Retrieval settings and answer prompts were tuned for BEAM comparability, as described below.

### 3.1 Experimental Design

At **100K**, prompts and retriever settings were tuned on **two** conversations and checked on **two** additional conversations. A separate conversation was held out from that process and used for the reported score.

At **10M**, we performed one ingestion of one conversation and started with the fixed retrieval settings selected at 100K. Those settings produced a lower score, so we used follow-up sweeps on the same 10M questions to select retrieval strategies by question type. We then reran the selected routed configuration over repeated evaluation rounds to average out scoring noise. Because selection and reporting use the same question set, this is an in-sample exploratory result.

We also tested multi-turn agentic retrieval, but excluded it from the results because it changes the interaction protocol from one-shot to iterative retrieval. Those experiments are outside the scope of this report.

### 3.2 Data Ingestion

**Source data.** Conversations come from [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) and [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M). We ingested them with Cognee **`add`** and **`cognify`**.

**Models.** Model assignments remained consistent across scales except that the distributed 10M run used a smaller ingestion model. Appendix A lists the models and their roles.

**Document and chunk mapping.** In BEAM, conversations below 10M are split into **batches**, each representing one continuous user-assistant dialogue. At 10M, conversations are split into **plans**, and each plan is split into batches. We mapped each batch to one Cognee document and each user-assistant turn to one chunk. Turn order, timestamps, and session metadata were preserved where BEAM provides them.

BEAM batch boundaries served as the local units, so we did not use overlap-based chunking. Appendix B describes the ingestion representation.

**Run mode.** The 10M ingestion ran in Cognee distributed mode on Modal. The public code does not include the BEAM-specific distributed orchestration from that run; Appendix B describes the available reproduction path.

**Preprocessing.** Less than **0.2%** of turns required cleanup before ingestion, mostly long assistant turns. Noise included repeated random strings, nonsensical text, and leaked conversation-plan text. Those turns were passed through a small LLM compression step using a generic, non-benchmark-specific prompt. Appendix B gives the prompt and execution details.

The cleanup criteria addressed corpus noise, not question outcomes or retrieval failures.

**Global context index.** Ingestion also built the global context index (§2.2), so retrieval could surface temporal links and broader themes alongside chunk, graph, and vector signals.

**Cognee sessions.** BEAM is a sequence of completed conversations, so each batch was mapped to a Cognee QA session (§2.2), retaining distilled learnings such as updates and preferences. Session processing did not add new benchmark dialogue: any assistant responses it produced were discarded.

### 3.3 Question Answering and Retrieval

For each probing question, a Cognee retriever selected context (§2.2), and a reader LLM generated the response. Answer prompts followed BEAM question types so that responses matched the format expected by each ability-specific rubric. Appendix C describes how the prompts are attached to an evaluation run.

For retrieval, we first searched over fixed Cognee retriever and depth configurations. We also tested question-type routing, where different BEAM question types use different configurations and channel depths. The primary 100K result uses fixed retrieval; the 10M result uses routing.

**Fixed retrieval.** Development sweeps led us to select hybrid retrieval for the reported fixed configuration. Other Cognee retrievers and depth settings were also tested. The 100K configuration uses hybrid retrieval with per-question-type prompts. Appendix D lists the exact configuration.

**Question-type routing.** Routed configurations use different retrieval settings by question type instead of one global configuration. This changes how the existing memory is queried rather than adding a separate memory component. We also explored routing at 100K; those experiments suggested additional headroom but are outside the reported result and artifact bundle.

Appendix D describes the retrieval search space and reported settings. The 100K search and held-out split is summarized in §3.1.

### 3.4 Evaluation Protocol

Answers were evaluated with BEAM's upstream rubric-based LLM procedure, ported into Cognee's evaluation framework without changing its scoring rules. BEAM scores each rubric criterion at 0, 0.5, or 1 and averages the criteria into a question score. The primary metric is the mean question score on a **0–1** scale.

Answer generation and BEAM judging ran as separate calls. We repeated the full QA-and-scoring round four times at 100K and five times at 10M, holding the ingestion and selected retrieval configuration fixed within each scale, then averaged the results. Failed questions were not retried. Repetition captures run-to-run variation from answer generation and LLM judging; it does not add new question samples. Appendix A lists the models.

## 4. Results

### 4.1 Summary

| Scale | Retrieval configuration | Score | Interpretation |
| --- | --- | --- | --- |
| 100K | Fixed hybrid retrieval | **0.79** | Primary repeated result |
| 10M | Question-type routed hybrid retrieval | **0.67** | Exploratory scale check |

These scores are not a controlled comparison across scales: the conversations, ingestion models, and retrieval-selection procedures differ.

### 4.2 100K

After methodology development and transfer checks on the other 100K conversations, the fixed-retrieval setup with per-type prompts reached **0.79** on the held-out second conversation. The result averages four QA-and-evaluation rounds over 20 questions; the run standard deviation is **0.005**.

The aggregate score changed little across rounds. Appendix E reports the per-question-type breakdown and descriptive pooled bootstrap intervals.

### 4.3 10M

Applying the fixed 100K retrieval setup to the 10M conversation produced a lower score. Follow-up sweeps on the same ingestion and question set indicated that different question types benefited from different retriever settings; no single global setting worked best across the question mix.

The 10M scale check therefore uses question-type routing, which produced **0.67** averaged over five evaluation rounds. Because routing was selected on the same questions used for scoring, the result describes this configuration on this question set rather than a held-out estimate.

Appendix G records the candidate configuration, routing map, and evaluation runs. Appendix F describes the remaining reproduction gap.

## 5. Discussion

**Retrieval settings depended on the question type.** BEAM's abilities stress different views of the memory (§2.2), and no single balance of chunk, graph, summary, and global-context signals served the whole question mix. Development sweeps led us to select hybrid retrieval for the fixed configuration, while routing experiments suggested that different question types could benefit from different channel depths. The 10M work covers one conversation, so it does not establish a general routing benefit. At 100K, the held-out result provided some evidence that retrieval and prompt choices developed on other conversations could transfer. Similar transfer at 10M is plausible, but the single-conversation experiment does not test it.

**BEAM's question taxonomy made routing practical.** The routed configuration uses BEAM's question-type labels to select an answer prompt and retrieval settings. Outside a benchmark with explicit labels, the retrieval approach would need to be inferred from the question itself. One possible extension is an LLM-assisted router in front of the existing retrievers; we did not evaluate one in these runs.

**A fully public benchmark limits evidence of generalization.** BEAM exposes its conversations, questions, and rubrics and provides no hidden test set. Once those materials inform prompts, routing, or retriever settings, further score gains can reflect general improvement, adaptation to BEAM, or both, without proportionally strengthening the evidence that the system generalizes. At 100K, the development, transfer-check, and held-out split limited this risk, but it is not equivalent to a hidden test set. The 10M work has no such separation, which is why we treat it as exploratory. Stronger validation would require freezing the configuration and testing it on unseen data or another benchmark without further tuning.

**Data quality and scoring affected different parts of the pipeline.** Noisy turns can propagate into chunks, summaries, and graph extraction even when they represent a small fraction of the dataset. Apparent discrepancies in some gold references and LLM-judge variation affect the other end of the evaluation by limiting score precision. We therefore document the preprocessing criteria (§3.2), repeat evaluation rounds, and interpret small score differences cautiously.

**The components appeared useful for different reasons, but were not isolated.** During development, the global context index provided access to broader themes and temporal links, while session processing represented updates and preferences from completed conversations. Preprocessing prevented obvious corpus artifacts from entering the memory pipeline. These observations shaped the setup, but we did not run controlled ablations. The scores therefore describe the combined pipeline, not the contribution of each component.

## 6. Conclusion

This evaluation applied Cognee's existing open-source memory pipeline to BEAM at the 100K and 10M scales. Most of the work went into preserving conversation structure, configuring retrieval and answer prompts, and integrating the evaluation workflow. We did not add a BEAM-specific memory component or retriever implementation.

BEAM also made a simple point clear: the basics matter. Preprocessing and representation shape what reaches memory; retrieval choices shape what reaches the reader. Cognee already provided the ingestion interfaces, memory views, and retrieval strategies needed to make those adjustments. The core pipeline stayed the same while the setup around it changed, which is a practical sign of flexibility.

BEAM is a useful structured checkpoint, but it covers a defined set of synthetic tasks rather than the full conditions a memory system faces in practice. Real-world assessment also depends on the data, operating constraints, and reliability over time. Within that scope, the benchmark required a new setup around Cognee, not a new memory system.

## Appendix A. Reproducibility Notes

The model assignments were:

| Stage | 100K | 10M |
| --- | --- | --- |
| Embeddings | `openai/text-embedding-3-large` | `openai/text-embedding-3-large` |
| Ingestion: extraction, summarization, and session distillation | `openai/gpt-5` | `openai/gpt-5-mini` |
| Answer generation | `openai/gpt-5` | `openai/gpt-5` |
| Rubric judge, in a separate call | `openai/gpt-5` | `openai/gpt-5` |

The current code exposes the main reproducibility entry points for preprocessing, local ingestion, and BEAM evaluation:

- `cognee.eval_framework.beam.preprocessing.preprocess`
- `cognee.eval_framework.beam.local_ingest`
- `cognee.eval_framework.beam.eval.run_sweep`
- `cognee.eval_framework.beam.eval.aggregate_cross_run`

Appendices B and C show how to run them; supporting configs, prompts, and result artifacts are inventoried in Appendix F.

## Appendix B. Running an Ingestion

First preprocess BEAM conversations into audited JSON and ingestion-ready JSON-list session files:

```bash
uv run python -m cognee.eval_framework.beam.preprocessing.preprocess \
  --dataset beam \
  --splits 100K \
  --max-conversations 2 \
  --output-dir temp/beam_preprocessed_documents
```

To execute LLM compression for over-limit turns, add `--execute-compressions`. Without it, preprocessing runs as a dry-run audit. Compression uses the prompt file given by `--prompt-path`, which defaults to the published [`beam_turn_compression_prompt.txt`](preprocessing/prompts/beam_turn_compression_prompt.txt) — the reported runs used that default.

Preprocessing defines each turn unit before ingestion. The JSON-list chunker reads each list item as one chunk and does not add overlap.

Then ingest the second generated conversation folder used for the reported 100K evaluation:

```bash
uv run python -m cognee.eval_framework.beam.local_ingest \
  temp/beam_preprocessed_documents/100k/conversation_000001_id_<conversation_id> \
  --dataset-name beam_100k_local \
  --prune-first
```

The ingestion command writes an `ingestion_report.json` in the run directory. It uses Cognee ingestion and memory components, with JSON-list session files preserving BEAM turn boundaries.

The 10M ingestion ran in Cognee distributed mode on Modal. The public code includes the same preprocessing representation and the local path above, but not the BEAM-specific distributed orchestration used for that run. The JSON-list session representation is compatible with Cognee's Modal distributed execution.

## Appendix C. Running an Evaluation

After ingestion, run a BEAM sweep against the existing Cognee corpus. The invocation matching the reported fixed-retrieval 100K run is:

```bash
uv run python -m cognee.eval_framework.beam.eval.run_sweep \
  --split 100K \
  --conversation-index 1 \
  --num-runs 4 \
  --config-json-path cognee/eval_framework/beam/report_artifacts/100k_fixed/beam_hybrid_completion_20_20_qa_v1_config.json
```

The per-question-type answer prompts are not passed on the command line: each retriever variant in the sweep config carries a `qa_prompt_paths` map from BEAM question type to prompt file. The archived configs point those maps at [`report_artifacts/qa_prompts/`](report_artifacts/qa_prompts/); a config without `qa_prompt_paths` still runs, but does not reproduce the reported per-type-prompt setup.

BEAM provides an ability label for each question. We used that label to select an ability-specific answer prompt, following other BEAM results reported by the community and thus preserving directional comparability. Without those labels, an LLM-assisted router could infer the question type and select the prompt instead. We did not add that step here because it would introduce a component outside the benchmark convention we aimed to follow.

Note that `--num-runs 4` repeats the full QA-and-evaluation round four times; budget LLM usage accordingly.

Then aggregate the repeated runs into the cross-run summary the results tables are built from:

```bash
uv run python -m cognee.eval_framework.beam.eval.aggregate_cross_run \
  --output-dir <sweep_output_dir> \
  --conversation-index 1 \
  --retriever hybrid_completion_20_20_qa_v1
```

The `--retriever` value must match the resolved retriever variant name from the sweep config, as used in the sweep's metrics file names.

For BEAM-10M, pass exported questions with `--questions-path`.

## Appendix D. Retrieval Strategies and Reported Configurations

### D.1 Retrieval Strategies Considered

The strategies considered during development come from the registry in [`eval/registry.py`](eval/registry.py). They wrap Cognee retriever implementations; none are BEAM-specific.

| Strategy | Description |
| --- | --- |
| `cognee_completion` | Chunk-vector retrieval followed by LLM completion (RAG-style). |
| `cognee_graph_completion` | Retrieval over the extracted entity-relationship graph, followed by LLM completion. |
| `cognee_graph_completion_cot` | Graph completion with chain-of-thought reasoning over retrieved context. |
| `cognee_graph_completion_context_extension` | Graph completion with iterative context extension. |
| `graph_completion_decomposition` | Decomposes the question into sub-questions, then runs graph completion over them. |
| `graph_summary_completion` | Retrieval over pre-computed summaries with graph context. |
| `hybrid_completion` | Combines chunk, graph, and summary signals in a single retrieval pass. |

### D.2 Reported 100K Fixed Configuration

| Setting | Value |
| --- | --- |
| Retriever strategy | `hybrid_completion` |
| Variant name | `hybrid_completion_20_20_qa_v1` |
| Chunk top-k | 20 |
| Entity top-k | 20 |
| Other retriever settings | Standard `HybridRetriever` defaults |
| Global context index in search space | Built during ingestion, but not queried by this configuration (`include_global_context_index=false`) |
| Answer prompts | Per question type, from [`report_artifacts/qa_prompts/`](report_artifacts/qa_prompts/) |
| Packaged reproduction config | [`beam_hybrid_completion_20_20_qa_v1_config.json`](report_artifacts/100k_fixed/beam_hybrid_completion_20_20_qa_v1_config.json) |

## Appendix E. 100K Per-Question-Type Summary

This table breaks down the primary 100K result by BEAM ability. Each conversation has only a few questions per ability, so the results should be read directionally, not as stable per-ability estimates.

The descriptive pooled bootstrap interval treats each scored question from each repeated round as an observation. Because the rounds reuse the same questions, it is not a confidence interval over independent question samples. Run standard deviation provides the complementary spread across repeated rounds.

| Ability | Questions | Mean BEAM rubric | Descriptive pooled bootstrap interval | Run std |
| --- | ---: | ---: | ---: | ---: |
| Information extraction | 2 | 0.979 | 0.938-1.000 | 0.042 |
| Multi-session (multi-hop) reasoning | 2 | 0.500 | 0.125-0.875 | 0.000 |
| Knowledge updates | 2 | 1.000 | 1.000-1.000 | 0.000 |
| Temporal reasoning | 2 | 0.500 | 0.125-0.875 | 0.000 |
| Summarization | 2 | 0.802 | 0.698-0.906 | 0.050 |
| Preference following | 2 | 1.000 | 1.000-1.000 | 0.000 |
| Abstention | 2 | 0.500 | 0.125-0.875 | 0.000 |
| Contradiction resolution | 2 | 0.875 | 0.781-0.969 | 0.000 |
| Event ordering | 2 | 0.788 | 0.675-0.900 | 0.048 |
| Instruction following | 2 | 1.000 | 1.000-1.000 | 0.000 |

Values come from the `by_question_type` section of the `cognee.eval_framework.beam.eval.aggregate_cross_run` output for the fixed-retrieval 100K run. Row names follow the dataset's `question_type` keys. Kendall tau applies only to event-ordering questions and is kept in the JSON artifact.

## Appendix F. Supporting Artifacts

The following report artifacts are included in [`report_artifacts/`](report_artifacts/):

- Archived per-question-type QA prompts: [`report_artifacts/qa_prompts/`](report_artifacts/qa_prompts/).
- Turn compression prompt: [`preprocessing/prompts/beam_turn_compression_prompt.txt`](preprocessing/prompts/beam_turn_compression_prompt.txt).
- Primary 100K fixed-retrieval bundle: [`report_artifacts/100k_fixed/`](report_artifacts/100k_fixed/), containing the packaged reproduction config, four per-run metric files, and the [cross-run summary](report_artifacts/100k_fixed/hybrid_completion_20_20_qa_v1_cross_run_summary.json) behind §4.2 and Appendix E.
- Exploratory 10M routed bundle: [`report_artifacts/10m_routed/`](report_artifacts/10m_routed/), containing the candidate retriever config, selected routing map, five per-run evaluation records, and cross-run summary listed in Appendix G.

The preprocessing and ingestion entry points also generate reports and manifests, including `ingestion_report.json`. The original distributed orchestration and ingestion run metadata for the reported 10M corpus are not included in this version.

## Appendix G. 10M Exploratory Run Record

The 10M result comes from the first BEAM-10M conversation and evaluates all 20 probing questions for that conversation. This record provides the evaluation-layer audit trail; the original distributed-ingestion metadata is outside the included bundle.

| Item | Value |
| --- | --- |
| Dataset / split | BEAM-10M / 10M |
| Conversation index | 0 |
| Evaluated questions | 20 |
| Question types | 10 types; 2 questions per type |
| QA/eval runs | 5 |
| Mean BEAM rubric | **0.67** |
| Run std | 0.020 |
| Candidate retriever config | [`beam_qa_v1_hybrid_routing_configs.json`](report_artifacts/10m_routed/beam_qa_v1_hybrid_routing_configs.json) |
| Selected routing map | [`routing.json`](report_artifacts/10m_routed/routing.json) |
| Per-run evaluation records | [`report_artifacts/10m_routed/`](report_artifacts/10m_routed/) (`run0` through `run4`) |
| Cross-run summary | [`routed_by_question_type_cross_run_summary.json`](report_artifacts/10m_routed/routed_by_question_type_cross_run_summary.json) |

The routed retrieval configuration was selected after inspecting behavior on this single 10M ingestion, and candidate selection and scoring used the same question set. The result is therefore exploratory rather than a held-out estimate. The public BEAM dataset remains the authoritative source for the questions and rubrics; they are also embedded in each per-run evaluation artifact for auditability.
