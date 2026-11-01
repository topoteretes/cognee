# Cognee on LoCoMo - Technical Evaluation Report

## 1. Introduction

This report documents our evaluation of Cognee, an open-source AI memory platform, on LoCoMo, a benchmark of very long-term conversational memory. We used standard Cognee memory components with LoCoMo-specific data formatting, one answer prompt for every question, a fixed retrieval configuration and an LLM judge. The goal was to measure the existing pipeline with a setup that others can inspect and rerun.

Cognee answers **91%** of the 1,762 evaluated questions correctly according to the LLM judge, using the hybrid retrieval approach we previously evaluated on [BEAM](../beam/REPORT.md). We audited LoCoMo's reference answers and exclude the questions that our audit or an external one found faulty. We also leave out open-domain questions, since this evaluation focuses on memory rather than general knowledge. As is now common for LoCoMo, an LLM judge grades the answers, with a single prompt for all questions; we comment on current judging practices in Section 5.

Section 2 introduces LoCoMo and the parts of Cognee used here. Section 3 describes the method: how the conversations are ingested, how questions are answered, and the evaluation protocol, including the scope and the grading rules. Section 4 reports the results, overall and by category. Section 5 discusses a configuration at the retriever's default depth, how our grading compares with published LoCoMo judges, and the limits of the evaluation. Section 6 concludes, and Appendix A gives the models, configurations, rerun commands, audits and artifacts behind the results.

## 2. Preliminaries

### 2.1 LoCoMo

LoCoMo ([paper](https://arxiv.org/abs/2402.17753), [repository](https://github.com/snap-research/locomo)) contains 10 conversations between two people, each with 19–32 dated sessions and 369–689 turns, and 1,986 questions about them. The questions fall into five categories: single-hop (841), multi-hop (282), temporal (321), open-domain (96) and adversarial (446). Open-domain questions combine what a speaker said with general knowledge. Adversarial questions cannot be answered from the conversation: the expected response is to say so, and the dataset pairs each with a plausible wrong answer.

The [original scorer](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py) uses token overlap with the reference answer; many published evaluations use an LLM judge instead. §3.3 describes the dataset limitations and evaluation choices relevant to this study.

### 2.2 Cognee

Cognee builds memory with `remember()`, which runs `add` (register the documents), `cognify` (split them into chunks, extract entities and relationships into a knowledge graph with an LLM, summarize the chunks, and embed the results for vector search) and `improve` (enrich and index the graph). At question time, a **retriever** selects context from this memory, and a reader LLM answers from it under an **answer prompt**.

The **hybrid** retriever used here combines three channels in one pass: conversation passages (chunks), entities with their relationships, and further relationships (facts) not already listed with an entity. Cognee also offers other retrievers and memory features, among them time-aware retrieval, session memory for ongoing conversations, self-improvement from feedback, and a global context index over the chunk summaries.

## 3. Methodology

The experiment measures accuracy: the share of evaluated LoCoMo questions that Cognee's memory pipeline answers correctly, as graded by an LLM judge. Each conversation is ingested into its own memory (§3.1); each question is answered from that memory by one retrieval configuration under one answer prompt (§3.2) and graded correct or wrong against its reference answer (§3.3). The configuration in §5 changes the ingestion and retrieval settings and keeps the questions, answer prompt and judge.

### 3.1 Data Ingestion

The source is `locomo10.json` from the LoCoMo repository. Every session is cut into windows of six consecutive turns. Each window starts with a header naming the session, its date and time, and the two speakers, so the date stays next to the events it dates; a shared photo is kept as its caption. A short overview document per conversation lists the speakers and the session timeline. The ten conversations give 1,048 windows.

Each conversation's documents go through one `remember()` call into a separate memory, with Cognee's standard extraction and summary prompts. The chunk size (4,096 tokens) is larger than any window, so each window becomes exactly one chunk. Ingestion used `gpt-5-mini` for extraction and summaries and `text-embedding-3-large` for embeddings. Excluding a question from evaluation removes nothing from ingestion: every conversation is stored in full.

### 3.2 Question Answering and Retrieval

For each question, the hybrid retriever (§2.2) selects context from the conversation's memory, and `gpt-5-mini` answers from it. One answer prompt ([`qa_prompts/unified.txt`](qa_prompts/unified.txt)) is used for every question, and the question is asked as written: neither its category label nor its gold answer reaches the answering model. The labels carry information that an ordinary question does not, such as that an adversarial question has no answer in the conversation. The prompt asks for the shortest complete answer, resolves relative dates against the session date, collects every item when the question asks for several, and replies "Not mentioned in the conversation" when the context does not attribute the fact to the person asked about. Appendix A.2 gives the exact retrieval configuration and the reproduction command.

### 3.3 Evaluation Protocol

**Scope.** We evaluate single-hop, multi-hop, temporal and adversarial questions, and leave out open-domain questions and questions whose reference answer an audit flags. Open-domain questions need general knowledge, so they test more than memory. Adversarial questions stay in, because recognizing that a conversation holds no answer is part of conversational memory.

We use an external audit and our own (Appendix A.3) to identify faulty reference answers, which can make a faithful response count as an error. For example, both audits flag the question "When did Evan have a drunken night with his friends?" in conversation `conv-49`: the reference answer is "January 9, 2023", but in a session dated 10 January 2024 Evan says that it happened "yesterday", which makes the date 9 January 2024.

The 224 excluded questions are the 96 open-domain questions and the 143 questions flagged by either audit, 15 of which are also open-domain, so 1,762 of the 1,986 questions are evaluated. The exclusion list, with each audit's error type or label and reason, is [`report_artifacts/gold_exclusions.json`](report_artifacts/gold_exclusions.json).

**Judge.** An LLM judge, `gpt-5.1` with medium reasoning effort, grades each answer as CORRECT or WRONG in a separate call, so the answering model never grades its own output. The judge sees the question, its category, the gold answer and the response; it does not see the retrieved context or the conversation.

Its [prompt](prompts/locomo_judge_prompt.txt) accepts paraphrases and equivalent formats. A date must be at least as precise as the gold answer and consistent with it, and a duration must describe the same span. For a gold answer with several parts, a response is correct when it gives at least half of them, rounding up, and contradicts none; extra details are allowed unless they contradict the gold. A response that offers several alternatives where only one matches is wrong, as is one that declines when the gold answer is a concrete fact. For adversarial questions, where the conversation contains no answer, only declining is correct; the prompt names the dataset's plausible wrong answer as an example of a wrong response. See §5 and Appendix A.5 for comparison with other LoCoMo judges.

**Runs.** Two published runs of the main configuration answered and judged the questions on the same memories, built by one ingestion per conversation.

## 4. Results

### 4.1 Main Result

Cognee answered about **91%** of the 1,762 evaluated questions correctly: 1,598 (90.7%) and 1,608 (91.3%) in the two published runs (§3.3).

### 4.2 By Category

The table breaks down the [run that scored 91.3%](report_artifacts/topk20_run2/) by category.

| Category | Questions | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Single-hop | 796 | 736 | 92.5% |
| Multi-hop | 236 | 212 | 89.8% |
| Temporal | 284 | 256 | 90.1% |
| Adversarial | 446 | 404 | 90.6% |
| **Total** | **1,762** | **1,608** | **91.3%** |

Accuracy ranges from 89.8% on multi-hop to 92.5% on single-hop questions. Adversarial questions reach 90.6% under the same prompt as the answerable categories. The smaller categories have a few hundred questions each, so differences of a point or two between them should be read with caution.

## 5. Discussion

**Default retrieval depth.** The main configuration uses deeper retrieval than Cognee's default, so we also evaluated the default as a more economical option. With adapted ingestion and a global context index, it reached **89.7%** (1,580 of 1,761 scored questions) in one run, within two percentage points of the main result. This shows that accuracy close to 90% is achievable at the default depth using Cognee's existing memory components and these ingestion adaptations. Appendix A.2 compares the two configurations; A.4 describes the adapted ingestion.

**Grading for factual recall.** Cognee's roughly 91% score is achieved under the grading rules described in §3.3. These accept paraphrases while requiring factual agreement and dates consistent with and at least as precise as the benchmark's expected answer. Published LoCoMo evaluations use different conventions, with some accepting broader topical matches or date tolerances (Appendix A.5). Cognee performs strongly under these stricter requirements for factual and temporal recall. Including adversarial questions in the headline score also tests Cognee's ability to recognize when the conversation provides no answer.

**What the results show.** Cognee's strong performance across all four question categories supports its ability to handle long conversational histories. The audits strengthen this evidence by addressing faulty reference answers, and the [published exclusions](report_artifacts/gold_exclusions.json) let readers inspect those decisions. That evidence comes from ten conversations used to develop the setup, so the evaluation is not held out. How consistently this performance extends to other data and practical use remains to be tested.

## 6. Conclusion

Cognee answered about **91%** of the evaluated LoCoMo questions correctly under our LLM judge. Alongside our earlier [BEAM](../beam/REPORT.md) evaluation, this adds evidence for Cognee's existing pipeline across different conversational memory tasks.

Evaluation will continue across a broader range of memory tasks, with methods and artifacts published for inspection and reuse. The findings will guide improvements to Cognee and its evaluation, building a clearer public record of what the system can do.

## Appendix A. Reproducibility and Supporting Artifacts

### A.1 Models and Environment

| Stage | Model |
| --- | --- |
| Ingestion: graph extraction and summaries | `openai/gpt-5-mini` |
| Embeddings | `openai/text-embedding-3-large` |
| Date annotation (additional experiment only) | `openai/gpt-5-mini`, low reasoning effort |
| Answer generation | `openai/gpt-5-mini`, default reasoning effort |
| Judge, in a separate call | `openai/gpt-5.1`, medium reasoning effort, at most 6,000 completion tokens |

Stack: Cognee 1.5.4, Ladybug 0.19.0 (graph), LanceDB 0.38.0 (vectors), SQLite (relational), litellm 1.96.2. Environment: `ENABLE_BACKEND_ACCESS_CONTROL=false`, `CACHING=true`, `AUTO_FEEDBACK=false`. The main configuration was ingested on 7 October 2026; its published runs were answered and judged on 9 October (90.7%) and 13 October 2026 (91.3%). The additional experiment was ingested, answered and judged on 12 October 2026. The dataset file has SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; each run's `run_config.json` also records the hashes of the answer, judge and ingestion prompts.

### A.2 Configurations and Rerun Commands

| Setting | Main configuration | Additional experiment (§5) |
| --- | --- | --- |
| Windows | 6 turns with a dated header, one chunk each (4,096-token chunk size) | same |
| Relative dates | as written | annotated with absolute dates before ingestion ([`prompts/resolve_dates.txt`](prompts/resolve_dates.txt)) |
| Extraction prompt | Cognee default | [`prompts/graph_extraction_conversations.txt`](prompts/graph_extraction_conversations.txt) |
| Summary prompt | Cognee default | [`prompts/summarize_content_questions.txt`](prompts/summarize_content_questions.txt) |
| Global context index | none | built after `remember()`; 3 entries per question |
| Retriever | `hybrid_completion`, [`configs/locomo_hybrid_20_20_20.json`](configs/locomo_hybrid_20_20_20.json) | `hybrid_completion`, [`configs/locomo_hybrid_5_5_5_gci.json`](configs/locomo_hybrid_5_5_5_gci.json) |
| Passages / entities / facts per question | 20 / 20 / 20 | 5 / 5 / 5 (the retriever's defaults) |
| Relationships per entity | 10 | 10 |
| Answer prompt | [`qa_prompts/unified.txt`](qa_prompts/unified.txt), the same for every question | same |
| Judge prompt | [`prompts/locomo_judge_prompt.txt`](prompts/locomo_judge_prompt.txt) | same |
| Run record | [`topk20_run2/run_config.json`](report_artifacts/topk20_run2/run_config.json) | [`topk5_adapted_prompts/run_config.json`](report_artifacts/topk5_adapted_prompts/run_config.json) |

Rerun commands, from the repository root:

```bash
# main configuration (the driver uses configs/locomo_hybrid_20_20_20.json unless told otherwise)
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9

# additional experiment
uv run python -m cognee.eval_framework.locomo.run_locomo_eval --conversations 0-9 \
    --resolve-dates \
    --extraction-prompt cognee/eval_framework/locomo/prompts/graph_extraction_conversations.txt \
    --summary-prompt cognee/eval_framework/locomo/prompts/summarize_content_questions.txt \
    --global-context-index \
    --config-json-path cognee/eval_framework/locomo/configs/locomo_hybrid_5_5_5_gci.json
```

Each command creates a fresh run directory, ingests the conversations, answers and judges the evaluated questions, and writes `locomo_summary.json` and `locomo_summary.md` there. The dataset is downloaded to `temp/locomo_data/locomo10.json` when missing; check its SHA-256 against A.1. Values in a `.env` file take precedence over the settings the driver passes to its child processes, such as `LLM_MODEL`, so keep the file consistent with the driver's options. The additional experiment's command has not been rerun end to end (Appendix A.4). Answering and judging also vary between runs (§4.1), so identical scores are not guaranteed. More options are documented in the [LoCoMo README](README.md).

### A.3 Evaluation Scope and Audits

The two audits behind the exclusions in §3.3 are:

- **External audit** by Penfield Labs ([audit report](https://github.com/dial481/locomo-audit/blob/main/AUDIT_REPORT.md)): 99 score-corrupting errors among the non-adversarial questions, by error type: hallucination (33), temporal error (26), attribution error (24), ambiguous (13) and incomplete (3). Entries where the gold answer is right and only the cited evidence is wrong are kept.
- **Our audit**: every non-adversarial gold answer checked by `gpt-5.1` with medium reasoning effort, in two stages, against the conversation and its session dates. The criteria are in [`gold_audit_prompt.txt`](report_artifacts/gold_audit_prompt.txt). Gold answers labelled WRONG, UNSUPPORTED, MALFORMED or AMBIGUOUS with high or medium confidence are excluded: 93 questions (43 unsupported with high confidence and 1 with medium confidence, 39 wrong, 7 ambiguous and 3 malformed).

49 questions are flagged by both audits, so 143 are flagged in all: 45 single-hop, 46 multi-hop, 37 temporal and 15 open-domain.

The benchmark package applies this scope before retrieval: [`scope.py`](scope.py) defines it, and `answer.py` drops excluded questions, so they are never answered or judged; question ids are unchanged. The published runs answered and judged all 1,986 questions; their records in `report_artifacts/` were filtered to the evaluated questions afterwards, leaving every kept record unchanged. The filter is in the [artifact README](report_artifacts/README.md#filtering-existing-run-records).

### A.4 Additional Experiment

Before ingestion, relative time expressions in each window were annotated with the date or period they refer to, computed from the session date, for example "next month [= the month after 23 March 2023, i.e. April 2023]". An annotation output was discarded, and the original window kept, when removing its annotations did not give back the original text word for word. 414 of the 1,048 windows received at least one annotation, 596 in all, and 5 outputs were discarded. The extraction prompt adds rules to name the person in every fact, date every event, and keep the date of facts that can change. The summary prompt also has the summarizing model write 3–5 questions that someone could later ask about the chunk's details; it sees only the chunk, and ingestion never reads the benchmark's questions.

One judge verdict could not be parsed: conversation `conv-47`, question 163, where the judge replied with the label `WRNG`. Its score is left empty rather than counted as wrong, so the experiment has 1,762 evaluated questions and 1,761 scored ones.

The saved run was produced by earlier scripts with the same prompts and settings as the command in A.2, which has not been rerun end to end.

### A.5 Judge Comparison

The published LoCoMo judges below were inspected on 19 October 2026 at the linked revisions. They describe the current implementations, not necessarily the configuration behind every published score.

| Implementation | What its LoCoMo judge accepts | Source |
| --- | --- | --- |
| Mem0, current benchmark | One matching item from a reference list; dates within 14 days; durations within 50%. Adversarial questions are excluded. | [prompts and category selection](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/locomo/prompts.py) |
| Mem0, legacy evaluation | A response that "touches on the same topic" as the reference; GPT-4o-mini, binary verdicts. | [judge](https://github.com/mem0ai/mem0/blob/aae5989e78a6188b3b047c104d960c9ad0927e75/evaluation/metrics/llm_judge.py) |
| Zep, published evaluation | The same topical-match rule; GPT-4o-mini, binary verdicts. | [judge](https://github.com/getzep/zep-papers/blob/4434662f9ddd0f5441df4987fdfdff13362fb055/kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py) |
| Hindsight, current Agent Memory Benchmark | The LoCoMo-specific judge repeats the topical-match rule. Adversarial questions are excluded. | [LoCoMo dataset and prompt](https://github.com/vectorize-io/agent-memory-benchmark/blob/aef3702e0586a959c845cc2c61f10d5e8ff0f94a/src/memory_bench/dataset/locomo.py), [prompt selection](https://github.com/vectorize-io/agent-memory-benchmark/blob/aef3702e0586a959c845cc2c61f10d5e8ff0f94a/src/memory_bench/judge.py) |
| Supermemory, current LoCoMo path | Mem0's legacy topical-match prompt, for every provider. Adversarial questions are scored separately and left out of the headline. | [LoCoMo prompts](https://github.com/supermemoryai/memorybench/blob/e8d6e1f40e8d0cca1b90463af2776e05efccc720/src/benchmarks/locomo/prompts.ts), [dataset settings](https://github.com/supermemoryai/memorybench/blob/e8d6e1f40e8d0cca1b90463af2776e05efccc720/src/benchmarks/locomo/index.ts) |

Supermemory changed its LoCoMo judging on 19 October 2026 ([PR #96](https://github.com/supermemoryai/memorybench/pull/96)); the row describes the revision after that change. Before it, its LoCoMo path used a stricter generic completeness prompt. Our judge's rules are in §3.3.

### A.6 Artifacts and Provenance

The [`report_artifacts/`](report_artifacts/) folder contains, for a run of the main configuration ([`topk20_run2/`](report_artifacts/topk20_run2/)) and the additional experiment ([`topk5_adapted_prompts/`](report_artifacts/topk5_adapted_prompts/)):

- per-question answers and judge verdicts for the 1,762 evaluated questions, one file per conversation, with the original question ids;
- the summary tables behind §4 and §5, regenerated from those files by `aggregate.py`;
- the run configuration with models, settings, dates and prompt hashes.

It also contains both ingestion reports with their window maps, the date annotations of the additional experiment, the exclusion list and the prompt of our audit. Its [README](report_artifacts/README.md) describes the layout and the attribution for the dataset and the external audit.

Another run of the main configuration scored 1,598 of 1,762 evaluated questions (90.7%). Its complete records are at [commit](https://github.com/topoteretes/cognee/tree/012e8a039b5116bc30325a9c10192d9c9336e399/cognee/eval_framework/locomo/report_artifacts/qa) `012e8a0`, which also holds the unfiltered records of `topk20_run2/` and `topk5_adapted_prompts/`.
