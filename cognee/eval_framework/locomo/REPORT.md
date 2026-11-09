# Benchmarking cognee on LoCoMo - Technical Evaluation Report

*Revised November 19, 2026. This revision describes the answer prompt's decline rules and the judge's partial-credit rule in more detail, including when that rule changed; explains how the exclusion list was built and what our audit could see; reports both runs of the main configuration by category and with nothing excluded; states which cognee version produced the runs; and removes observations we can't back up with hard data, such as the default depth being more economical. Earlier versions are in the repository history.*

## 1. Introduction

This report documents our evaluation of cognee, an open-source AI memory platform, on LoCoMo, a benchmark of very long-term conversational memory. We used cognee's existing memory components and built a LoCoMo-specific setup around them: how the conversations are formatted for ingestion, one answer prompt for every question, the retrieval depth and the judge's grading prompt were all developed on the ten conversations we report on. The goal was to measure that pipeline with a setup that others can inspect and rerun.

cognee answered **~91%** of the 1,762 evaluated questions correctly according to the LLM judge (90.7% and 91.3% in two runs), using the hybrid retriever we previously evaluated on [BEAM](../beam/REPORT.md). We audited LoCoMo's reference answers and excluded the questions that our audit or an external one flagged. We also left out open-domain questions, which need knowledge from outside the conversation. As is now common for LoCoMo, an LLM judge grades the answers, with one prompt for all questions; Section 5 compares its rules with those of other published LoCoMo judges.

Section 2 introduces LoCoMo and the parts of cognee used here. Section 3 describes the method: how the conversations are ingested, how questions are answered, and the evaluation protocol, including the scope and the grading rules. Section 4 reports the results, overall and by category. Section 5 discusses an additional experiment at the retriever's default depth, how our grading compares with published LoCoMo judges, and what the evaluation doesn't show. Section 6 concludes, and Appendix A gives the models, configurations, rerun commands, audits and artifacts behind the results.

## 2. Preliminaries

### 2.1 LoCoMo

LoCoMo ([paper](https://arxiv.org/abs/2402.17753), [repository](https://github.com/snap-research/locomo)) has a public release, `locomo10.json`, with 10 conversations between two people, a subset of the dataset described in the paper. Each conversation has 19–32 dated sessions and 369–689 turns, and the release has 1,986 questions about them. The questions fall into five categories: single-hop (841), multi-hop (282), temporal (321), open-domain (96) and adversarial (446). Open-domain questions combine what a speaker said with general knowledge. Adversarial questions cannot be answered from the conversation: the expected response is to say so, and the dataset pairs each with a plausible wrong answer. In most of them, the question names one speaker but the cited evidence comes only from the other (§3.2).

The [original scorer](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py) measures token overlap (F1) with the reference answer (also called the gold answer), and counts an adversarial answer as correct when it contains "not mentioned" or "no information available"; many published evaluations use an LLM judge instead. §3.3 describes the dataset limitations and evaluation choices relevant to this study.

### 2.2 cognee

cognee builds memory with `remember()`, which runs `add` (register the documents), `cognify` (split them into chunks, extract entities and relationships into a knowledge graph with an LLM, summarize the chunks, and embed the results for vector search) and `improve`, which in the version used here (1.5.4) embedded the graph's `subject–relationship–object` triplets for search (Appendix A.2 describes how it has changed since). At question time, a retriever selects context from this memory, and a reader LLM answers from it under an answer prompt.

The **hybrid** retriever used here combines three channels in one pass: conversation passages (chunks), entities with their relationships, and further relationships (facts) not already listed with an entity. The main configuration uses only these three channels; other cognee retrievers and memory features, such as time-aware retrieval, session memory and feedback-based improvement, were not used. The additional experiment (§5) adds the global context index, which in the version used here held summaries built over groups of chunk summaries, searched together with the three channels.

## 3. Methodology

The experiment measures accuracy: the share of evaluated LoCoMo questions that cognee's memory pipeline answers correctly, as graded by an LLM judge. Each conversation is ingested into its own memory (§3.1); each question is answered from that memory by one retrieval configuration under one answer prompt (§3.2) and graded correct or wrong against its reference answer (§3.3). The configuration in §5 changes the ingestion and retrieval settings and keeps the questions, answer prompt and judge.

### 3.1 Data Ingestion

The source is `locomo10.json` from the LoCoMo repository. Every session is cut into consecutive windows of six turns. A session's last window holds the turns left over, and a single leftover turn joins the window before it, so the ten conversations give 1,048 windows, 813 of them with exactly six turns. Each window starts with a header naming the session, its date and time, and the two speakers, so the date stays next to the events it dates. A shared photo is represented by the dataset's machine-generated caption; the photo itself isn't ingested. A short overview document per conversation lists the speakers and the session timeline. The dataset's session summaries and observations, which were generated from the conversations, are not ingested.

Each conversation's documents go through one `remember()` call into a separate memory, with cognee's standard extraction and summary prompts. The chunk size (4,096 tokens) is larger than any document, so each window and each overview becomes exactly one chunk. Ingestion used `gpt-5-mini` for extraction and summaries and `text-embedding-3-large` for embeddings. Excluding a question from evaluation removes nothing from ingestion: every conversation is stored in full.

### 3.2 Question Answering and Retrieval

For each question, the hybrid retriever (§2.2) selects context from the conversation's memory, and `gpt-5-mini` answers from it. [One answer prompt](qa_prompts/unified.txt) is used for every question, and the question is asked as written: neither its category label nor its gold answer reaches the answering model. The labels carry information that an ordinary question does not, such as that an adversarial question has no answer in the conversation. Appendix A.2 gives the exact retrieval configuration and the reproduction command.

The prompt asks for the shortest complete answer, collects every item when the question asks for several, and resolves relative dates against the date of the session in which they were said. It asks for dates as "D Month, YYYY", and for a relative period in the form LoCoMo's reference answers use, such as "the week before 10 March, 2021". The rest of the prompt decides when to answer and when to decline, and was written while developing the evaluation on these conversations:

- If a question contains would, might, could, likely, unlikely, probably, possibly or suspected, the model must give its most likely inference ("Likely yes") and may not decline. Only 44 of the 1,986 questions use these words, 43 of them open-domain, and all 44 are excluded, so this rule applies to none of the evaluated questions.
- For any other question, the model first checks that the context attributes the fact to the person named. The prompt says that questions are sometimes built from what the other speaker did or said, with the name swapped, or from the person's own fact with one detail changed, such as another object, place or recipient. In those cases it tells the model to reply "Not mentioned in the conversation".
- A closing list names situations that are not reasons to decline, such as a question dated to the right month of the wrong year.

We based the attribution rule on how LoCoMo's adversarial questions are built: of the 443 that name one speaker and cite evidence, 333 cite only turns spoken by the other speaker. Using one prompt for every question keeps category labels out of the answering step. The decline rules, though, come from how LoCoMo builds its adversarial questions, so the adversarial score measures how well the system declines questions built that way. We haven't measured how the scores change without these rules, including on answerable questions, which the model sometimes declines (§4.2).

### 3.3 Evaluation Protocol

**Scope.** We evaluate single-hop, multi-hop, temporal and adversarial questions, and leave out open-domain questions and questions whose reference answer an audit flags. Open-domain questions need general knowledge, so they test more than memory. Adversarial questions stay in, because recognizing that a conversation holds no answer is part of conversational memory.

We use an external audit and our own (Appendix A.3) to find reference answers that are wrong or that the conversation text doesn't support; a wrong reference answer can make a faithful response count as an error. For example, both audits flag the question "When did Evan have a drunken night with his friends?" in conversation `conv-49`: the reference answer is "January 9, 2023", but in a session dated 10 January 2024 Evan says that it happened "yesterday", which makes the date 9 January 2024. We built the exclusion list from the questions and reference answers alone, without looking at any of cognee's answers: the external audit first, then ours, to catch errors it missed.

The 224 excluded questions are the 96 open-domain questions and the 143 questions flagged by either audit, 15 of which are also open-domain, so 1,762 of the 1,986 questions are evaluated. [Here's the full exclusion list](report_artifacts/gold_exclusions.json), with each audit's error type or label and reason. §4.1 also scores the excluded questions.

**Judge.** An LLM judge, `gpt-5.1` with medium reasoning effort, grades each answer as CORRECT or WRONG in a separate call, so the answering model never grades its own output. The judge sees the question, its category, the gold answer and the response; it does not see the retrieved context or the conversation.

Its [prompt](prompts/locomo_judge_prompt.txt) accepts paraphrases and equivalent formats. A date must be consistent with the gold answer and at least as precise: "27 May, 2023" is correct for the gold "May, 2023", and "2023" is wrong for "7 May, 2023". The prompt also counts "the week before 9 June, 2023" and "around 2 June, 2023" as equivalent. A duration must be the same length, rounding allowed ("six months" for "half a year").

For a gold answer with several items, reasons or parts, a response is correct when it gives at least half of them, rounding up, and contradicts none; in the prompt's own example, "for work" is correct for the gold "she moved for work and to be near her family". The half rule replaced an earlier version of the judge prompt that required every item (commit `c6e49bf`, 6 September 2026), the reason being that a judge that can only say CORRECT or WRONG has to round partial answers one way or the other, and requiring every item rounds a nearly complete list down to WRONG, so we adopted the usual rounding: at least half of the items counts as correct.

Both published runs used the half rule, and its hash is in each run record. Compared with requiring every item, it raises scores on list questions, and we haven't regraded the answers under the stricter rule to measure by how much.

Extra details are allowed unless they contradict the gold, and so is hedging around a correct core answer. A response that offers several alternatives where only one matches is wrong, as is one that declines when the gold answer is a concrete fact. For adversarial questions, the judge is told that the question can't be answered from the conversation and that only declining is correct, and the dataset's plausible wrong answer is named as an example of a wrong response. See §5 and Appendix A.5 for comparison with other LoCoMo judges.

**Runs.** Two published runs of the main configuration, on 9 and 13 October 2026, answered and judged the questions on the same memories, built by one ingestion per conversation.

## 4. Results

### 4.1 Main Result

cognee answered about **91%** of the 1,762 evaluated questions correctly: 1,598 (90.7%) and 1,608 (91.3%) in the two published runs (§3.3). The runs share their memories, so the difference comes from answering and judging. The answer text differed for 556 of the 1,762 questions, and 70 verdicts changed: 40 from wrong to correct and 30 the other way (exact McNemar test, p = 0.28). The 95% bootstrap interval of the 91.3% run is 90.0–92.5%.

The table also scores the excluded questions, from the unfiltered records at commit `012e8a0` (Appendix A.6). With nothing excluded, scoring all 1,986 questions against LoCoMo's reference answers as published, with the same judge, the runs score 85.2% and 85.6%. The flagged questions score about a third, as expected when many of their reference answers are wrong or describe photos the system never sees (Appendix A.3).

| Questions | 9 October | 13 October |
| --- | --- | --- |
| Evaluated (1,762) | 1,598 (90.7%) | 1,608 (91.3%) |
| All questions, original reference answers (1,986) | 1,693 (85.2%) | 1,701 (85.6%) |
| Open-domain (96) | 53 (55.2%) | 50 (52.1%) |
| Open-domain, not flagged by an audit (81) | 48 (59.3%) | 47 (58.0%) |
| Flagged by an audit (143) | 47 (32.9%) | 46 (32.2%) |

### 4.2 By Category

The table breaks down both runs by category. The [13 October run](report_artifacts/topk20_run2/) is the one in `report_artifacts/`; the records of the 9 October run are at commit `012e8a0` (Appendix A.6). The intervals are 95% bootstrap intervals computed by [`aggregate.py`](aggregate.py).

| Category | Questions | 9 October | 13 October | 95% interval, 13 October |
| --- | --- | --- | --- | --- |
| Single-hop | 796 | 731 (91.8%) | 736 (92.5%) | 90.5–94.3% |
| Multi-hop | 236 | 211 (89.4%) | 212 (89.8%) | 86.0–93.6% |
| Temporal | 284 | 258 (90.8%) | 256 (90.1%) | 86.6–93.7% |
| Adversarial | 446 | 398 (89.2%) | 404 (90.6%) | 87.9–93.3% |
| **Total** | **1,762** | **1,598 (90.7%)** | **1,608 (91.3%)** | **90.0–92.5%** |

The categories are close together, and their order changes between the runs: adversarial questions score lowest on 9 October, multi-hop questions on 13 October. All four intervals overlap, so we don't rank the categories.

On adversarial questions, the model declined 403 of 446 in the 13 October run (395 on 9 October), and the judge accepted one further answer that wasn't a decline (three on 9 October). The model also declined 19 of the 1,316 answerable questions (20 on 9 October). Without adversarial questions, the runs score 91.2% and 91.5% (1,200 and 1,204 of 1,316).

## 5. Discussion

**Default retrieval depth.** The main configuration retrieves 20 passages, 20 entities and 20 facts per question, four times the hybrid retriever's default of 5 each. The additional experiment uses the default depth and scored **89.7%** (1,580 of 1,761 scored questions) in one published run; counting its unparseable verdict (A.4) as wrong gives the same 89.7%. It also changes ingestion: relative dates are annotated before ingestion, the extraction and summary prompts are adapted to conversations, and a global context index is built and searched (Appendix A.2 and A.4). With several changes at once and a single run, the result doesn't show what depth alone does.

Question by question, the additional experiment answered 52 questions correctly that the 13 October run got wrong, and 79 the other way (exact McNemar test, p = 0.02); against the 9 October run, the counts are 62 and 79 (p = 0.18). A later rerun of the additional experiment's scripts on cognee 1.6.3 scored 89.0% (A.4).

**Grading.** Our judge's rules (§3.3) differ from those of other published LoCoMo judges (Appendix A.5). On dates, it requires consistency with the reference at the same precision or better, while Mem0's current benchmark accepts dates within 14 days, and the topical-match prompt used by Mem0's legacy evaluator, Zep and Supermemory accepts any answer that refers to the same date or time period. On multi-part answers, it gives credit for half the items: stricter than Mem0's current one-item rule and the topical-match rule, looser than Supermemory's earlier prompt, which required every part. Our headline also includes adversarial questions, which several of these evaluations leave out; §4.2 gives our score without them.

**What the evaluation doesn't show.**

- **Results on new conversations.** The windowing, the answer prompt, the judge prompt and the retrieval depth were developed on these ten conversations, and several date examples in the judge prompt are LoCoMo reference answers. Unlike in our BEAM evaluation at 100K tokens, no conversation was held out, so the results show what this setup does on the data it was tuned on.
- **The effect of the decline rules.** They follow how LoCoMo's adversarial questions are built (§3.2), and we haven't measured the scores without them, on adversarial or on answerable questions; on 13 October the model declined 19 of the 1,316 answerable questions.
- **That every exclusion is right.** The exclusions combine an external audit with our own LLM audit, of whose 93 flags we spot-checked only some by hand. Publishing the reasons lets others check them, but it doesn't settle every case, and both audits checked non-adversarial answers only.
- **How often the judge errs.** One LLM judge graded every answer, and we haven't compared its verdicts with human grading. The records show some of its mistakes: it accepted "Congrats!" as a correct response to an adversarial question in both runs, and one of its verdicts in the additional experiment came back as the malformed label `WRNG` (A.4).
- **Stability across fresh ingestions.** Both main runs read the same memories, and 70 verdicts still changed between them. Rebuilding the memories adds variation of its own, which we haven't measured.
- **The knowledge graph's contribution.** We didn't ablate the retrieval channels, so the runs don't show how much the graph adds over passage retrieval alone.
- **Performance at scale.** Each LoCoMo conversation is well within a current model's context window, while BEAM runs from 100,000 to 10 million tokens, and this report has no full-context baseline in which the answering model reads the whole conversation.

## 6. Conclusion

cognee answered 90.7% and 91.3% of the 1,762 evaluated LoCoMo questions in two runs under our LLM judge, with its existing memory components and a setup developed on the same ten conversations. An additional experiment at the retriever's default depth, with adapted ingestion, scored 89.7% in one published run. Together with our earlier [BEAM](../beam/REPORT.md) evaluation, these are two public results for the same memory components on different conversational memory tasks.

Four tests would tell more than further runs on this data: held-out conversations, the answer prompt without its decline rules, a check of the judge against human grading, and measured token and latency costs for both configurations. The prompts, configurations, exclusions and per-question records are published so that others can rerun or regrade them.

## Appendix A. Reproducibility and Supporting Artifacts

### A.1 Models and Environment

| Stage | Model |
| --- | --- |
| Ingestion: graph extraction and summaries | openai/gpt-5-mini |
| Embeddings | openai/text-embedding-3-large |
| Date annotation (additional experiment only) | openai/gpt-5-mini, low reasoning effort |
| Answer generation | openai/gpt-5-mini, default reasoning effort |
| Judge, in a separate call | openai/gpt-5.1, medium reasoning effort, at most 6,000 completion tokens |

The published runs used cognee 1.5.4, Ladybug 0.19.0 (graph), LanceDB 0.38.0 (vectors), SQLite (relational) and litellm 1.96.2, as recorded in each run's `run_config.json`. The benchmark package was added to the repository later, at version 1.6.3 (A.2).

Environment: `ENABLE_BACKEND_ACCESS_CONTROL=false`, `CACHING=true`, `AUTO_FEEDBACK=false`.

The main configuration was ingested on 7 October 2026; its published runs were answered and judged on 9 October (90.7%) and 13 October 2026 (91.3%).

The additional experiment was ingested, answered and judged on 12 October 2026.

The dataset file has SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; each run's `run_config.json` also records the hashes of the answer and judge prompts, and the additional experiment's also records the hashes of its ingestion prompts.

### A.2 Configurations and Rerun Commands

| Setting | Main configuration | Additional experiment (§5) |
| --- | --- | --- |
| Windows | 6 turns with a dated header, one chunk each (4,096-token chunk size) | same |
| Relative dates | as written | annotated with absolute dates before ingestion ([prompts/resolve_dates.txt](prompts/resolve_dates.txt)) |
| Extraction prompt | cognee default | [prompts/graph_extraction_conversations.txt](prompts/graph_extraction_conversations.txt) |
| Summary prompt | cognee default | [prompts/summarize_content_questions.txt](prompts/summarize_content_questions.txt) |
| Global context index | none | built after remember(); 3 entries per question |
| Retriever | hybrid_completion, [configs/locomo_hybrid_20_20_20.json](configs/locomo_hybrid_20_20_20.json) | hybrid_completion, [configs/locomo_hybrid_5_5_5_gci.json](configs/locomo_hybrid_5_5_5_gci.json) |
| Passages / entities / facts per question | 20 / 20 / 20 | 5 / 5 / 5 (the retriever's defaults) |
| Relationships per entity | 10 | 10 |
| Answer prompt | [qa_prompts/unified.txt](qa_prompts/unified.txt), the same for every question | same |
| Judge prompt | [prompts/locomo_judge_prompt.txt](prompts/locomo_judge_prompt.txt) | same |
| Run record | [topk20_run2/run_config.json](report_artifacts/topk20_run2/run_config.json) | [topk5_adapted_prompts/run_config.json](report_artifacts/topk5_adapted_prompts/run_config.json) |

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

Each command creates a fresh run directory, ingests the conversations, answers and judges the evaluated questions, and writes `locomo_summary.json` and `locomo_summary.md` there.

The dataset is downloaded to `temp/locomo_data/locomo10.json` when missing; check its SHA-256 against A.1.

Values in a `.env` file take precedence over the settings the driver passes to its child processes, such as `LLM_MODEL` and the database locations, so keep the file consistent with the driver's options.

Answering and judging vary between runs (§4.1), so identical scores are not guaranteed. More options are documented in the [LoCoMo README](README.md).

To recompute the summary of the 13 October run from its stored verdicts, without new model calls:

```bash
uv run python -m cognee.eval_framework.locomo.aggregate \
  cognee/eval_framework/locomo/report_artifacts/topk20_run2
```

That checks only the arithmetic; it doesn't re-examine the answers or verdicts.

The published runs used cognee 1.5.4. This package was added to the repository later, at version 1.6.3, and the commands run the cognee code of the checkout, so a run from main builds its memories with newer code than the published runs. As of version 1.6.8 (main on 16 November 2026), `improve()` no longer embeds triplets, and by default it builds a global context index and reviews dated facts for conflicts; the global context index pipeline also builds its index from entity counts, without an LLM, by default. So a checkout of main doesn't rebuild the memories behind the published results. Record the commit you run, and compare the dataset and prompt hashes with A.1.

### A.3 Evaluation Scope and Audits

The two audits behind the exclusions in §3.3 are:

- **External audit** by Penfield Labs ([dial481/locomo-audit](https://github.com/dial481/locomo-audit)): 99 questions with score-corrupting errors among the 1,540 non-adversarial questions, by error type: `HALLUCINATION` (33), `TEMPORAL_ERROR` (26), `ATTRIBUTION_ERROR` (24), `AMBIGUOUS` (13) and `INCOMPLETE` (3). Its 57 citation-only errors (`WRONG_CITATION`), where the gold answer is right and only the cited evidence is wrong, are kept.
- **Our audit**: every non-adversarial gold answer checked by `gpt-5.1` with medium reasoning effort against the conversation and its session dates. Like our pipeline, it read the photo captions, not the photos. The criteria are in [`gold_audit_prompt.txt`](report_artifacts/gold_audit_prompt.txt). Gold answers labeled `WRONG`, `UNSUPPORTED`, `MALFORMED` or `AMBIGUOUS` with high or medium confidence are excluded; answers labeled `IMPRECISE` or `INFERRED` stay in. That gives 93 questions, each listed with its label and the reason for it: 44 unsupported, 39 wrong, 7 ambiguous and 3 malformed, all labeled with high confidence except one unsupported. We spot-checked some of the 93 flags by hand, not all of them.

Forty-nine questions are flagged by both audits, so 143 are flagged in all: 45 single-hop, 46 multi-hop, 37 temporal and 15 open-domain.

Not every flag means the reference answer is wrong. Of our audit's 44 UNSUPPORTED labels, 33 cite a turn where someone shared a photo, against 45% of non-adversarial questions overall, and some of those reference answers describe the photo. For "What symbols are important to Caroline?" (`conv-26`, flagged by both audits), the reference answer includes a transgender symbol; the photo's search query in the dataset is "pendant transgender symbol", but its caption reads "a photo of a person holding a necklace with a cross and a heart". Excluding questions like that makes sense for a pipeline that reads text and captions only, but their reference answers aren't necessarily wrong, and leaving them out raises the score.

The benchmark package applies this scope before retrieval: [`scope.py`](scope.py) defines it, and [`answer.py`](answer.py) drops excluded questions, so they are never answered or judged; question ids are unchanged. The published runs answered and judged all 1,986 questions; their records in `report_artifacts/` were filtered to the evaluated questions afterwards, leaving every kept record unchanged. The filter is in the [artifact README](report_artifacts/README.md#filtering-existing-run-records).

### A.4 Additional Experiment

Before ingestion, relative time expressions in each window were annotated with the date or period they refer to, computed from the session date, for example "next month [= the month after 23 March 2023, i.e. April 2023]". An annotation output was discarded, and the original window kept, when removing its annotations did not give back the original text word for word. 414 of the 1,048 windows received at least one annotation, 596 in all, and 5 outputs were discarded. The extraction prompt adds rules to name the person in every fact, date every event, and keep the date of facts that can change. The summary prompt also has the summarizing model write 3–5 questions that someone could later ask about the chunk's details; it sees only the chunk, and ingestion never reads the benchmark's questions.

One judge verdict could not be parsed: conversation `conv-47`, question 163, where the judge replied with the label `WRNG`. Its score is left empty, so the experiment has 1,762 evaluated questions and 1,761 scored ones. Counting it as wrong gives the same 89.7% (1,580 of 1,762).

The saved run was produced by earlier scripts with the same prompts and settings as the command in A.2, which has not been rerun end to end. A single rerun of those scripts on cognee 1.6.3, which rebuilt the memories, scored 89.0%, against 89.7% on 1.5.4. That 0.7-point difference is close to the 0.6 points between the two runs of the main configuration, which shared their memories, so one run on each version can't show whether the version made a difference. The rerun's records are not published.

### A.5 Judge Comparison

The published LoCoMo judges below were inspected on 19 October 2026 at the linked revisions. They describe those revisions, not necessarily the setup behind every published score.

| Implementation | What its LoCoMo judge accepts | Source |
| --- | --- | --- |
| Mem0, current benchmark | One matching item from a reference list; dates within 14 days; durations within 50%. Adversarial questions are excluded. | [prompts and category selection](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/locomo/prompts.py) |
| Mem0, legacy evaluation | A response that "touches on the same topic" as the reference, and a date that refers to the same date or time period; GPT-4o-mini, binary verdicts. Adversarial questions are excluded. | [judge](https://github.com/mem0ai/mem0/blob/aae5989e78a6188b3b047c104d960c9ad0927e75/evaluation/metrics/llm_judge.py) |
| Zep, published evaluation | The same topical-match rule; GPT-4o-mini, binary verdicts. | [judge](https://github.com/getzep/zep-papers/blob/4434662f9ddd0f5441df4987fdfdff13362fb055/kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py) |
| Hindsight, current Agent Memory Benchmark | The LoCoMo-specific judge repeats the topical-match rule. Adversarial questions are excluded. | [LoCoMo dataset and prompt](https://github.com/vectorize-io/agent-memory-benchmark/blob/aef3702e0586a959c845cc2c61f10d5e8ff0f94a/src/memory_bench/dataset/locomo.py), [prompt selection](https://github.com/vectorize-io/agent-memory-benchmark/blob/aef3702e0586a959c845cc2c61f10d5e8ff0f94a/src/memory_bench/judge.py) |
| Supermemory, current LoCoMo path | Mem0's legacy topical-match prompt, for every provider, adopted in [PR #96](https://github.com/supermemoryai/memorybench/pull/96) (merged 19 October 2026); before it, a generic prompt that required every part of the answer. Adversarial questions are scored separately and left out of the headline. | [LoCoMo prompts](https://github.com/supermemoryai/memorybench/blob/e8d6e1f40e8d0cca1b90463af2776e05efccc720/src/benchmarks/locomo/prompts.ts), [dataset settings](https://github.com/supermemoryai/memorybench/blob/e8d6e1f40e8d0cca1b90463af2776e05efccc720/src/benchmarks/locomo/index.ts) |
| cognee, this evaluation | A date consistent with the reference and at least as precise; at least half the items of a multi-part answer, rounding up, with no contradictions; gpt-5.1 with medium reasoning effort, binary verdicts. Adversarial questions are included, and only declining is correct. | [judge prompt](prompts/locomo_judge_prompt.txt) |

### A.6 Artifacts and Provenance

The [`report_artifacts/`](report_artifacts/) folder contains, for the 13 October run of the main configuration ([`topk20_run2/`](report_artifacts/topk20_run2/)) and the additional experiment ([`topk5_adapted_prompts/`](report_artifacts/topk5_adapted_prompts/)):

- per-question answers and judge verdicts for the 1,762 evaluated questions, one file per conversation, with the original question ids;
- the summary tables behind §4 and §5, regenerated from those files by [`aggregate.py`](aggregate.py);
- the run configuration with models, settings, dates and prompt hashes.

It also contains both ingestion reports with their window maps, the date annotations of the additional experiment, the exclusion list and the prompt of our audit. Its [README](report_artifacts/README.md) describes the layout and the attribution for the dataset and the external audit.

The 9 October run of the main configuration scored 1,598 of 1,762 evaluated questions (90.7%). Its complete records are at [commit](https://github.com/topoteretes/cognee/tree/012e8a039b5116bc30325a9c10192d9c9336e399/cognee/eval_framework/locomo/report_artifacts/qa) `012e8a0`, which also holds the unfiltered records of `topk20_run2/` and `topk5_adapted_prompts/`.
