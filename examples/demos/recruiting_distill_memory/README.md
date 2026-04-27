# Recruiting Demo — Distilled Human Memory → Grounded Agent

A startup (Ledgerline, a 40-person payments fintech) is automating parts of its
recruiting workflow. Rather than letting the AI agent build its own memory
from scratch, they want it to **consult their senior recruiter's knowledge**
when it acts. This demo shows that end-to-end on Cognee.

**The arc, in four beats:**
1. Alex (senior recruiter) contributes a playbook and a rulebook. An ingest
   pipeline persists the rulebook as typed `Rule` graph nodes in a
   `human_memory` dataset.
2. The agent replaces Alex for a narrow task — drafting an interview loop
   for a new candidate. Its tools are wrapped with `@cognee.agent_memory`
   against `human_memory`. A tiny planner LLM picks which tool to call next.
3. We run the agent across **four candidates × two modes** (naive = no
   memory, grounded = retrieves the rulebook). The naive plans violate
   Ledgerline's policies almost everywhere; the grounded plans follow them
   almost everywhere. `check_rule_compliance.py` prints the full matrix.
4. After the grounded run, `cognee`'s built-in memify pipeline extracts
   "learnings" from the decorator's session traces and persists them into
   the graph under `node_set=['agent_proposed_rule']`. A human reviews
   those nodes and promotes accepted ones into proper `Rule` entries.

## Why this shape

- **Tools are unaware they're being observed.** They return pure domain
  objects — no citation fields, no "propose a rule" language. Extraction
  happens outside the agentic loop.
- **The decorator does the trace capture and the memify.** Each call to
  a `@cognee.agent_memory`-wrapped tool produces a trace entry with an
  LLM-generated `session_feedback` summary. After the loop ends we call
  `persist_agent_trace_feedbacks_in_knowledge_graph` — the exact memify
  pipeline the decorator can auto-invoke — to cognify those summaries
  into graph nodes tagged `agent_proposed_rule`.
- **Humans gate writes to the rulebook.** `review_pending_rules.py`
  surfaces every `agent_proposed_rule` node, and only explicitly approved
  ones become `Rule` DataPoints on `node_set=['rule','approved','agent_authored']`.

## Scenario

- **Company:** Ledgerline — 40-person payments fintech based in Berlin.
- **Role:** Staff Backend Engineer.
- **Recruiter being replaced:** Alex Chen.
- **Candidates:** 4 — covering the rule-triggering matrix.

### The rules are deliberately weird

The six seed rules (`data/seed_rules.yaml`) are designed to **contradict
industry defaults and depend on Ledgerline-specific facts**. A strong
pretrained LLM will guess "90-min live coding, ≥4 hours, CTO on panel"
without any help. Memory only does interesting work where pretraining
is *wrong*, so each rule below picks an off-by-default number, a
specific internal name, or a strict position requirement:

| id                      | trigger                                             | action                                                                                          | why a pretrained LLM gets it wrong          |
|-------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------|
| `R1_80_min_live_coding` | Staff-level scheduling                              | Exactly **80-min live coding**, no take-home                                                     | default guess is 60 or 90                    |
| `R2_exact_panel`        | Staff-level panel                                   | Exactly 4 panelists by name: **Sam (CTO), Jordan (VP Eng), Leila (Staff BE), Ravi (SPM)**       | names are unguessable                        |
| `R3_medium_onsite`      | Staff-level interviews                              | **Onsite only** in Berlin, never video                                                           | default is remote-first                      |
| `R4_noncompete_first`   | prior_company ∈ {Stripe, Plaid, Adyen}              | "**non-compete**" must be the **first** `disclosure_questions` entry, verbatim                   | LLM places it anywhere in the list           |
| `R5_streamtap_mention`  | Screening invite                                    | Body must mention **"streamtap"** (Ledgerline's OSS Kafka fork) verbatim                         | product name is unguessable                  |
| `R6_hours_exactly_4`    | Staff-level loop                                    | `total_hours` exactly **4.0**, not 4.5, not 5                                                    | generic floor ≥4 happens to coincide here    |

### Candidates

| name          | prior_company | triggers R4? | purpose                          |
|---------------|---------------|--------------|----------------------------------|
| Dev Rao       | Stripe        | yes          | full path, all rules apply       |
| Maria Cruz    | Revolut       | no           | tests R4 does not over-fire      |
| Arjun Mehta   | Plaid         | yes          | R4 via a different company       |
| Priya Sharma  | Google        | no           | no R4, different stack           |

## Layout

```
data/
  alex_playbook.md                  prose rulebook (LLM context at retrieval)
  seed_rules.yaml                   hand-authored Rule records
  candidates/{dev_rao,maria_cruz,arjun_mehta,priya_sharma}.json
rule_models.py                      Rule + plain domain tool-output models
ingest_human_memory.py              ingest rulebook → human_memory dataset
agent_tools.py                      three @agent_memory-decorated tools (pure domain)
agent_loop.py                       minimal planner loop: LLM picks next tool
_run.py                             shared runner — loop + post-loop memify
run_naive.py                        with_memory=False, honours RECRUITING_CANDIDATE
run_grounded.py                     with_memory=True,  honours RECRUITING_CANDIDATE
run_matrix.py                       subprocess through all candidates × both modes
check_rule_compliance.py            PASS/FAIL matrix across every plan in output/
review_pending_rules.py             human approval CLI for agent-proposed nodes
inspect_rulebook.py                 prints all Rule nodes in the graph
visualize.py                        renders human_memory graph HTML
output/                             generated plan JSONs (gitignored)
```

## Prerequisites

1. Working Cognee install (see repo root CLAUDE.md / README).
2. `.env` at the repo root with `LLM_API_KEY` / `LLM_MODEL` set. Defaults
   (OpenAI + local Kuzu/LanceDB/SQLite) are fine.
3. `ENABLE_BACKEND_ACCESS_CONTROL` at default (`true`): per-user storage
   isolation matters — review/inspection scripts run inside
   `run_custom_pipeline` so they share the scope the grounded run wrote to.

## Run the demo

All commands below are run from the **repo root**.

```bash
# 1. Ingest the rulebook into human_memory (add --reset to prune first)
python -m examples.demos.recruiting_distill_memory.ingest_human_memory

# 2. Run the full 4-candidate × 2-mode matrix (~30s per combo)
python -m examples.demos.recruiting_distill_memory.run_matrix

# 3. The payoff table: every rule × every candidate, naive vs grounded
python -m examples.demos.recruiting_distill_memory.check_rule_compliance

# 4. Review agent-proposed nodes — approve any worth codifying
python -m examples.demos.recruiting_distill_memory.review_pending_rules

# (Optional) Single-candidate runs
RECRUITING_CANDIDATE=maria_cruz python -m examples.demos.recruiting_distill_memory.run_grounded
RECRUITING_CANDIDATE=dev_rao    python -m examples.demos.recruiting_distill_memory.run_naive

# (Optional) Inspect / visualize the rulebook
python -m examples.demos.recruiting_distill_memory.inspect_rulebook
python -m examples.demos.recruiting_distill_memory.visualize
```

## Expected payoff

After step 3 you should see something like this — naive is near-zero,
grounded is near-perfect:

```
Rule                               | arjun N  arjun G | dev N  dev G | maria N maria G | priya N priya G
---------------------------------------------------------------------------------------------------------
Format == live_coding              | FAIL    PASS     | FAIL   PASS  | FAIL    PASS    | FAIL    PASS
Duration == 80 min                 | FAIL    PASS     | FAIL   PASS  | FAIL    PASS    | FAIL    PASS
Panel == {Sam,Jordan,Leila,Ravi}   | FAIL    PASS     | FAIL   PASS  | FAIL    PASS    | FAIL    PASS
Medium == onsite                   | FAIL    PASS     | PASS   PASS  | FAIL    PASS    | FAIL    PASS
Non-compete is FIRST question      | FAIL    PASS     | FAIL   PASS  | --      --      | --      --
Body mentions 'streamtap'          | FAIL    PASS     | FAIL   PASS  | FAIL    FAIL    | FAIL    PASS
total_hours == 4.0                 | PASS    PASS     | PASS   PASS  | PASS    PASS    | PASS    PASS
---------------------------------------------------------------------------------------------------------
TOTAL                              | 1/7     7/7      | 2/7    7/7   | 1/6     5/6     | 1/6     6/6

Aggregate across 4 candidates: naive 5/26, grounded 25/26
```

**Reading the naive column.** The base model (Baseten `gpt-oss-120b`) hits
almost every rule wrong: picks `system_design` over live coding, 60 or 90
minutes instead of 80, video instead of onsite, 4–5 made-up placeholder
panelists ("Alice — CTO, Bob — Engineering Manager…"), misses the streamtap
mention entirely. The only two freebies are:
- **R6 total_hours == 4.0** — generic best-practice floor happens to coincide
  exactly with Ledgerline's policy. Trivial pass.
- **Dev Rao R3 onsite** — a single lucky guess out of four attempts. Not a
  pattern, just base-model variance.

**Reading the grounded column.** One rule predicate out of 26 fails —
Maria's screening invite doesn't mention streamtap. That's real retrieval
noise: her candidate query ("ex-Revolut, London") is semantically further
from R5's trigger than other candidates' queries are. Raising `memory_top_k`
from 6 to 8 or tightening R5's trigger string fixes it. The interesting
data point is that **the pipeline is not magic** — the demo is honest
about that.

## The review gate — agent proposes, human approves

Each grounded run's session traces are cognified into
`agent_proposed_rule` graph nodes. Running
`review_pending_rules.py --auto-approve` after the matrix will surface a
mix of proposals per candidate — in practice most are noise:

- **Duplicates** of existing seed rules at varying specificity (e.g. an
  "80-minute live coding for Staff Backend" proposal when `R1` already
  says exactly that).
- **Fixation on panelists**: the memify pipeline sometimes latches onto
  the four panelists' names (Sam, Jordan, Leila, Ravi) as if they were
  candidate attributes, producing rules like "VP of Engineering panel
  inclusion".
- **Hallucinations**: because summaries are one-sentence LLM output,
  occasionally a proposal invents numbers — e.g. a fabricated "Senior
  Product Manager salary floor" derived from Ravi (a panelist) having
  that title.
- **Occasional keepers**: a plausible generalization of a screening-invite
  behaviour, for instance, that a human might want to codify.

This is exactly why the gate matters. In production, every agent
proposal requires an explicit `[a]pprove` keypress before it becomes a
`Rule`. `--auto-approve` exists for CI and for demos that want to stress
the quality distribution honestly.

Human-authored vs agent-authored rules are kept cleanly separable:

| field             | human-authored               | agent-authored                           |
|-------------------|------------------------------|------------------------------------------|
| `source`          | `alex_playbook`              | `agent_proposal:<node_uuid>`             |
| `belongs_to_set`  | `['rule','approved','human_authored']` | `['rule','approved','agent_authored']`  |
| `rule_id` prefix  | `R1`…`R6` (seeded)           | monotonically assigned (`R7`, `R8`, …)   |

## How the agentic loop works

`agent_loop.run_agentic_plan(candidate)` runs a small planner loop:

1. Build a `candidate_summary` string.
2. The planner LLM sees the candidate + any tools already called + a
   one-sentence description of each remaining tool. It returns
   `next_tool ∈ {propose_interview_format, schedule_panel,
   compose_screen_invite, done}` plus one-sentence reasoning.
3. If `next_tool == 'done'`, return. Otherwise call the tool (through
   `@cognee.agent_memory`, so memory is retrieved and the trace is saved)
   and append the result. Go back to step 2. Max 8 steps as a safety cap.

The tools themselves are pure domain functions — they have no idea the
planner chose them, no idea the decorator wraps them, no idea the traces
they leave behind will be mined for new rules.

## Troubleshooting

**`review_pending_rules.py` finds no nodes**
  Either the grounded run hasn't happened in this installation, or the
  memify pipeline didn't tag its output. Check `run_grounded.py` output
  for the "Memifying session traces" log line. The review script filters
  by `belongs_to_set` containing `agent_proposed_rule`.

**Grounded run still fails a compliance rule**
  Inspect the raw session trace (SessionManager stores it under your
  user's ACL scope). Usually the summarizer dropped the full rule_id
  prefix or the planner skipped a tool. Raising `memory_top_k` in
  `agent_tools.py` helps.

**Naive run passes everything**
  The base model occasionally picks "live coding" unprompted. Tighten the
  tool's system prompt to emphasize Ledgerline-specific policy over
  generic best practice so there's no free-lunch from pretraining.
