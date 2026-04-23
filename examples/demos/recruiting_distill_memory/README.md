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
3. We run the agent **twice** on the same candidate (Dev Rao, ex-Stripe):
   once **naive** (no memory), once **grounded** (retrieves the rulebook).
   The naive plan violates Ledgerline's policies; the grounded plan follows
   them.
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

- **Company:** Ledgerline — 40-person payments fintech
- **Role:** Staff Backend Engineer
- **Recruiter being replaced:** Alex Chen
- **Demo candidate:** Dev Rao — 8 YOE, prior at Stripe, target base $170k

The six seed rules (`data/seed_rules.yaml`):

| id                       | trigger                                              | action                                    |
|--------------------------|------------------------------------------------------|-------------------------------------------|
| `R1_live_coding`         | Staff-level scheduling                                | 90-min live coding, never take-home       |
| `R2_panel_footprint`     | Staff-level interview loop                            | ≥4 hours across ≥3 panelists              |
| `R3_cto_on_panel`        | Staff-level offer path                                | Sam (CTO) on panel                        |
| `R4_noncompete_screen`   | candidate.prior_company in {Stripe, Plaid, Adyen}     | probe non-compete in screen invite        |
| `R5_staff_backend_floor` | Staff Backend offer drafting                          | base ≥$180k, negotiate upward             |
| `R6_one_counter`         | candidate has counter-offered once                    | do not counter twice; walk                |

R1–R4 get exercised by scheduling; R5–R6 stay in the rulebook as retrieval
noise — a nice check that trigger-matching keeps them out of the scheduling
tool calls.

## Layout

```
data/
  alex_playbook.md                  prose rulebook (LLM context at retrieval)
  seed_rules.yaml                   hand-authored Rule records
  candidates/dev_rao.json           main demo input (ex-Stripe, R4 triggers)
  candidates/maria_cruz.json        edge-case input (ex-Revolut, R4 skipped)
rule_models.py                      Rule + plain domain tool-output models
ingest_human_memory.py              ingest rulebook → human_memory dataset
agent_tools.py                      three @agent_memory-decorated tools (pure domain)
agent_loop.py                       minimal planner loop: LLM picks next tool
_run.py                             shared runner — loop + post-loop memify
run_naive.py                        with_memory=False → output/naive_plan.json
run_grounded.py                     with_memory=True  → output/grounded_plan.json
run_grounded_edge.py                edge case: Maria Cruz (ex-Revolut)
check_rule_compliance.py            5-row PASS/FAIL table across both plans
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
# 1. Ingest the rulebook into human_memory
python -m examples.demos.recruiting_distill_memory.ingest_human_memory

# 2. Naive run — agent has no memory, plan violates Ledgerline rules
python -m examples.demos.recruiting_distill_memory.run_naive

# 3. Grounded run — agent retrieves rulebook, plan satisfies rules,
#    session traces are memified into agent_proposed_rule nodes
python -m examples.demos.recruiting_distill_memory.run_grounded

# 4. The payoff table: 5 rules × naive/grounded PASS/FAIL
python -m examples.demos.recruiting_distill_memory.check_rule_compliance

# 5. Review agent-proposed nodes — approve any worth codifying
python -m examples.demos.recruiting_distill_memory.review_pending_rules

# 6. (Optional) Inspect the rulebook state
python -m examples.demos.recruiting_distill_memory.inspect_rulebook

# 7. (Optional) Visualize the human_memory graph
python -m examples.demos.recruiting_distill_memory.visualize
```

### Edge-case run: agent proposes, human approves

Maria Cruz is ex-Revolut — a fintech with similar non-compete exposure as
Stripe/Plaid/Adyen, but not in R4's enumerated list. Running her through
the grounded agent exercises the human-in-the-loop path:

```bash
python -m examples.demos.recruiting_distill_memory.run_grounded_edge
python -m examples.demos.recruiting_distill_memory.review_pending_rules
```

Expected behavior:
1. The grounded tools run without any proposal-flavored prompts — they
   just act. The decorator captures their traces regardless.
2. The post-loop memify pipeline extracts entities from the session
   feedback summaries (e.g. "screening invite for a Revolut candidate",
   "non-compete probing") and adds them as graph nodes tagged
   `agent_proposed_rule`.
3. `review_pending_rules.py` surfaces each of those nodes. On `[a]pprove`,
   an LLM structures the node into a proper `Rule` (trigger / action /
   rationale) and it's ingested into `human_memory` under
   `belongs_to_set=['rule','approved','agent_authored']`. Subsequent
   grounded runs retrieve it alongside Alex's seed rules.

## Expected payoff

After step 4 you should see something like:

```
Rule                             Naive      Grounded
------------------------------------------------------
Live coding, not take-home       FAIL       PASS
≥3 panelists                     PASS       PASS
≥4 hours total                   PASS       PASS
Sam (CTO) on panel               PASS       PASS
Non-compete probed (Stripe)      FAIL       PASS
------------------------------------------------------
TOTAL                            3/5        5/5
```

The base model's pretrained knowledge is enough for some generic best
practices (panel size, duration floor, sometimes-CTO) but not for
Ledgerline-specific policy: it picks the wrong format and misses the
Stripe non-compete probe. Memory closes the gap.

## Rules after one grounded run + auto-approve

After `run_grounded.py` followed by `review_pending_rules.py --auto-approve`,
the `human_memory` rulebook looks like this (6 seed + 10 agent-proposed).
The `source` field distinguishes them deterministically, and
`belongs_to_set` tags agent-authored ones with `agent_authored` on top of
`rule,approved`.

| rule_id                              | author | domain     | source                       |
|--------------------------------------|--------|------------|------------------------------|
| `R1_live_coding`                     | human  | scheduling | `alex_playbook`              |
| `R2_panel_footprint`                 | human  | scheduling | `alex_playbook`              |
| `R3_cto_on_panel`                    | human  | scheduling | `alex_playbook`              |
| `R4_noncompete_screen`               | human  | screening  | `alex_playbook`              |
| `R5_staff_backend_floor`             | human  | offer      | `alex_playbook`              |
| `R6_one_counter`                     | human  | offer      | `alex_playbook`              |
| `R7_screening_email_staff_backend`   | agent  | screening  | `agent_proposal:<node_uuid>` |
| `R8_live_coding_panel_cto`           | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R9_90min_live_coding`               | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R10_vp_engineering_panel`           | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R11_staff_backend_panel`            | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R12_staff_backend_rule`             | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R13_senior_product_manager_panel`   | agent  | scheduling | `agent_proposal:<node_uuid>` |
| `R14_senior_product_manager_salary`  | agent  | offer      | `agent_proposal:<node_uuid>` |
| `R15_invitation_email_staff_backend` | agent  | screening  | `agent_proposal:<node_uuid>` |
| `R16_staff_backend_screening`        | agent  | screening  | `agent_proposal:<node_uuid>` |

A real human reviewer would reject most of the agent-authored proposals:
- **R8 / R9 / R12 / R16** are duplicates of Alex's R1–R3 at varying specificity.
- **R10 / R11 / R13** latched onto panelists in the screenshot as if they
  were a candidate attribute — a classic trace-summary artefact.
- **R14** is an outright hallucination — it invented a "Senior Product
  Manager salary floor" with fabricated numbers from the fact that Ravi
  (a panelist) has that title.
- **R7 / R15** are reasonable generalizations of the screening-invite
  behaviour and plausibly worth keeping.

That's the point of the review gate: the memify pipeline does not
silently append to the rulebook. `--auto-approve` exists for CI and
demos; in production, every row in the second half of that table would
require an explicit `[a]` keypress.

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
