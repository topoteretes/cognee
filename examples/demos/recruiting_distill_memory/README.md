# Recruiting Demo — Distilled Human Memory → Grounded Agent

A startup (Ledgerline, a 40-person payments fintech) is automating parts of its
recruiting workflow. Rather than letting the AI agent build its own memory
from scratch, they want it to **consult their senior recruiter's knowledge**
when it acts. This demo shows that end-to-end on Cognee.

**The arc, in four beats:**
1. Alex (senior recruiter) contributes a playbook and a rulebook. An ingest
   pipeline persists the rulebook as typed `Rule` graph nodes in a
   `human_memory` dataset.
2. The agent replaces Alex for a narrow task — scheduling an interview loop
   for a new candidate. Its tools are wrapped with `@cognee.agent_memory`
   against `human_memory`.
3. We run the agent **twice** on the same candidate (Dev Rao, ex-Stripe):
   once **naive** (no memory), once **grounded** (retrieves the rulebook).
   The naive plan violates Ledgerline's policies; the grounded plan follows
   them.
4. A deterministic pass links the agent's trace steps to the rules they
   cited — so the action → rule path is inspectable as a graph.

## Why this pattern (Mechanism A: pre-distilled rules)

We extract Alex's knowledge into an explicit, typed rulebook *before* the
agent runs, rather than having the agent retrieve from raw chunks at
runtime, because:
- The rulebook is an **inspectable artifact** stakeholders can read.
- Rules have **stable IDs**, so trace → rule linking is deterministic
  (no LLM judge).
- Retrieval noise is lower: trigger-matching surfaces only relevant rules,
  not whatever semantically-adjacent chunks happen to rank highly.

## Scenario

- **Company:** Ledgerline — 40-person payments fintech
- **Role:** Staff Backend Engineer
- **Recruiter being replaced:** Alex Chen
- **Demo candidate:** Dev Rao — 8 YOE, prior at Stripe, target base $170k

The six rules in the rulebook (`data/seed_rules.yaml`):

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
rule_models.py                      Rule, ProposedRule, tool-output models
ingest_human_memory.py              ingest rulebook → human_memory dataset
agent_tools.py                      three @agent_memory-decorated tools
_run.py                             shared plan runner (reads candidate via env)
run_naive.py                        with_memory=False → output/naive_plan.json
run_grounded.py                     with_memory=True  → output/grounded_plan.json
run_grounded_edge.py                edge case: Maria Cruz (ex-Revolut)
check_rule_compliance.py            5-row PASS/FAIL table across both plans
link_traces_to_rules.py             trace → Rule edges in agent_memory graph
review_pending_rules.py             human approval CLI for agent-proposed rules
output/                             generated plan JSONs (gitignored)
```

## Prerequisites

1. Working Cognee install (see repo root CLAUDE.md / README).
2. `.env` at the repo root with `LLM_API_KEY` / `LLM_MODEL` set. Defaults
   (OpenAI + local Kuzu/LanceDB/SQLite) are fine.
3. `ENABLE_BACKEND_ACCESS_CONTROL` at default (`true`): per-user storage
   isolation matters here — the linker needs to read from the same scope
   the grounded run wrote into.

## Run the demo

All commands below are run from the **repo root**.

```bash
# 1. Ingest the rulebook into human_memory
python -m examples.demos.recruiting_distill_memory.ingest_human_memory

# 2. Naive run — agent has no memory, plan violates Ledgerline rules
python -m examples.demos.recruiting_distill_memory.run_naive

# 3. Grounded run — agent retrieves rulebook, plan satisfies rules,
#    traces persisted via SessionManager
python -m examples.demos.recruiting_distill_memory.run_grounded

# 4. The payoff table: 5 rules × naive/grounded PASS/FAIL
python -m examples.demos.recruiting_distill_memory.check_rule_compliance

# 5. Link each grounded trace step to the rules it cited, in the graph
python -m examples.demos.recruiting_distill_memory.link_traces_to_rules

# 6. (Optional) Review any agent-proposed rules. No-ops for Dev Rao.
python -m examples.demos.recruiting_distill_memory.review_pending_rules

# 7. (Optional) Visual verification — both graphs side by side with
#    applied_rule cross-edges between agent_memory and human_memory
cognee-cli -ui
```

### Edge-case run: trigger-matching correctness

To verify that R4's trigger (only Stripe/Plaid/Adyen) keeps it out of an
unrelated candidate's screen invite:

```bash
python -m examples.demos.recruiting_distill_memory.run_grounded_edge
python -m examples.demos.recruiting_distill_memory.link_traces_to_rules \
  --session-id recruiting-demo-grounded-maria
```

Maria Cruz is ex-Revolut. Expected behavior: `compose_screen_invite`
does **not** apply R4 (Revolut isn't in R4's enumerated list), even
though Revolut is a fintech and a human might argue it's relevant. This
is trigger-matching working correctly — and it's the counterpart to
Dev Rao's run, where R4 does fire. Any "extend R4 to cover Revolut"
decision is deferred to a human via `review_pending_rules.py` rather
than made implicitly by the LLM.

## Expected payoff

After step 4 you should see:

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
practices (having a panel, meeting a duration floor, including the CTO by
coincidence) but not for Ledgerline-specific policy: it picks the wrong
interview format and misses the Stripe-non-compete probe entirely. Memory
closes the gap.

After step 5:

```
propose_interview_format       bucket=grounded_in_rule   applied=['R1_live_coding']
schedule_panel                 bucket=grounded_in_rule   applied=['R1_live_coding', 'R2_panel_footprint', 'R3_cto_on_panel']
compose_screen_invite          bucket=grounded_in_rule   applied=['R4_noncompete_screen']
```

The linker will occasionally log a hallucinated rule ID — the summarizer
LLM that formats `memory_context` sometimes drifts (e.g. citing
`R5_salary_floor_backend` instead of the real `R5_staff_backend_floor`).
The linker skips these so only real citations become graph edges; the
hallucinations get printed for inspection.

## Troubleshooting

**`link_traces_to_rules.py` says "No traces found"**
  You're running it outside the per-user storage scope. The script now
  executes its work inside a `run_custom_pipeline` task so the ACL scope
  is active — make sure you ran `run_grounded.py` first in the same
  installation (same `SYSTEM_ROOT_DIRECTORY` / `DATA_ROOT_DIRECTORY`).

**Grounded run still fails a rule**
  Inspect `memory_context` in that step's trace. Usually the summarizer
  dropped the full rule_id prefix (`R1_live_coding` → just `R1`). The
  `_MEMORY_SYSTEM_PROMPT` in `agent_tools.py` is tightened to discourage
  this, but if you swap models it may regress. Raising `memory_top_k` or
  widening the system-prompt wording helps.

**Naive run passes everything**
  The base model occasionally picks "live coding" or the right panel size
  unprompted. Tighten the tool's system prompt in `agent_tools.py` to
  emphasize Ledgerline-specific policy over generic best practice so
  there's no free-lunch from pretraining.

**Proposed rules show up in grounded plan**
  The agent occasionally proposes a new rule (`proposed_new_rules` list
  in the JSON output). Run `review_pending_rules.py` to promote accepted
  ones to `Rule` nodes in `human_memory`.
