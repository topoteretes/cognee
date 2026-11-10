# Daytona RL rollouts with cognee memory, annotated

An RL rollout loop in the shape of the HUD cookbook: a bug-fix task with variants, one fresh Daytona sandbox per attempt, pytest as the grader, a binary reward, groups of attempts per variant.

What the cookbook leaves ephemeral, this keeps: one memory sandbox runs cognee for the whole run. The agent recalls lessons before each group and writes every scored attempt back; improve() turns a step's attempts into lessons for the next. Held-out variants are then tried with memory off and on.

The number in front of each line is seconds since the run started.

## snapshot 'cognee-memory' exists, reusing

```text
[    2.2s] === snapshot 'cognee-memory' exists, reusing ===
```

The snapshot (Debian, Python 3.12, pip install cognee) exists and is active, so no sandbox has to install anything. Under a second.

## snapshot 'cognee-rl-env' exists, reusing

```text
[    2.3s] === snapshot 'cognee-rl-env' exists, reusing ===
```

The rollout image (Debian, Python 3.12, pytest; 1 vCPU, 1 GB) exists and is active. Under a second.

## run 143a00

```text
[    2.3s] === run 143a00 ===
```

143a00 is this run's id. It is in every sandbox name and in a run= label, which the cleanup lists by, so nothing created here is left behind.

## memory sandbox: cognee-rl-memory-143a00 (cognee API server)

```text
[    2.3s] === memory sandbox: cognee-rl-memory-143a00 (cognee API server) ===
[    6.0s]     api: https://8000-9ff25cc8-df84-4225-90b1-77ea9e578104.daytonaproxy01.eu (private; preview token on every request)
[   19.0s]     healthy after 13s; dataset 'rl_memory', single-user mode
```

The one long-lived sandbox: the cognee API server from the cognee snapshot, with the LLM key as a Secret placeholder. Private sandbox; the host reaches it through the preview URL with its token. Up in 13s.

## step 1/3: variants A1,B1,C1,D1 (group 8, parallel 4)

```text
[   19.0s] === step 1/3: variants A1,B1,C1,D1 (group 8, parallel 4) ===
[   19.3s]     [A1 money_rounding] lessons recalled: none yet
[   77.5s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[   77.5s]     [A1 money_rounding] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  23s/attempt
[   85.1s]     [B1 idempotency] lessons recalled: none yet
[  123.3s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  123.3s]     [B1 idempotency] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  17s/attempt
[  130.9s]     [C1 retries] lessons recalled: none yet
[  191.2s]     sweep failed: Failed to remove sandbox: Sandbox with ID or name 2a62dc64-798b-41bc-8efb-13f63847cf7e not found
[  191.2s]     [C1 retries] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  28s/attempt
[  198.1s]     [D1 refunds] lessons recalled: none yet
[  242.5s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  242.5s]     [D1 refunds] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  21s/attempt
[  249.9s]     step 1: pass 0/32, 0/4 groups learnable
```

Training step 1 of 3: one variant per bug family, 8 attempts each, every attempt in its own fresh sandbox. Before a group, one recall fetches the family's lessons; 'none yet' means the memory is still empty.

Rewards are 1 when every hidden test passes, else 0. A group with mixed rewards is one a group-relative trainer can learn from; a uniform group gives it nothing. No mixed group here. Uniform: A1, B1, C1, D1.

Step total 0/32 passed, 0/4 groups learnable. Each attempt is now in memory as a scored entry: the task, the answer, and the failing assertions when it failed. 3.8 min.

## improve after step 1 (memory sandbox)

```text
[  249.9s] === improve after step 1 (memory sandbox) ===
[  268.1s]     feedback_weights: completed sessions=1
[  268.1s]     persist_session_qa: completed sessions=1
[  268.1s]     persist_agent_traces: already_completed (no_new_trace_steps)
[  268.1s]     extract_agent_context: skipped (auto_feedback_disabled)
[  268.1s]     distill_sessions: completed documents=0, sessions_completed=0, sessions_failed=0, entries_uncurated=0
[  268.1s]     update_user_preferences: skipped (personalization_disabled)
[  268.1s]     build_truth_subspace: skipped (opt_in_disabled)
[  268.1s]     review_conflicts: skipped (opt_in_disabled)
[  268.1s]     global_context_index: skipped (opt_in_disabled)
```

improve() over the step's session, inside the memory sandbox. Feedback weights apply the scores, and the scored attempts are persisted into the graph (completed). The lesson-drafting stages are off here (they hang on per-turn feedback analysis), so the lessons the next step sees are synthesized at recall time from these persisted attempts: their changes, verdicts and failing assertions. Skipped: extract_agent_context, update_user_preferences, build_truth_subspace, review_conflicts, global_context_index. 18s.

## step 2/3: variants A2,B2,C2,D2 (group 8, parallel 4)

```text
[  268.1s] === step 2/3: variants A2,B2,C2,D2 (group 8, parallel 4) ===
[  274.7s]     [A2 money_rounding] lessons recalled: 1085 chars: ### A1 — Discount arithmetic - Expected: `apply_discount(125, 10) == 113`. - Required change: use half-up rounding (`112.5 → 113`), not floor division…
[  325.2s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  325.2s]     [A2 money_rounding] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  24s/attempt
[  337.0s]     [B2 idempotency] lessons recalled: 501 chars: ### B1 — Repeated idempotency keys - Same idempotency key and same amount must return the existing charge ID. - Same key with a different amount must …
[  363.4s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  363.4s]     [B2 idempotency] rewards 1 1 1 0 0 1 0 1  pass 5/8  mixed (learnable)  12s/attempt
[  376.3s]     [C2 retries] lessons recalled: 575 chars: - `with_retry` must use **total-attempt semantics**: call `fn()` up to `attempts` times, sleeping only between failures. - Backoff must be exponential…
[  403.1s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  403.1s]     [C2 retries] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  13s/attempt
[  414.0s]     [D2 refunds] lessons recalled: 434 chars: - `test_refund_complete` expected a completed refund to make the next charge ID distinct: `assert second != first`; the observed failure was `assert '…
[  447.5s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  447.5s]     [D2 refunds] rewards 1 1 1 1 1 1 1 1  pass 8/8  uniform (no gradient)  14s/attempt
[  454.1s]     step 2: pass 13/32, 1/4 groups learnable
```

Training step 2 of 3: one variant per bug family, 8 attempts each, every attempt in its own fresh sandbox. Before a group, one recall fetches the family's lessons; 'none yet' means the memory is still empty.

Rewards are 1 when every hidden test passes, else 0. A group with mixed rewards is one a group-relative trainer can learn from; a uniform group gives it nothing. Mixed here: B2. Uniform: A2, C2, D2.

Step total 13/32 passed, 1/4 groups learnable. Each attempt is now in memory as a scored entry: the task, the answer, and the failing assertions when it failed. 3.1 min.

## improve after step 2 (memory sandbox)

```text
[  454.1s] === improve after step 2 (memory sandbox) ===
[  471.5s]     feedback_weights: completed sessions=1
[  471.5s]     persist_session_qa: completed sessions=1
[  471.5s]     persist_agent_traces: already_completed (no_new_trace_steps)
[  471.5s]     extract_agent_context: skipped (auto_feedback_disabled)
[  471.5s]     distill_sessions: completed documents=0, sessions_completed=0, sessions_failed=0, entries_uncurated=0
[  471.5s]     update_user_preferences: skipped (personalization_disabled)
[  471.5s]     build_truth_subspace: skipped (opt_in_disabled)
[  471.5s]     review_conflicts: skipped (opt_in_disabled)
[  471.5s]     global_context_index: skipped (opt_in_disabled)
```

improve() over the step's session, inside the memory sandbox. Feedback weights apply the scores, and the scored attempts are persisted into the graph (completed). The lesson-drafting stages are off here (they hang on per-turn feedback analysis), so the lessons the next step sees are synthesized at recall time from these persisted attempts: their changes, verdicts and failing assertions. Skipped: extract_agent_context, update_user_preferences, build_truth_subspace, review_conflicts, global_context_index. 17s.

## step 3/3: variants A1,B1,C1,D1 (group 8, parallel 4)

```text
[  471.5s] === step 3/3: variants A1,B1,C1,D1 (group 8, parallel 4) ===
[  475.1s]     [A1 money_rounding] lessons recalled: 530 chars: - **Expected assertion:** `apply_discount(125, 10) == 113`; attempts using floor division returned `112`. - **Required convention:** use half-up round…
[  511.2s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  511.2s]     [A1 money_rounding] rewards 1 1 1 1 1 0 1 1  pass 7/8  mixed (learnable)  17s/attempt
[  523.0s]     [B1 idempotency] lessons recalled: 537 chars: - Same idempotency key with the same amount must return the existing charge ID. - Reusing a key with a different amount must raise `LedgerError` with …
[  547.7s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  547.7s]     [B1 idempotency] rewards 1 1 1 1 1 1 1 1  pass 8/8  uniform (no gradient)  12s/attempt
[  559.6s]     [C1 retries] lessons recalled: 703 chars: ### C1/C2 — `with_retry` attempt count and backoff - Expected **total-attempt semantics**: call `fn()` up to `attempts` times, sleeping only between f…
[  604.4s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  604.4s]     [C1 retries] rewards 0 0 0 0 0 0 0 0  pass 0/8  uniform (no gradient)  17s/attempt
[  615.9s]     [D1 refunds] lessons recalled: 596 chars: ### D1 — Refund limits and accumulation - `test_refund_complete` expected the next charge ID to be distinct: `assert second != first`. - The failing v…
[  637.6s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  637.6s]     [D1 refunds] rewards 1 1 1 1 1 1 1 1  pass 8/8  uniform (no gradient)  10s/attempt
[  645.6s]     step 3: pass 23/32, 1/4 groups learnable
```

Training step 3 of 3: one variant per bug family, 8 attempts each, every attempt in its own fresh sandbox. Before a group, one recall fetches the family's lessons; 'none yet' means the memory is still empty.

Rewards are 1 when every hidden test passes, else 0. A group with mixed rewards is one a group-relative trainer can learn from; a uniform group gives it nothing. Mixed here: A1. Uniform: B1, C1, D1.

Step total 23/32 passed, 1/4 groups learnable. Each attempt is now in memory as a scored entry: the task, the answer, and the failing assertions when it failed. 2.9 min.

## improve after step 3 (memory sandbox)

```text
[  645.6s] === improve after step 3 (memory sandbox) ===
[  662.6s]     feedback_weights: completed sessions=1
[  662.6s]     persist_session_qa: completed sessions=1
[  662.6s]     persist_agent_traces: already_completed (no_new_trace_steps)
[  662.6s]     extract_agent_context: skipped (auto_feedback_disabled)
[  662.6s]     distill_sessions: completed documents=0, sessions_completed=0, sessions_failed=0, entries_uncurated=0
[  662.6s]     update_user_preferences: skipped (personalization_disabled)
[  662.6s]     build_truth_subspace: skipped (opt_in_disabled)
[  662.6s]     review_conflicts: skipped (opt_in_disabled)
[  662.6s]     global_context_index: skipped (opt_in_disabled)
```

improve() over the step's session, inside the memory sandbox. Feedback weights apply the scores, and the scored attempts are persisted into the graph (completed). The lesson-drafting stages are off here (they hang on per-turn feedback analysis), so the lessons the next step sees are synthesized at recall time from these persisted attempts: their changes, verdicts and failing assertions. Skipped: extract_agent_context, update_user_preferences, build_truth_subspace, review_conflicts, global_context_index. 17s.

## evaluation: held-out variants A3,B3,C3,D3 (4 attempts each, memory off then on)

```text
[  662.6s] === evaluation: held-out variants A3,B3,C3,D3 (4 attempts each, memory off then on) ===
[  662.6s]     [A3 money_rounding] lessons recalled: none yet
[  693.6s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  693.6s]     [A3 money_rounding] rewards 1 1 1 1  pass 4/4  uniform (no gradient)  26s/attempt
[  693.6s]     [B3 idempotency] lessons recalled: none yet
[  724.5s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  724.5s]     [B3 idempotency] rewards 0 0 0 0  pass 0/4  uniform (no gradient)  25s/attempt
[  724.5s]     [C3 retries] lessons recalled: none yet
[  738.4s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  738.4s]     [C3 retries] rewards 0 0 0 0  pass 0/4  uniform (no gradient)  13s/attempt
[  738.4s]     [D3 refunds] lessons recalled: none yet
[  766.6s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  766.6s]     [D3 refunds] rewards 0 0 0 0  pass 0/4  uniform (no gradient)  25s/attempt
[  766.6s]     memory off: pass 4/16
[  772.0s]     [A3 money_rounding] lessons recalled: 510 chars: ### Expected failures - `apply_discount(125, 10)` was expected to equal `113`; floor division returned `112`. - No discount-specific exception type or…
[  785.5s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  785.5s]     [A3 money_rounding] rewards 1 1 1 1  pass 4/4  uniform (no gradient)  12s/attempt
[  790.6s]     [B3 idempotency] lessons recalled: 626 chars: - Same idempotency key and same amount must return the existing charge ID. - Reusing a key with a different amount must raise `LedgerError` with exact…
[  803.3s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  803.3s]     [B3 idempotency] rewards 1 1 1 1  pass 4/4  uniform (no gradient)  11s/attempt
[  810.0s]     [C3 retries] lessons recalled: 651 chars: - `with_retry` must use **total-attempt semantics**: call `fn()` at most `attempts` times, sleeping only between failures. - Backoff must be exponenti…
[  821.7s]     sweep failed: Failed to remove sandbox: Sandbox with ID or name f85d374a-531d-485e-adc6-127f589051b6 not found
[  821.7s]     [C3 retries] rewards 0 0 0 0  pass 0/4  uniform (no gradient)  11s/attempt
[  827.5s]     [D3 refunds] lessons recalled: 861 chars: - **Failing expectation:** After a complete refund, the next charge ID must be distinct: `assert second != first`. The failure was `assert 'ch_1' != '…
[  842.2s]     sweep failed: Failed to remove sandbox: Sandbox state change in progress
[  842.2s]     [D3 refunds] rewards 0 0 0 0  pass 0/4  uniform (no gradient)  14s/attempt
[  842.2s]     memory on: pass 8/16
```

Held-out variants: same families, bugs never seen in training. First with no memory, then with the lessons recalled.

Memory off 4/16, memory on 8/16. That difference is what memory alone buys the policy, with no weights changed. 3.0 min.

## exporting memory (COGX)

```text
[  842.2s] === exporting memory (COGX) ===
[  844.8s]     api server and its database workers stopped; exporting from the sandbox's state
[  854.1s]     [cogx] exported rl_memory: 90 nodes, 392 edges, 29 kB
[  855.0s]     pulled to <repo>/examples/integrations/daytona/rl/out/rl_memory.cogx.tar.gz
```

The memory checkpoint: cognee.export(format='cogx') of the rl_memory dataset, 90 nodes and 392 edges in 29 kB, pulled to the host. Restore it into another memory sandbox to continue training from here. 13s.

## results

```text
[  855.0s] === results ===
[  855.0s]     step 1: pass 0/32, learnable groups 0/4
[  855.0s]     step 2: pass 13/32, learnable groups 1/4
[  855.0s]     step 3: pass 23/32, learnable groups 1/4
[  855.0s]     held-out memory off: 4/16  (A3 4/4 B3 0/4 C3 0/4 D3 0/4)
[  855.0s]     held-out memory on: 8/16  (A3 4/4 B3 4/4 C3 0/4 D3 0/4)
[  855.0s]     rollouts: 128, 17s per attempt
[  855.0s]     report: <repo>/examples/integrations/daytona/rl/out/report-143a00.json
```

Per step: pass rate and learnable groups. Held-out: memory off against memory on. The report JSON has every attempt. Cleanup then deletes the run's sandboxes by label.
