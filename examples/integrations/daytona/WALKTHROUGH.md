# Daytona memory handover, annotated

Three Daytona sandboxes, one per phase, all from one snapshot with cognee installed. A supervisor agent writes memory, a worker agent in a second sandbox reads it and reports back, the supervisor reads the report in a third.

Memory is cognee's embedded state on the sandbox disk. The two agents are two cognee users; cognee's ACLs decide what each can read. Between sandboxes travel one dataset as a COGX archive and a JSON token with its id.

The number in front of each line is seconds since handover.py started.

## snapshot 'cognee-memory' exists, reusing

```text
[    1.0s] === snapshot 'cognee-memory' exists, reusing ===
```

The snapshot (Debian, Python 3.12, pip install cognee) exists and is active, so no sandbox has to install anything. Under a second.

## run 58de37

```text
[    1.0s] === run 58de37 ===
```

58de37 is this run's id. It is in every sandbox name and in a run= label, which the cleanup lists by, so nothing created here is left behind.

## creating sandbox: cognee-supervisor-58de37 (from snapshot)

```text
[    1.0s] === creating sandbox: cognee-supervisor-58de37 (from snapshot) ===
[    1.2s]     this organization's tier sets network policy for every sandbox; the
[    1.2s]     sandbox-level allow list (api.openai.com,extension.ladybugdb.com) was rejected. Running behind
[    1.2s]     the organization tier policy (essential services only) instead.
[    4.9s]
```

The first sandbox, asked for with the Secret bound to LLM_API_KEY and an allow list of two hosts.

Daytona rejected the allow list: this organization is Tier 1/2, where network policy is set org-wide. The sandbox runs behind that policy instead (essential services only, enforced on direct connections).

Then the payload and the COGX helper were uploaded. No memory yet. 4s.

## sandbox: cognee-supervisor-58de37 (phase: brief)

```text
[    4.9s] === sandbox: cognee-supervisor-58de37 (phase: brief) ===
[   52.9s]     sync_operations table already exists, skipping creation
[   52.9s]     principal_configuration table already exists, skipping creation
[   52.9s]     session_records table already exists, skipping creation
[   52.9s]     session_model_usage table already exists, skipping creation
[   52.9s]     [supervisor] remembering private note (dataset 'supervisor_private')...
[   52.9s]     [supervisor] remembering handover briefing (dataset 'handover')...
[   52.9s]     [supervisor] granting worker READ + WRITE on the handover dataset...
[   52.9s]     [supervisor] handover token issued -> handover-out/handover_token.json: {"dataset_id": "993feb18-2e1f-5c9e-93f6-e923947f53f7"}
[   52.9s]
```

The supervisor remembers a private note and a handover briefing, in two datasets of its own. Each remember() runs cognify: LLM entity extraction through the proxy, where the key is substituted. That is where the time goes.

Then it grants the worker read and write on the briefing dataset and writes its id (993feb18…) into the token. Ids are the only cross-user address; names are per user. 48s.

## security check inside cognee-supervisor-58de37 (network wall: organization tier policy (essential services only))

```text
[   52.9s] === security check inside cognee-supervisor-58de37 (network wall: organization tier policy (essential services only)) ===
[   53.0s]     credential wall: LLM_API_KEY inside the sandbox is the placeholder
[   54.8s]     credential wall: api.openai.com via proxy path: 200 (key substituted in transit)
[   55.9s]     credential wall: api.openai.com via direct path (proxy env ignored): 401 (placeholder is not a key)
[   56.2s]     network wall: WARNING https://example.com via proxy path: REACHABLE, organization tier policy (essential services only) not enforced
[   56.2s]         Daytona Tier 1/2 enforces the organization policy on direct connections
[   56.2s]         only; its egress proxy is not policed. Pass --strict-walls to fail on this.
[   56.9s]     network wall: https://example.com via direct path (proxy env ignored): blocked
```

Inside the sandbox LLM_API_KEY is the placeholder, not a key.

The same placeholder gets 200 from OpenAI through the proxy and 401 when the proxy is bypassed: the key exists only in transit.

example.com is blocked on a direct connection but reachable through the proxy. On Tier 1/2 Daytona polices direct connections only. Reported, not fatal; --strict-walls would fail here. 4s.

## exporting cognee-supervisor-58de37 -> cognee-worker-58de37 (the memory handover, COGX)

```text
[   56.9s] === exporting cognee-supervisor-58de37 -> cognee-worker-58de37 (the memory handover, COGX) ===
[   67.0s]     [cogx] exported dataset 'handover' (993feb18-2e1f-5c9e-93f6-e923947f53f7) as supervisor@handover.demo: 21 nodes, 25 edges
[   67.0s]     [cogx] archive files: documents.jsonl, entities.jsonl, facts.jsonl, handover.json, manifest.json, nodes.jsonl, permissions.json
[   67.0s]     [cogx] packed -> /tmp/cognee-handover.cogx.tar.gz (4 kB)
[   69.4s]     cognee-supervisor-58de37 stopped (its phase is done; state handed over)
```

The handover. cognee.export(format='cogx', include_permissions=True) on the shared dataset, as its owner: 21 nodes, 25 edges, 4 kB packed.

The archive is plain files: documents.jsonl, entities.jsonl, facts.jsonl, handover.json, manifest.json, nodes.jsonl, permissions.json. permissions.json carries the owner and the worker's grants. The private dataset is not in it and stays behind. Parent stopped once the archive was pulled. 13s.

## creating sandbox: cognee-worker-58de37 (from snapshot)

```text
[   69.4s] === creating sandbox: cognee-worker-58de37 (from snapshot) ===
```

A fresh sandbox for the worker phase: same snapshot, same posture (Secret binding, labels, auto-stop). It starts empty; the next block fills it. 3s.

## importing into cognee-worker-58de37 (COGX restore)

```text
[   72.3s] === importing into cognee-worker-58de37 (COGX restore) ===
[  100.3s]     sync_operations table already exists, skipping creation
[  100.3s]     principal_configuration table already exists, skipping creation
[  100.3s]     session_records table already exists, skipping creation
[  100.3s]     session_model_usage table already exists, skipping creation
[  100.3s]     [cogx] restored dataset 'handover' under owner supervisor@handover.demo: 21 nodes, 25 edges, status completed
[  100.3s]     [cogx] grant restored: supervisor@handover.demo -> read, write, delete, share
[  100.3s]     [cogx] grant restored: worker@handover.demo -> read, write
[  100.3s]     [cogx] token re-issued: dataset 993feb18-2e1f-5c9e-93f6-e923947f53f7 is 398a2f4d-27cf-5156-a26a-51c5595a3de7 here (restored accounts have new ids, and a dataset id derives from its owner's)
[  100.5s]
```

cognee.remember(COGXArchiveSource) restored 'handover' with 21 nodes and 25 edges, no LLM call; the time is the embeddings being recomputed. The owner and the granted users were recreated from permissions.json, grants included.

Restored accounts get new ids and a dataset id derives from its owner's, so the dataset is 398a2f4d… here instead of 993feb18…. The token was rewritten to match. 28s.

## sandbox: cognee-worker-58de37 (phase: work)

```text
[  100.5s] === sandbox: cognee-worker-58de37 (phase: work) ===
[  137.1s]     [worker] recalling the briefing from the shared dataset (by UUID)...
[  137.1s]
[  137.1s]     2026-10-20T17:59:42.775762 [error    ] PermissionDeniedError raised (Status code: 403): Request owner does not have necessary permission: [read] for all datasets requested. [cognee.shared.logging_utils]
[  137.1s]
[  137.1s]     2026-10-20T17:59:42.788597 [error    ] DatasetNotFoundError raised (Status code: 404): Dataset(s) not found: 'handover'. Dataset names resolve only among the datasets you own in the current tenant; pass dataset_ids for datasets shared with you. [cognee.shared.logging_utils]
[  137.1s]
[  137.1s]     Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
[  137.1s]     LiteLLM.Info: If you need to debug this error, use `litellm.turn_on_debug()'.
[  137.1s]
[  137.1s]     [worker] briefing recalled: Your task is to migrate the payments service to the v2 billing API and confirm the retry policy with the SRE team.
[  137.1s]
[  137.1s]     Deployment:
[  137.1s]     - Merge changes to the `dev` branch for automatic staging deployment.
[  137.1s]     - After contract tests pass, run `make deploy-prod` for production.
[  137.1s]     [worker] trying the supervisor's PRIVATE dataset (should be denied)...
[  137.1s]     [worker] correctly denied: PermissionDeniedError: PermissionDeniedError: Request owner does not have necessary permission: [read] for all datasets requested. (Status code: 403)
[  137.1s]     [worker] trying the shared dataset by NAME (names don't cross users)...
[  137.1s]     [worker] correctly not found: DatasetNotFoundError: DatasetNotFoundError: Dataset(s) not found: 'handover'. Dataset names resolve only among the datasets you own in the current tenant; pass dataset_ids for datasets shared with you. (Status code: 404)
[  137.1s]     [worker] writing the completion report back into the shared dataset...
[  137.1s]     [worker] report stored.
```

The two error lines at the top are cognee logging the denials provoked below; stderr arrives before the buffered prints.

The litellm banner is noise from a non-fatal internal error.

The worker recalls the briefing by dataset id and gets an answer from the graph that arrived. Its two other attempts fail as they should: the private dataset id is denied (it does not even exist here), and the name 'handover' resolves to nothing, because names are per user.

Then it writes its report into the shared dataset by id; another cognify run. 37s.

## exporting cognee-worker-58de37 -> cognee-review-58de37 (the memory handover, COGX)

```text
[  137.1s] === exporting cognee-worker-58de37 -> cognee-review-58de37 (the memory handover, COGX) ===
[  147.1s]     [cogx] exported dataset 'handover' (398a2f4d-27cf-5156-a26a-51c5595a3de7) as supervisor@handover.demo: 29 nodes, 66 edges
[  147.1s]     [cogx] archive files: documents.jsonl, entities.jsonl, facts.jsonl, handover.json, manifest.json, nodes.jsonl, permissions.json
[  147.1s]     [cogx] packed -> /tmp/cognee-handover.cogx.tar.gz (6 kB)
[  149.7s]     cognee-worker-58de37 stopped (its phase is done; state handed over)
```

The handover. cognee.export(format='cogx', include_permissions=True) on the shared dataset, as its owner: 29 nodes, 66 edges, 6 kB packed.

The archive is plain files: documents.jsonl, entities.jsonl, facts.jsonl, handover.json, manifest.json, nodes.jsonl, permissions.json. permissions.json carries the owner and the worker's grants. The private dataset is not in it and stays behind. Parent stopped once the archive was pulled. 13s.

## creating sandbox: cognee-review-58de37 (from snapshot)

```text
[  149.7s] === creating sandbox: cognee-review-58de37 (from snapshot) ===
```

A fresh sandbox for the review phase: same snapshot, same posture (Secret binding, labels, auto-stop). It starts empty; the next block fills it. 3s.

## importing into cognee-review-58de37 (COGX restore)

```text
[  152.5s] === importing into cognee-review-58de37 (COGX restore) ===
[  189.5s]     sync_operations table already exists, skipping creation
[  189.5s]     principal_configuration table already exists, skipping creation
[  189.5s]     session_records table already exists, skipping creation
[  189.5s]     session_model_usage table already exists, skipping creation
[  189.5s]     [cogx] restored dataset 'handover' under owner supervisor@handover.demo: 29 nodes, 66 edges, status completed
[  189.5s]     [cogx] grant restored: default_user@example.com -> read
[  189.5s]     [cogx] grant restored: supervisor@handover.demo -> read, write, delete, share
[  189.5s]     [cogx] grant restored: worker@handover.demo -> read, write
[  189.5s]     [cogx] token re-issued: dataset 398a2f4d-27cf-5156-a26a-51c5595a3de7 is a45afa1d-23eb-5b59-aa40-af9894261c71 here (restored accounts have new ids, and a dataset id derives from its owner's)
[  189.6s]
```

cognee.remember(COGXArchiveSource) restored 'handover' with 29 nodes and 66 edges, no LLM call; the time is the embeddings being recomputed. The owner and the granted users were recreated from permissions.json, grants included.

The default_user grant is the previous sandbox's importer keeping read access, as cognee's import always does; it travelled along in permissions.json.

Restored accounts get new ids and a dataset id derives from its owner's, so the dataset is a45afa1d… here instead of 398a2f4d…. The token was rewritten to match. 37s.

## sandbox: cognee-review-58de37 (phase: review)

```text
[  189.6s] === sandbox: cognee-review-58de37 (phase: review) ===
[  210.0s]     [supervisor] recalling the worker's report...
[  210.0s]
[  210.0s]     Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
[  210.0s]     LiteLLM.Info: If you need to debug this error, use `litellm.turn_on_debug()'.
[  210.0s]
[  210.0s]     [supervisor] report recalled: The worker reported that the payments service was migrated to the v2 billing API, and the retry policy was confirmed as exponential backoff with up to 5 attempts. Yes, it was deployed to production with `make deploy-prod` after the contract tests passed.
[  210.7s]
```

Back as the supervisor, in the third sandbox: the report the worker wrote in the second one comes back from a recall. Round trip closed; the private note never moved. 21s.

## Handover round trip passed across three Daytona sandboxes.

```text
[  210.7s] Handover round trip passed across three Daytona sandboxes.
[  210.7s] Handover:     cogx
[  210.7s] Network wall: organization tier policy (essential services only)
[  210.7s]   WARNING https://example.com via proxy path: REACHABLE, organization tier policy (essential services only) not enforced
[  210.7s] Token exchanged via: <repo>/examples/integrations/daytona/handover-out/handover_token.json
[  210.7s] Dataset shared by UUID: a45afa1d-23eb-5b59-aa40-af9894261c71
```

Handover mode cogx, network wall and warnings as above, and the id of the one dataset that crossed (a45afa1d…, as re-issued in the last sandbox).

Cleanup deleted the run's sandboxes, children first. Snapshot and Secret stay.
