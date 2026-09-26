# brainbox: disposable sandboxes, durable brain

v2 of the sandbox-kit demo. In v1 (`../demo/handover.sh`) memory lived
*inside* each sandbox and moved between them as a snapshot. Here the memory
stays in one central cognee API server — the **brain** — and a sandbox is
compute on loan: it borrows a narrow, revocable identity, does one job, hands
its result back, and is destroyed.

```console
$ ./run.sh code-architecture \
    --repo ../../../../cognee/infrastructure/databases/graph \
    --read demo_deal_desk \
    --question "Who owns the deal desk and what approval is pending?" --purge
```

One run (≈55 s on a laptop) does, in order:

| step | host / sandbox | what it proves |
|---|---|---|
| mint | host → brain | a cognee agent identity, read on the `--read` datasets, **explicitly denied on every other family dataset** |
| sandbox | host | created from `../cognee-memory-remote` under `deny-all`; a sandbox-scoped rule admits `localhost:<port>` only |
| secret | host | the agent key becomes a sandbox-scoped proxy-managed secret; the VM sees `sbx-cs-…` |
| recall | sandbox → brain | the question is answered from the granted dataset only; the brain does the LLM work |
| build | sandbox | `cognee.remember(repo, content_type="code")`: enola code graph, no LLM, no embeddings, no network; `architecture` view + Mermaid |
| promote | sandbox → brain | `cognee.push`: COGX archive imported with zero LLM calls into an output dataset the identity owns |
| walls | sandbox | `echo $COGNEE_API_KEY` → placeholder · `curl example.com` → 403 · recall on a non-granted dataset → 403 · `sbx policy log` |
| revoke | host | sandbox, secret and rule removed; grants revoked. Default keeps the identity + output dataset for review; `--purge` deletes both |

Outputs land in `work/<sandbox>/out/` (`architecture.html`, `architecture.mmd`,
`recall.json`, `report.json`); `work/` is git-ignored.

## Prerequisites

- `brew install docker/tap/sbx`, `sbx login`, `sbx policy init deny-all`
- a running cognee API. Default `BRAIN_URL=http://127.0.0.1:8011`; the owner's
  API key at `~/.cognee-plugin/api_key.json` (or `BRAIN_OWNER_KEY`). The brain
  can bind loopback only — the sandbox reaches it as `host.docker.internal`,
  which the sandbox proxy evaluates as `localhost:<port>`.
- the `--read` datasets must be cognified (a dataset with data but no graph
  answers `404 NoDataError`).

## What we learned building it (the interesting slides)

- **Host loopback is reachable and policed.** `host.docker.internal:<port>` is
  rewritten to `localhost:<port>` by the proxy and matched against the rule
  set; both the explicit-proxy and transparent paths are blocked without the
  rule. Rules can be sandbox-scoped and port-specific.
- **Placeholder substitution needs the proxy path.** Clients must honour
  `HTTP(S)_PROXY`; `aiohttp` does not by default (`trust_env=True`). Fixed in
  cognee's `CloudClient` by [#5196](https://github.com/topoteretes/cognee/pull/5196);
  `tasks/_compat.py` detects an installed release without it and shims aiohttp.
- **Two permission layers, and a default-allow to close.** The brain checks
  the ACL grant *and* the owner's sharing policy. With `share_sessions: true`
  every session-bearing dataset is shared with any agent that has no explicit
  rule — the first run leaked `slack` and `agent_sessions` into a
  deal-desk-only identity. `brain_admin.py mint` therefore writes an explicit
  allow/deny row for every family dataset.
- **Names never cross users.** A shared dataset is addressed by UUID; the
  COGX import on the brain is name-based, so the sandbox writes into a
  dataset it owns and the human promotes from there.

## Files

```
brainbox/
├── run.sh                    # host orchestrator: mint → sandbox → secret → task → walls → revoke
├── brain_admin.py            # brain REST admin with the owner key: mint / revoke / purge / datasets
├── tasks/code_architecture.py# in-sandbox payload: recall → local code graph → push
├── tasks/_compat.py          # aiohttp trust_env shim for cognee releases without #5196
└── work/                     # per-run workspaces (git-ignored)
```
