# Cognee memory handover on Daytona sandboxes

The [Daytona](https://www.daytona.io/) port of the
[Docker Sandboxes kit](../docker-sandbox-kit/): a supervisor agent and a worker
agent, each in its own cloud sandbox, handing persistent cognee memory to each
other under cognee's per-user ACLs. Everything runs embedded inside the
sandboxes; there is no cognee server.

The payload is the **same** `supervisor_worker_handover.py` the Docker demo
uses (it is read from `../docker-sandbox-kit/demo/`, not copied). What this
directory adds is the Daytona orchestration, `handover.py`.

## How the handover works here

Docker Sandboxes move the cognee state between two long-lived sandboxes with
`sbx cp`. On Daytona the state moves between three short-lived sandboxes, one
per phase:

```
snapshot "cognee-memory"           debian-slim + cognee, built once, reused
  └─ cognee-supervisor-<run>  --phase brief    private note + briefing, grant worker read/write, emit token
       └─ handover: cognee-worker-<run>  --phase work     recall by UUID, prove denials, write report
            └─ handover: cognee-review-<run>  --phase review   supervisor recalls the worker's report
```

The handover itself has three forms (`--handover`, default `cogx`):

- **cogx** — the shared dataset travels as a [COGX](https://docs.cognee.ai/examples/migrate-memory-systems)
  archive, cognee's exchange format. In the parent, `cogx_handover.py export`
  runs `cognee.export(dataset_id, format="cogx", include_permissions=True)` as
  the dataset's owner: a directory of plain files (`manifest.json`,
  `entities.jsonl`, `facts.jsonl`, `documents.jsonl`, `nodes.jsonl` with the
  chunks, and `permissions.json` with the owner and ACL grants, credentials
  included), packed to a few kB. The host pulls the tarball and the token and
  pushes them into a fresh sandbox from the snapshot, where
  `cogx_handover.py import` runs `cognee.remember(COGXArchiveSource(dir))`:
  the dataset is restored under its archived owner with the archived grants,
  with no LLM call (embeddings are recomputed). Only that dataset crosses; the
  supervisor's private dataset never leaves the first sandbox. Restored
  accounts get new ids, and a dataset id derives from its owner's, so the
  import re-issues the token with the restored dataset's id.
- **fork** — Daytona clones a running sandbox copy-on-write, so the handover
  *is* the fork: no copying, no shared mount. Fork exists for **Linux VM
  sandboxes only**, and a VM sandbox can only come from a VM-class snapshot
  (`CreateSnapshotParams(sandbox_class=SandboxClass.LINUX_VM, image="<registry image>")`
  in a region that has VM runners). The declarative builder this demo uses for
  its snapshot produces a container snapshot, which cannot be forked (Daytona
  answers `422 Forking is not supported for this sandbox`).
- **copy** — the `sbx cp` mechanism: the parent tars its cognee state and demo
  directory (payload + token), the host pulls the archive and pushes it into a
  fresh sandbox created from the snapshot with the same posture. Nothing else
  crosses. `auto` tries a fork and falls back to this on that 422, then uses it
  for the rest of the chain.

`fork` and `copy` move the *whole* embedded state (every dataset, the private
one included, protected only by the ACLs that travel with it); `cogx` moves
one dataset. Both `cogx` and `copy` run on a stock account today (verified:
Tier 1/2 organization, shared `eu`/`us` regions, no VM runners in either). The
fork path needs a VM-class snapshot and an organization with VM runners; it
has not been exercised end to end here.

`<run>` is a random suffix minted per invocation. Names never collide with
sandboxes a previous `--keep` left behind, and every sandbox of a run carries a
`run=<run>` label, which is what the cleanup lists by — including a sandbox the
script lost its handle to (a `create` that timed out after the sandbox existed,
a fork whose posture re-assertion failed). Cleanup deletes children first,
because a forked parent cannot be deleted while a fork of it is alive.

Either way the next sandbox ends up with cognee state on sandbox-local disk
(SQLite users/ACLs, one `.lbug` graph and one `.lance.db` vector store per
dataset) — the whole of it with `copy`/`fork`, exactly like the `sbx cp`
snapshot in the Docker demo; the shared dataset rebuilt from the archive with
`cogx`. And in every case what the worker can *read* is gated by cognee's
ACLs: the private dataset raises `PermissionDeniedError` (with `cogx` it is
not even on the machine — a dataset id the user has no grant on is denied the
same way), the shared dataset is reachable by UUID only (dataset names are
namespaced per user), and the report is written back with `dataset_id=`.

## Mapping the sbx kit onto Daytona

| Docker Sandboxes kit (`spec.yaml`) | Daytona (`handover.py`) |
|---|---|
| `credentials.proxyManaged` + `inject` on `api.openai.com` | organization **Secret** `cognee-openai` with `hosts=["api.openai.com"]`, mounted via `secrets={"LLM_API_KEY": "cognee-openai"}`. The sandbox sees an opaque placeholder; the proxy substitutes the key on requests to that host only. |
| `sbx policy init deny-all` + `permissions.network.allow` | `domain_allow_list="api.openai.com,extension.ladybugdb.com"` **on Tier 3/4 organizations**. On Tier 1/2 Daytona sets network policy for the whole organization and rejects a sandbox-level list (`400 … cannot be overridden at the sandbox level`); the script then runs behind the tier policy — everything blocked except Daytona's [essential services](https://www.daytona.io/docs/en/network-limits/#essential-services) (package registries, git hosts, model providers, `*.openai.com` among them) — and says so. PyPI is on neither list by choice: cognee is installed at snapshot build time. `raw.githubusercontent.com` (the Docker kit lists it for litellm's model-cost map) is not either: `LITELLM_LOCAL_MODEL_COST_MAP=True` makes litellm use the map bundled in the package. An allow-list domain is open for the sandbox's whole life, not just setup, so each one is a place the agent can read from. |
| `setup.install` (`uv tool install cognee` on every sandbox creation) | a **Snapshot** built once with `Image.debian_slim("3.12").pip_install("cognee")`; every sandbox in the chain starts from it warm. |
| `environment.variables` | passed per `exec` (`DATA_ROOT_DIRECTORY`, `SYSTEM_ROOT_DIRECTORY`, `LLM_MODEL`, `TELEMETRY_DISABLED`, `ENABLE_BACKEND_ACCESS_CONTROL=true`, `LITELLM_LOCAL_MODEL_COST_MAP=True`). |
| `agentInstructions` appended to the agent's memory file | no Daytona equivalent — Daytona has no agent-kit layer. Drop a `CLAUDE.md`/`AGENTS.md` into the work dir with `sandbox.fs.upload_file` when you run a real agent instead of the payload. |
| `sbx cp` state in and out of each sandbox | `sandbox.fork()` on Linux VM sandboxes; otherwise a tar archive pulled with `fs.download_file` and pushed with `fs.upload_file` (see "How the handover works here"). |
| `sbx policy log` | none locally; the script **checks** the walls from inside the sandbox instead and fails the run when one does not hold (see below). |

## The walls are checked, not shown

After the `brief` phase, `check_boundaries` runs inside the supervisor sandbox.
Every probe runs twice: on the **proxy path** (a client that honours the
`HTTP(S)_PROXY` variables Daytona sets, pointing at its egress proxy) and on the
**direct path** (the same request with proxy handling disabled — what a client
that ignores those variables does, as `aiohttp` did before cognee's remote
client set `trust_env=True` in
[#5196](https://github.com/topoteretes/cognee/pull/5196)).

Credential wall:

- `$LLM_API_KEY` must equal the Secret's placeholder (only the verdict and the
  length are printed, never the value);
- a request to `api.openai.com/v1/models` with that value must answer `200`
  via the proxy (the key is substituted in transit) and **not** `200` on the
  direct path (`401 Incorrect API key provided: dtn_secr…` — the value in the
  VM is not a key). A `200` on the direct path would mean the real key is in
  the sandbox.

Network wall: `https://example.com` — on neither allow list — must be
unreachable on both paths.

Any breach raises after all verdicts are printed; the sandboxes are still
cleaned up, and the run exits non-zero instead of ending in "round trip
passed". A sandbox that could not be deleted is reported the same way: a
running sandbox keeps a mounted Secret and an open allow list, so a failed
cleanup is a failed run.

**One known gap, reported rather than failed.** On a Tier 1/2 organization the
tier policy is enforced on **direct connections only**: `example.com` is reset
on the direct path, but Daytona's own egress proxy — the one every sandbox is
pre-configured to use — forwards to it (`200`). Since every ordinary HTTP
client honours `HTTP(S)_PROXY`, that is open egress for anything running in the
sandbox; it is the mirror image of the Docker finding in
[#5225](https://github.com/topoteretes/cognee/pull/5225), where only the
proxy-*ignoring* client got around the wall. The script prints it as a
`WARNING` and finishes, because it is Daytona's wall and not one the sandbox
owner can set on that tier; `--strict-walls` turns it into a failure. On Tier
3/4, where the sandbox-level allow list applies, a reachable `example.com` on
either path fails the run.

Verified live on a Tier 1/2 organization (Daytona API v0.221): placeholder
present, `200` via proxy / `401` direct against `api.openai.com`,
`example.com` `200` via proxy / connection reset direct.

Not used on purpose: Daytona **Volumes**. They are S3-backed FUSE mounts, the
same class of storage the Docker demo had to move away from (embedded LanceDB
would not run on the shared virtiofs workspace). Keep cognee's embedded
databases on sandbox-local disk; a volume is fine for the handover token or
other plain files.

## Run it

```console
$ pip install daytona
$ export DAYTONA_API_KEY=...          # https://app.daytona.io
$ export LLM_API_KEY=sk-...           # first run only: creates the host-scoped Secret
$ python handover.py                  # ~3 min snapshot build the first time, then the 3 phases
$ python handover.py --keep           # leave the sandboxes for inspection
$ python handover.py --rebuild        # rebuild the snapshot after a cognee release
$ python handover.py --handover copy  # move the raw state instead of a COGX archive
$ python handover.py --strict-walls   # fail on the Tier 1/2 proxy-path gap
```

A snapshot Daytona has deactivated (it does that to snapshots that sit unused
for a while; a `create` from one is refused with `400 Snapshot … is inactive`)
is activated and waited for before the first sandbox is created. Activating a
declaratively built snapshot is a rebuild on Daytona's side, so a run after a
quiet fortnight starts with a few minutes of `snapshot state: building` instead
of a traceback.

Output ends with the handover mode and network wall the run got, the dataset
UUID that crossed the user boundary, and the token in
`handover-out/handover_token.json`. Phase output is printed when each phase
finishes (`remember` builds a knowledge graph, so `brief` and `work` take a
minute or two each; a full run on a stock account is about five minutes).

Inspecting a kept sandbox (the run id is printed at the start of the run and
in the `Sandboxes kept:` line). A sandbox is stopped as soon as its phase is
done and its state handed over, so start it first:

```python
from daytona import Daytona

sb = Daytona().get("cognee-review-<run>")
sb.start()
print(sb.process.exec("printf '%s' \"$LLM_API_KEY\"").result)  # the placeholder, never a key
print(sb.process.exec("ls ~/cognee-state/system/databases/").result)  # cognee_db + per-owner dirs
```

Cleanup: the script deletes the run's sandboxes unless `--keep` is passed, in
which case it prints a one-liner that deletes them by the `run` label. The
snapshot and the Secret are kept for the next run.

## Notes

- The Secret is scoped to `api.openai.com`. Using another provider means a
  different `hosts=` list, a different `LLM_MODEL`, and its API host on the
  domain allow list — see the [provider docs](https://docs.cognee.ai/).
- `handover.py` re-applies `update_secrets`, `update_network_settings` and the
  auto-stop interval on every fork instead of assuming the child inherits
  them; if Daytona changes that behaviour the demo keeps its guarantees. The
  child is registered for cleanup before any of that runs, so a failed
  re-assertion cannot leave a running sandbox behind.
- A sandbox is stopped the moment its phase is done and its state has been
  handed over. That keeps the mounted Secret out of a running VM for the rest
  of the run and keeps the run inside the tier's running-memory quota (a Tier
  1/2 organization gets 10 GiB; three 4 GiB sandboxes do not fit).
- `extension.ladybugdb.com` is on the Tier 3/4 allow list as insurance only:
  the cognee wheel bundles the ladybug JSON extension
  (`cognee_db_workers/ladybug_extensions/`), and the run on Tier 1/2 — where
  that host is blocked on the direct path and answers 403 via the proxy — built
  and queried its graphs without it.
- Secret substitution happens on the proxy path. cognee's LLM and embedding
  calls go through litellm and httpx, which honour `HTTP(S)_PROXY`; if you add
  your own HTTP client to the payload, make sure it does too (`aiohttp` needs
  `trust_env=True`), or its requests leave with the placeholder and the
  provider answers 401.
- For concurrent agents that need live shared memory rather than a sequential
  handover, run a central cognee API server (see
  `distributed/deploy/daytona_sandbox.py`) and put its host on each sandbox's
  domain allow list.

## Walkthrough: the run, explained step by step

`walkthrough.py` shows a run of `handover.py` as terminal output, one block
at a time, and after every block says what just happened: which Daytona or
cognee mechanism produced those lines, what each line proves, how long the
step took. The explanations are derived from the lines themselves (status
codes, archive size, dataset UUID, elapsed time), so they describe the run on
screen, not an idealised one.

```console
$ python walkthrough.py                      # replay transcript.txt; needs no account
$ python walkthrough.py --pause              # wait for Enter between steps (presenting)
$ python walkthrough.py --live               # run handover.py now and annotate it as it streams
$ python walkthrough.py --live --record transcript.txt -- --keep   # record; flags pass through
$ python walkthrough.py --markdown > WALKTHROUGH.md                # the same as a document
```

`transcript.txt` is a real run on a stock (Tier 1/2) account, recorded with
`--live --record`; `WALKTHROUGH.md` is its Markdown rendering. A transcript is
the stdout+stderr of `handover.py` with an elapsed-time stamp in front of each
line, so `python handover.py 2>&1 | python walkthrough.py --stamp` annotates
a run started by hand as well.

## Memory in an RL rollout loop

[`rl/`](rl/) takes the same pieces (snapshot, Secret, posture, COGX) into the
shape of Daytona's [HUD RL cookbook](https://www.daytona.io/docs/en/guides/reinforcement-learning/hud-rl-cookbook/):
a bug-fix task with variants, one fresh 1 GB sandbox per attempt, pytest as
the grader, binary rewards, groups of attempts per variant. What it adds is
the part the cookbook leaves ephemeral: a memory sandbox running cognee for
the whole run, recalled before every group and scored after every attempt,
with `improve()` distilling each step into lessons for the next, and the
held-out pass rate measured with memory off and on. See [`rl/README.md`](rl/README.md).

## Layout

```
daytona/
├── handover.py        # snapshot → supervisor → handover → worker → handover → review
├── cogx_handover.py   # runs in the sandbox: export / import the shared dataset as COGX
├── walkthrough.py     # the run as annotated terminal output (replay, live, or Markdown)
├── transcript.txt     # a recorded run on a stock account, what walkthrough.py replays
├── WALKTHROUGH.md     # the same run rendered as a document
├── handover-out/      # the JSON handover token (created at runtime)
├── rl/                # memory in an RL rollout loop (HUD cookbook shape); own README
└── README.md
```
