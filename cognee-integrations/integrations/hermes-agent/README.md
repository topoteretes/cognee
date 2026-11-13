<div align="center">
  <a href="https://www.cognee.ai">
    <img src="https://raw.githubusercontent.com/topoteretes/cognee/main/assets/cognee-logo.svg" alt="Cognee" width="260">
  </a>
  <p><strong>Cognee memory for Hermes Agent</strong> — persistent, graph-backed memory with session recall, durable remember/forget, and automatic capture for your Hermes agents.</p>
  <p>
    <a href="https://docs.cognee.ai">Docs</a> ·
    <a href="https://discord.gg/NQPKmU5CCg">Discord</a> ·
    <a href="https://github.com/topoteretes/cognee">Cognee core</a>
  </p>
  <p>
    <a href="https://pypi.org/project/cognee-integration-hermes-agent/"><img src="https://img.shields.io/pypi/v/cognee-integration-hermes-agent" alt="PyPI version"></a>
    <a href="https://pypi.org/project/cognee-integration-hermes-agent/"><img src="https://img.shields.io/pypi/dm/cognee-integration-hermes-agent" alt="PyPI downloads"></a>
  </p>
</div>

# Cognee Memory Plugin for Hermes Agent

Standalone Hermes memory provider backed by Cognee.

This replaces the closed in-tree Hermes PR path. Hermes no longer accepts new
providers under `plugins/memory/`; this integration is shaped as a standalone
plugin that can be installed into `~/.hermes/plugins/cognee` or distributed as a
Python package with the `hermes_agent.plugins` entry point.

## Features

- Stores each completed Hermes turn in Cognee session memory, and the turn's
  tool calls as traces, under a capture policy that never stores calls touching
  secret files and redacts secrets from what it stores (see
  [Capture policy and redaction](#capture-policy-and-redaction)).
- Queues turns the server never received (it was booting, or down) and replays
  them once it answers again, so a cold start loses nothing.
- Recalls memory per prompt: the retrieved graph context, trimmed to what
  Hermes does not already have, plus the session's distilled guidance (see
  [Per-prompt memory block](#per-prompt-memory-block)), with a plain-words hit
  counter on top.
- Uses `cognee_recall` for explicit search, `cognee_remember` for durable
  graph memory.
- Exposes `cognee_forget` for user-directed, per-document deletion ("forget
  what we said about tennis"): find candidates with previews, then delete only
  what the user confirms.
- Recall is auto on the default dataset: the graph recall names no dataset, and
  the server picks the best-matching one. Routing needs a cognee 1.6.6 server; an
  older one searches every dataset you can read. Saves still go to the
  conversation's dataset. A configured dataset other than the default
  (`COGNEE_PLUGIN_DATASET`, or the setup wizard's dataset) keeps recall pinned to
  it while no recall mode is set (reason `configured_pin`), as before; the first
  reply of such a session says once (per machine and dataset) that auto exists.
  A set recall mode decides either way: `COGNEE_PLUGIN_RECALL_MODE=auto|pinned`,
  or `recall_mode` in `cognee.json` (the setup wizard asks, offering `pinned` for
  a non-default dataset). `COGNEE_PLUGIN_DATASET=auto` means the default dataset
  and says nothing about the recall mode.
- Exposes `cognee_switch_dataset` to move a conversation to another dataset
  mid-session, bridging the session it leaves behind, or to set its recall mode.
  The conversation's own choice wins over the configured mode: `action='pin'`
  pins recall to the active dataset (no new session; a `switch` to the dataset in
  use does the same), a `switch` pins recall to its target, and `action='auto'`
  lets the server pick, a configured pin lifted too, for that conversation.
  `reset` goes back to the configured dataset and recall mode. `current` reports
  the mode in effect and, when pinned, why (`dataset_picked` for the
  conversation's own pin, `configured_pin` for the configured mode or a
  configured non-default dataset with no mode set). Every result
  carries a `message` line to relay to the user. A pin without a switch is stored
  as the mode only, so it stays on the active dataset: after a change of the
  configured dataset, recall is pinned to the new one.
- Indexes repositories into a deterministic code graph (`hermes cognee
  index-repo`) and answers structural code questions exactly via
  `cognee_code_search`, plus an identifier-gated code recall lane.
- Recaps recorded sessions with `cognee_recap` and `hermes cognee
  standup|digest|timeline`: what was worked on and left open, the lessons
  distilled into memory, or a dated timeline of one topic.
- Runs `cognee.improve()` at Hermes session end to bridge session memory into
  the graph, and before Hermes compresses a long conversation, so turns
  compressed away still reach the graph.
- `cognee_remember` returns as soon as the server has the text; the graph is
  built in the background, so a slow LLM never turns a stored fact into a
  reported timeout.
- Mirrors explicit Hermes memory writes through `on_memory_write`, and steers
  the agent to Cognee first: Hermes' built-in memory notes (`MEMORY.md`,
  `USER.md`) are the fallback for when Cognee has nothing on a question, and
  when the two disagree the agent goes with Cognee and says so. The rule is in
  the system prompt and again on each prompt's Cognee memory block
  (`COGNEE_MEMORY_STEER=false` turns both off). While Cognee is unavailable the
  plugin adds nothing, so the agent simply uses Hermes' memory.
- Works with Cognee Cloud or another cognee server out of the box, or with a
  local cognee server run from a private cognee runtime the setup installs.
- Closes every session out of process, the way the other cognee plugins do: a
  detached worker bridges the session into the graph and only then unregisters
  from the server. Exiting Hermes never waits on a graph build, and the promotion
  is never cut short by the server retiring. The same worker covers an uncleanly
  died Hermes, so no session is lost and no server lingers either way.

## Quick start

### Prerequisites

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) installed — see
  the [Hermes installation guide](https://hermes-agent.nousresearch.com/docs/getting-started/installation).
  (The install one-liner is quoted there rather than here: the catalog's install
  scanner flags a piped shell script even inside a README, and a clean scan is
  one less thing for a catalog reviewer to read past.)
- **Cognee Cloud (remote mode, the default):** a tenant URL and API key from your
  [Cognee Cloud dashboard](https://platform.cognee.ai/). Nothing else: no LLM key
  and no cognee package — the tenant runs the models, and the plugin talks to it
  over plain HTTP.
- **Local mode:** about 1 GB of disk for the cognee runtime, which the setup
  installs for you (step 2b), and optionally an LLM API key (e.g. OpenAI), which
  cognee uses to build the knowledge graph. Without a key it uses local models
  instead (about 1.5 GB more, see 2b). It lives in a private environment,
  `~/.cognee-plugin/hermes/runtime/venv`, because Hermes installs every plugin into
  one shared environment, where cognee's dependencies conflict with Hermes's pinned
  ones and would refuse the install for Cloud users too.
- **Python 3.10 or newer** for a pip install (`requires-python = ">=3.10"`,
  cognee's own floor, since local mode builds cognee's environment from the
  interpreter running Hermes). Note that macOS's Xcode Command Line Tools ship
  Python 3.9.6 — use a Homebrew, python.org or
  [uv](https://docs.astral.sh/uv/)-managed 3.10+ interpreter instead.

### 1. Install the plugin

**Hermes catalog (recommended):**

```bash
hermes plugins install cognee
```

Catalog installs use the commit reviewed by Hermes. Update them with
`hermes plugins update cognee`; a newer PyPI release does not change the
reviewed catalog version. The pip installer refuses to overwrite a catalog copy.

**Via pip:**

```bash
pip install cognee-integration-hermes-agent
cognee-hermes-install
```

The pip package registers the memory provider through Hermes' entry-point
discovery. The second command copies it into `~/.hermes/plugins/cognee/` to
also provide the CLI and dashboard integration. For this installation method,
update with `pip install -U cognee-integration-hermes-agent` followed by
`cognee-hermes-install` (`hermes cognee status` reminds you when the two drift).

Or, for development, copy a checkout into a Hermes home with no existing
Cognee installation (do not copy over a catalog-managed plugin):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/topoteretes/cognee.git
git -C cognee sparse-checkout set cognee-integrations/integrations/hermes-agent
mkdir -p ~/.hermes/plugins/cognee
cp -R cognee/cognee-integrations/integrations/hermes-agent/. ~/.hermes/plugins/cognee/
```

### 2. Run the setup

In a terminal:

```bash
hermes memory setup cognee
```

Installing alone does not turn memory on: Hermes uses a memory plugin only once
it is the selected `memory.provider`. This command selects Cognee and asks for a
mode, **remote** (the default, 2a) or **local** (2b). `hermes plugins enable
cognee` and the toggle in the Desktop app enable the plugin but do **not**
select it, so after either one `hermes memory status` still shows
`Provider: (none)`. The wizard writes non-secrets to `~/.hermes/cognee.json` and
secrets to `~/.hermes/.env`.

#### 2a. Cognee Cloud (default)

Accept **Mode: remote** and enter your tenant URL
(`https://tenant-xxx.aws.cognee.ai`) and API key from the
[Cognee Cloud dashboard](https://platform.cognee.ai/). Nothing runs locally in
this mode — no server is spawned and no LLM key is used; every request goes to
the tenant, authenticated with your API key via `X-Api-Key`.

#### 2b. Local

Type **local** at the mode prompt and paste your LLM API key when asked, or leave
it blank to run without one (below). If the
cognee runtime is not installed yet, the wizard offers to install it:
`cognee==1.6.6` and its dependencies, about 1 GB from PyPI, into
`~/.cognee-plugin/hermes/runtime/venv`. With uv on your PATH that takes under a
minute; without it, pip takes a few. Choosing local also clears a Cognee Cloud
URL and key saved by an earlier remote setup.

To install, update or remove the runtime on its own:

```bash
hermes cognee install-runtime   # install, or move it to the plugin's pinned cognee
hermes cognee install-runtime --yes   # the same, stopping this plugin's server without asking
hermes cognee remove-runtime    # delete it (your data in ~/.cognee is kept)
```

After a plugin update that moves the pinned cognee, local mode stays off until
you run `hermes cognee install-runtime`: the plugin never starts a server on a
cognee it was not written against, and never downloads one behind your back.
Hermes reports Cognee as unavailable with that command, and
`hermes cognee status` shows the runtime as outdated. `install-runtime` upgrades
cognee in place in `runtime/venv`. It drops the ready marker first and writes it
back only once the new version checks out, so a failed or interrupted install
leaves local mode off rather than half-upgraded, and the next `install-runtime`
repairs the venv. Once it has finished, start a new Hermes session.

**`install-runtime` stops the local server this plugin started before it
installs**, so the venv is never upgraded under a running process (a live
server would load modules of the new version into the old one). It finds that
server by the pidfile the plugin writes when it starts one (checked to still run
`cognee.api.client:app`, so a reused pid is left alone) and says so first:
"This stops the local cognee server (pid N) on port 8011; other cognee plugins
using it will restart it on their next request." In a terminal it asks
`Continue? (y/n)` (default yes); without one it goes ahead only with `--yes`. The
setup wizard asks the same in its install question. The server gets SIGTERM only,
never SIGKILL, and up to 60 seconds to finish the work in flight; one still
running after that is left running and the install stops without changing
anything: try again in a minute, or stop it yourself. The server is shared:
Claude Code, Codex or OpenClaw sessions using it lose it for a moment, and their
plugin starts a new one on its next request.

**An outdated server the plugin cannot stop stops the install.** If a cognee
server still answers on the local port after that and reports a cognee older
than the pinned one, or no version at all (for example one 1.4.2 or earlier
started without a pidfile, or one another cognee plugin started on an older
cognee; on Windows the plugin stops nothing), `install-runtime` leaves the ready
marker and the venv as they were and names the port, the version and, when
`lsof` is installed, the pid: stop that server and re-run
`hermes cognee install-runtime`. A server on the pinned cognee or newer (another
plugin's current one) does not block the install. While `install-runtime` runs,
a Hermes session that would start a server reports Cognee as unavailable
instead.

**Troubleshooting an interrupted install.** If the server log
(`~/.cognee-plugin/hermes/server.log`) shows an import error after an install
was cut short (a hard kill mid-download), rebuild the runtime from scratch:
`hermes cognee remove-runtime`, then `hermes cognee install-runtime`. Your data
in `~/.cognee` is kept.

**A running server on an older cognee is not used.** At session start the plugin
reads the version the server on `127.0.0.1:8011` reports on `/health`. If it is
older than the pinned cognee (for example a 1.6.5 server another plugin
started), Hermes reports Cognee as unavailable and names the port: auto recall
needs the newer server to route queries. Stop that server (`install-runtime`
stops it for you only when this plugin started it) and start a new Hermes
session.
A server that reports no version is used.

**Without an LLM key**, cognee builds memory with local models: the GLiNER
extractor for the graph and the fastembed `BAAI/bge-small-en-v1.5` embedder. The
first graph build installs CPU PyTorch and `gliner2` into the runtime (about
800 MB more) and downloads the GLiNER model (about 750 MB); later builds take
seconds. Recall then returns the stored text that matches rather than a
generated answer, and the graphs are simpler than with an LLM. For an LLM on your
machine instead, set `LLM_PROVIDER=ollama` (see
[Local models via Ollama](#local-models-via-ollama-embedding-settings)).

> **The embedder sticks to the store.** Keyless mode embeds with fastembed
> (384 dimensions); adding an LLM key later switches cognee's default embedder
> to OpenAI, and the existing store then refuses it with
> `EmbeddingDimensionMismatchError`. To add a key to a keyless store, also set
> `EMBEDDING_PROVIDER=fastembed` and `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` in
> `~/.hermes/.env`, or start a new store (`COGNEE_DATA_ROOT`,
> `COGNEE_SYSTEM_ROOT`).

At session start the plugin attaches to a cognee server already running on
`127.0.0.1:8011` — for example one a sibling cognee plugin (Claude Code, Codex,
OpenClaw) runs — or starts one from the runtime, with storage in `~/.cognee`. The
very first boot runs database migrations; on a fresh store that takes under a
minute, and after that it's instant.

Local mode needs the runtime even when a sibling plugin's server is already up:
Hermes asks a provider whether it is available before anything connects, and the
plugin answers from the runtime alone. Without it, Hermes starts the chat with its
built-in memory only and warns that `cognee` is unavailable, naming
`hermes cognee install-runtime`. A `cognee` installed into Hermes's own
environment by hand is also used when it is the pinned version or newer; an older
one (such as the 1.6.5 that 1.3.x installed as a dependency and `pip install -U`
leaves behind) keeps local mode off, and `hermes cognee status` shows it as
outdated. Hermes rebuilds that environment whenever it updates or the enabled
plugins change, which removes it; the runtime is the durable path.

> **Upgrading from 1.3.x in local mode:** run `hermes cognee install-runtime`.
> Earlier versions installed cognee into Hermes's environment as a dependency;
> from 1.4.0 the plugin declares none, so the first environment Hermes builds after
> the upgrade no longer has it and local mode reports unavailable until the
> runtime is installed. Your data in `~/.cognee` is untouched.

> **An existing `~/.cognee` can take much longer the first time.** If another
> cognee plugin has filled `~/.cognee` with an older cognee version, the first
> start migrates every stored table to the runtime's format before the server
> answers, which can take many minutes on a large store (cognee 1.6.5's vector
> table migration also holds a whole table in memory while it rewrites it). The
> session does not wait for it: after 90 seconds it starts with Cognee's tools
> answering "still starting", and Cognee turns on in that session once the
> server is ready (`~/.cognee-plugin/hermes/server.log` shows progress). Do not
> stop the server while it migrates. To keep Hermes's local store apart, set
> `COGNEE_DATA_ROOT` and `COGNEE_SYSTEM_ROOT` to another folder before the first
> start.

> **Switching modes? Re-run the wizard.** Values in `~/.hermes/cognee.json`
> take precedence over environment variables, and a local setup records
> `"service_url": ""` there — so *only* exporting `COGNEE_BASE_URL` will not
> move an existing local install to the cloud. `hermes memory setup cognee` (or
> `hermes cognee setup`) rewrites both files consistently.

### 3. Start a new chat and check

Start a **new chat** in Hermes Desktop (or a new `hermes` session). Memory is set
up when a chat starts, so a chat that was already open keeps the memory it
started with. Then:

```bash
hermes memory status     # Provider: cognee
hermes cognee status     # mode, dataset, service URL; "Local runtime: ready" in local mode
```

In local mode, `curl -s http://127.0.0.1:8011/health` also answers once the
server is up. Then, in a chat: *"Remember that my favorite editor is Helix"* —
the agent should call `cognee_remember`. Start another new chat and ask *"What's
my favorite editor?"* — it should recall it via `cognee_recall`.

If it doesn't work:

- **`Provider: (none)`** — the setup hasn't run (enabling the plugin is not
  enough): run `hermes memory setup cognee`.
- **The agent says Cognee is unavailable or saves to its built-in memory
  instead** — the chat started before Cognee was ready; start a new one.
- **"still starting"** (local mode) — the server is migrating an existing store;
  watch `~/.cognee-plugin/hermes/server.log`, and Cognee turns on in that chat
  once it is ready.

## How the pip install works

Hermes discovers memory providers two ways, and the package serves both:

- **Pip entry point** — the wheel declares
  `[project.entry-points."hermes_agent.memory_providers"]`, the group Hermes'
  memory loader scans, so the provider activates from a plain `pip install`.
- **Directory install** — `cognee-hermes-install` copies the plugin into
  `$HERMES_HOME/plugins/cognee/` in the exact shape the directory scanner
  expects; the load-bearing file is the root `__init__.py` (Hermes silently
  skips a plugin directory without one). The directory install is the
  recommended path: it carries the `hermes cognee` subcommands and the
  dashboard config panel at full fidelity.

For a pip-managed directory *copy*, upgrading takes two steps:
`pip install -U cognee-integration-hermes-agent`, then `cognee-hermes-install`
again (`hermes cognee status` reminds you when the copy is stale). Catalog
installations instead use `hermes plugins update cognee`.

Releases are published to PyPI by the `release_hermes_plugin.yml` workflow in
the cognee repository, run by hand from `main`. It tags the released commit
`hermes-agent-v<version>`.

## Configuration

The quick start above covers the common cases; this section is the full
reference. Configuration comes from two places: `$HERMES_HOME/.env` (secrets
and environment variables — Hermes loads it for every session) and
`$HERMES_HOME/cognee.json` (non-secret settings). The setup wizard writes both.
When a key appears in both places, **the JSON file wins** — which is why mode
switches should go through the wizard rather than editing the environment
alone.

### Modes

The provider connects to cognee in one of three modes. It picks the mode
automatically from your config:

| Mode | When it's used | How it talks to cognee |
| --- | --- | --- |
| **remote** (setup's default) | `COGNEE_BASE_URL` is set | thin client to your managed / cloud cognee |
| **local-server** | no `COGNEE_BASE_URL`, `COGNEE_EMBEDDED` unset | ensures a local cognee server is running and connects as a thin client |
| **embedded** | `COGNEE_EMBEDDED=true` | runs cognee in-process |

**Why local mode runs a server.** cognee's local stores (SQLite, Kuzu/Ladybug,
LanceDB) are single-writer. Driving them in-process from the agent's background
threads — or from a second Hermes process sharing the same `data_root` — risks
`database is locked` errors and corruption. A local cognee server is the single
owner that serializes all access, so the agent just makes HTTP calls. This is the
same design the Claude Code and Codex plugins use. **`embedded` is opt-in and is
safe for single-process / offline use only.**

**No silent fallbacks.** The provider never downgrades modes behind your back. If
`COGNEE_BASE_URL` is set but unreachable, or the local server fails to start,
initialization raises rather than quietly switching to a different mode — silent
fallback would either mask a config error (remote → local data divergence) or
reintroduce the very DB-lock risk this design removes (local-server → embedded).
To accept the single-process trade-off, set `COGNEE_EMBEDDED=true` explicitly.
And if initialization does fail, memory stays *off*: Hermes logs the error and
starts anyway, so the provider refuses every call rather than operating a
half-connected backend.

### Transports

Mode decides *where* cognee is; the transport decides *how* the plugin talks to it.

| Transport | Selected by | What it does |
| --- | --- | --- |
| **http** (default) | nothing to set | builds requests against cognee's REST API directly, using only the standard library |
| **sdk** | `COGNEE_TRANSPORT=sdk`, or any `COGNEE_EMBEDDED=true` | drives the `cognee` Python package, via `cognee.serve()` when a server is involved |

Direct HTTP is the default because it is what the Claude Code, Codex and OpenClaw
plugins do, and because the SDK's `CloudClient` drops fields the server accepts —
most importantly `session_ids` on `improve()`, which is what promotes a session's
turns into the permanent graph. Two consequences worth knowing:

- The `sdk` transport needs the `cognee` package in Hermes's own environment, which
  the plugin does not install (see embedded mode under
  [One brain across agents](#one-brain-across-agents)). The
  default `http` transport needs no cognee in Hermes's environment: remote mode is
  plain HTTP, and local mode's server runs from the private runtime.
- Over HTTP, a `cognee_remember` write cannot be linked to the session it came
  from — `/api/v1/remember` has no field for it. Session-to-graph bridging is
  unaffected. The plugin logs this once rather than dropping it silently.

> **Upgrading from 0.1.x — three defaults moved to match the other cognee
> plugins.** The local port changed from 8000 to 8011 (leaving cognee's own
> default of 8000 to servers you start yourself); the default dataset changed
> from `hermes` to the shared `agent_sessions`; and local storage now defaults to
> the shared `~/.cognee/{data,system}` instead of cognee's global default. Your
> old memory is not deleted, but a recall against the new dataset/roots will not
> see it — set `COGNEE_DATASET=hermes` (or migrate the data) and, if an old
> plugin-started server is still listening on 8000, **stop it**: two servers
> sharing one data directory is exactly the single-writer contention this mode
> exists to avoid. `COGNEE_LOCAL_PORT=8000` restores the old port.

### One brain across agents

By default this plugin joins the same memory the Claude Code, Codex and OpenClaw
cognee plugins share: the same dataset (`agent_sessions`), the same local storage
(`~/.cognee/{data,system}`), the same server port (8011) and the same API key —
which this plugin looks up on the local server at each session start rather than
reading the others' key file. Whichever plugin boots the server first,
the rest attach to it — and a fact remembered in Claude Code is recallable in
Hermes, and vice versa. A separate `COGNEE_PLUGIN_DATASET` keeps Hermes's saves
apart, and recall stays on it while no recall mode is set (`recall_mode: auto` in
`cognee.json` or `COGNEE_PLUGIN_RECALL_MODE=auto` lets it reach the other
plugins' datasets; a conversation can lift the pin with
`cognee_switch_dataset action='auto'`); for full isolation give it its own
`COGNEE_DATA_ROOT` / `COGNEE_SYSTEM_ROOT` *and* `COGNEE_LOCAL_PORT` — a server
belongs to whoever reaches its port first, so a private store needs a private
port. The same goes for its LLM key: a server shared by several profiles keeps
the key of the profile that started it, so profiles that must not share an LLM
bill need their own `COGNEE_LOCAL_PORT` too.

The per-mode settings below live in `~/.hermes/.env` (the wizard puts them
there; you can also edit the file by hand).

local-server mode (default — just set your LLM creds):

```bash
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
# COGNEE_PLUGIN_DATASET=agent_sessions   # optional; the default is shared with the other plugins
# COGNEE_LOCAL_PORT=8011                 # optional; the other plugins' server port
```

Remote / cloud mode (tenant URL and API key from the
[Cognee Cloud dashboard](https://platform.cognee.ai/)):

```bash
COGNEE_BASE_URL=https://tenant-xxx.aws.cognee.ai   # canonical name
COGNEE_API_KEY=...
```

> **`COGNEE_API_KEY` is mandatory for any remote server.** On a local server the
> plugin mints a key on first use (a one-time login as the default user); remote
> servers — Cognee Cloud included — expose no login route, so there is nothing to
> mint with. A remote `COGNEE_BASE_URL` without a key fails at startup with a
> clear error rather than a 401 on every call.

> **Where the default user's password comes from.** cognee >= 1.6.0 ships no
> built-in default-user password: the API server creates the default user at
> startup only when `DEFAULT_USER_PASSWORD` is set in *its* environment. The
> local server this plugin spawns always gets `DEFAULT_USER_EMAIL=default_user@example.com`
> / `DEFAULT_USER_PASSWORD=default_password` — the same literals the Claude Code,
> Codex and Antigravity plugins pass, since they all share this server — unless
> you export `DEFAULT_USER_EMAIL` / `DEFAULT_USER_PASSWORD` yourself, in which
> case your values win. `COGNEE_USER_EMAIL` / `COGNEE_USER_PASSWORD` do *not*
> change the server's default user: they only select which user the plugin logs
> in as to mint its key, and a non-default user must already exist on the server.
> If you point the plugin at a local server you start yourself, start it with
> `DEFAULT_USER_PASSWORD` set (matching `COGNEE_USER_PASSWORD` if you changed
> that), or set `COGNEE_API_KEY` to a key that server issued. The server sets
> the password once and never rewrites an existing user's, so a later change has
> to be made on the server as well. A login the server refuses
> (`This user does not have a password` / `LOGIN_BAD_CREDENTIALS`) is logged as
> a warning at startup and repeated on the first `401`, naming the fix.

Embedded (in-process) mode — single-process / offline only:

```bash
COGNEE_EMBEDDED=true
LLM_API_KEY=sk-...
```

Embedded mode, and the `sdk` transport (`COGNEE_TRANSPORT=sdk`), run cognee inside
the Hermes process, so they need `cognee==1.6.6` in Hermes's own Python environment.
The plugin cannot declare it (its dependencies conflict with Hermes's, which is
why local mode uses a private runtime), and the private runtime does not help here.
You can install it into Hermes's venv yourself
(`uv pip install --python <hermes venv python> cognee==1.6.6`), but this is
fragile: Hermes rebuilds its environment from its own lock whenever it updates or
the enabled plugins change, which removes it, and cognee's pins may not resolve
against your Hermes build at all. Prefer local mode unless you specifically need
no server. With an older cognee there (such as the 1.6.5 that 1.3.x installed),
embedded mode is unavailable, and Hermes's warning names the found version and
the command that upgrades it (`<Hermes's python> -m pip install cognee==1.6.6`);
`install-runtime` does not help embedded mode. Unset `COGNEE_EMBEDDED` to use
local mode instead.

> **Embedded mode and the shared store do not mix.** Embedded drives the local
> single-writer databases from inside the Hermes process; if another plugin's
> server (or another process) is using `~/.cognee` at the same time, that is
> exactly the contention embedded mode is warned about. For embedded use, point
> `COGNEE_DATA_ROOT` / `COGNEE_SYSTEM_ROOT` at a private location.

### Optional settings

| Setting | Env var | Default |
| --- | --- | --- |
| `dataset` | `COGNEE_PLUGIN_DATASET` (canonical; `auto` = default) | `agent_sessions` |
| `recall_mode` | `COGNEE_PLUGIN_RECALL_MODE` (`auto`: the server picks the dataset; `pinned`: recall stays on the active dataset; anything else = `auto`). Unlike the other settings, an exported `COGNEE_PLUGIN_RECALL_MODE` overrides `recall_mode` in `cognee.json`, which the setup wizard always writes | unset: `auto` for the default dataset, `pinned` for any other configured dataset |
| `top_k` | `COGNEE_TOP_K` | `5` |
| `auto_route` | `COGNEE_AUTO_ROUTE` (`true`: Cognee picks the search type for each query; `false`: always graph completion; not asked by `hermes memory setup`) | `true` |
| `improve_on_end` | `COGNEE_IMPROVE_ON_END` | `true` |
| `improve_background` | `COGNEE_IMPROVE_BACKGROUND` | auto |
| `session_prefix` | `COGNEE_SESSION_PREFIX` | `hermes` |
| `service_url` | `COGNEE_BASE_URL` (canonical) | empty |
| `embedded` | `COGNEE_EMBEDDED` | `false` |
| `local_port` | `COGNEE_LOCAL_PORT` | `8011` |
| `server_boot_timeout` | `COGNEE_SERVER_BOOT_TIMEOUT` | `600` |
| `data_root` | `COGNEE_DATA_ROOT` | `~/.cognee/data` |
| `system_root` | `COGNEE_SYSTEM_ROOT` | `~/.cognee/system` |
| `recall_timeout` | `COGNEE_RECALL_TIMEOUT` | `120` (seconds) |
| `write_timeout` | `COGNEE_WRITE_TIMEOUT` | `120` (seconds) |
| `improve_timeout` | `COGNEE_IMPROVE_TIMEOUT` | `300` (seconds) |
| `recall_budget` | `COGNEE_RECALL_BUDGET` | `20` (seconds, bounds the per-prompt recall) |
| `recall_passage_chars` | `COGNEE_RECALL_PASSAGE_CHARS` | `2000` (per recalled passage; `0` = no cap) |
| `recall_context_chars` | `COGNEE_RECALL_CONTEXT_CHARS` | `12000` (the whole injection; `0` = no cap) |
| `capture` | `COGNEE_CAPTURE` | `true` (automatic capture of turns and tool calls) |
| `capture_tools` | `COGNEE_CAPTURE_TOOLS` | `*` (tool allow list) |
| `capture_deny_tools` | `COGNEE_CAPTURE_DENY_TOOLS` | empty (tool deny list) |
| `capture_deny_paths` | `COGNEE_CAPTURE_DENY_PATHS` | empty (extra secret-file globs) |
| `capture_redact` | `COGNEE_CAPTURE_REDACT` | `true` |
| `capture_redact_patterns` | `COGNEE_CAPTURE_REDACT_PATTERNS` | empty (extra regexes, one per line) |
| `chunk_size` | `COGNEE_CHUNK_SIZE` | unset (server default; tokens per chunk for `cognee_remember`) |
| `memory_steer` | `COGNEE_MEMORY_STEER` | `true` |
| `memory_steer_text` | `COGNEE_MEMORY_STEER_TEXT` | built-in wording |
| `memory_hits` | `COGNEE_MEMORY_HITS` | `true` |
| `dataset_switch_tool` | `COGNEE_DATASET_SWITCH_TOOL` | `true` |
| `code_search_tool` | `COGNEE_CODE_SEARCH_TOOL` | `true` |
| `code_graph_recall` | `COGNEE_CODE_GRAPH_RECALL` | `true` |
| `code_datasets` | `COGNEE_CODE_DATASETS` | empty (comma-separated extra code datasets) |
| `recap_tool` | `COGNEE_RECAP_TOOL` | `true` |
| `update_check` | `COGNEE_UPDATE_CHECK` | `true` (CLI-only PyPI check) |
| `update_check_interval` | `COGNEE_UPDATE_CHECK_INTERVAL` | `3600` (seconds) |

> **Storage is shared, and a server is per port.** The roots above are the ones
> every cognee agent plugin pins, so the store is the same no matter which plugin
> booted the server on 8011. Because the default roots live outside
> `HERMES_HOME`, `backup_paths()` reports them to `hermes backup` — a profile
> backup deliberately includes the machine's shared memory store. Roots you point
> elsewhere with `COGNEE_DATA_ROOT` / `COGNEE_SYSTEM_ROOT` are reported the same
> way (unless they sit inside `HERMES_HOME`, which `hermes backup` walks anyway).

> `COGNEE_SERVICE_URL` is a deprecated alias for `COGNEE_BASE_URL`, and
> `COGNEE_DATASET` (the 0.1.x name) a lower-precedence alias for
> `COGNEE_PLUGIN_DATASET`. Both still work; new setups should use the canonical
> names.

> **`improve_background`** decides where the session-end graph build
> (`improve()`) runs. Default `auto`: whenever a server is involved, the close is
> handed to a **detached worker** — the same process that already covers crashes —
> which runs `improve()` to completion and only then unregisters the agent
> connection. Hermes exits immediately; nothing waits on the graph build. That
> ordering is required, not stylistic: the local server runs with
> `COGNEE_AGENT_MODE=true` and retires itself within 60s of the last agent
> unregistering, so unregistering first would kill the promotion halfway. In
> `embedded` mode there is no server and no worker, so the build runs in-process
> and synchronously — it dies with the process otherwise.
>
> Setting `COGNEE_IMPROVE_BACKGROUND=true|false` opts out of the handoff and does
> the work in-process: `true` submits the build and returns (right for a
> cloud/remote server nothing here can shut down; on a local server it
> reintroduces the race above), `false` blocks Hermes' exit until the build
> finishes.

## Local models via Ollama: embedding settings

Ollama embeddings need cognee's `ollama` extra — `pip install "cognee[ollama]"`
— which brings the `transformers` package the token counting depends on; with
plain `cognee` the embedding engine fails to construct at all.

cognee reads these standard variables (put them in `$HERMES_HOME/.env` — Hermes
loads it every session, so the spawned server inherits them; see
[.env.example](./.env.example)):

| Env var | Meaning |
| --- | --- |
| `EMBEDDING_PROVIDER` | `ollama` for a local embedder |
| `EMBEDDING_MODEL` | e.g. `all-minilm`, `nomic-embed-text` |
| `EMBEDDING_ENDPOINT` | usually `http://localhost:11434/api/embed` |
| `EMBEDDING_DIMENSIONS` | the model's vector size (e.g. `384` for all-minilm) |
| `EMBEDDING_MAX_COMPLETION_TOKENS` | **must be ≤ the model's context length** |
| `HUGGINGFACE_TOKENIZER` | the HF tokenizer matching the model, used to count tokens |

**Why the token ceiling matters.** cognee sizes its text chunks from
`EMBEDDING_MAX_COMPLETION_TOKENS`, and its default (8191) is far above any local
embedding model's real context. Every substantial document then overflows the
model; cognee splits the text and **mean-pools the vectors while still reporting
success**, so the pipeline completes but the search index quietly fills with
lossy embeddings and retrieval degrades — the only trace is an
`Ollama embedding error` line in `~/.cognee-plugin/hermes/server.log`. Worse,
recent Ollama versions default to *silently truncating* oversized inputs, so
depending on the Ollama version the index degrades with no log line at all —
which is why a correct token ceiling matters even when the log is clean.

The plugin therefore pins safe defaults at server spawn when
`EMBEDDING_PROVIDER=ollama`: a context-matched
`EMBEDDING_MAX_COMPLETION_TOKENS` (a conservative 512 for models it does not
recognize) and, for recognized models, the matching `HUGGINGFACE_TOKENIZER`.
Explicit values always win over the pins. When the plugin detects an overflow in
the server log anyway, recall/remember results carry a `warning`/`error` naming
these levers.

> **Pins apply at server spawn.** A cognee server already running on the port
> keeps the environment it was started with — after changing embedding settings,
> stop that server (it is shared with the other cognee plugins) so the next
> session respawns it. An index written with wrong settings stays wrong until
> the dataset is rebuilt: follow [RUNBOOK.md](./RUNBOOK.md).

**If recall is slow or times out**: the default `GRAPH_COMPLETION` search runs
an LLM per query, which local models make slow. `search_type=CHUNKS` returns
matching stored text directly with no LLM in the loop; `COGNEE_RECALL_TIMEOUT`
raises the deadline.

### Per-prompt memory block

Every prompt makes up to three requests, dispatched together under one deadline
(`recall_budget`); none of them calls an LLM on the server:

- **Memory**: `scope=["graph"]`, `search_type=HYBRID_COMPLETION`,
  `only_context=true`, and **no session id**. On cognee >= 1.6.0 the server
  answers with the full input its completion would have received; with a session
  id that input would also carry this conversation's recent turns, which Hermes
  already has. The plugin keeps only the retrieved context: the question
  template and history go, passages recorded in this very session go, each
  passage is capped at `recall_passage_chars` (cut at a paragraph break), and
  entity lines that say nothing are pruned. Text it cannot parse (an older
  server's bare context) is injected whole. Injected as `<cognee_memory>`.
- **Guidance**: `scope=["session_context"]` with the session id, against the
  active dataset: the lessons the server distilled from this session, injected as
  `<cognee_guidance>`. None yet is normal and costs nothing.
- **Code**: the deterministic code-graph lane, when the
  [code recall lane](#code-graph-index-a-repository) is armed.

`recall_context_chars` then caps the whole injection: guidance first, then
memory, then code facts; the block that crosses it is cut at a line and marked.
A lane that fails or misses the deadline is dropped without costing the others.
The trimming rules are shared with the Claude Code, Codex and OpenClaw plugins
(`integrations/conformance/recall_text_cases.json`).

After three prompts in a row whose recall only timed out, each prompt first
spends a 3-second `/health` probe and skips recall while the server does not
answer, instead of waiting out the whole budget. Any answered recall ends it.

The explicit `cognee_recall` tool reads the graph the same way, without the
session id. It takes `query`, an optional `search_type` and `top_k`. The raw
session-cache scopes (`session`, `trace`) and the `auto` scope are never
requested: turns and traces are written to the session cache, which `improve()`
promotes into the graph at session end, but they are not searched as raw entries.

### Memory credit in the answer

On by default, as in the Cursor and Claude Code plugins. When an answer uses
recalled memory, the agent ends it with one plain line:

```
> Cognee: Mika likes Yuji and volleyball, based on dataset agent_sessions.
```

In Hermes Desktop the plugin replaces `Cognee:` with the Cognee pixel logo in the
brand purple (`#BC9BFF`), through Hermes' `transform_llm_output` hook. It is drawn
in KaTeX, which Desktop renders: one coloured `\rule` per run of squares on the
logo's five-row grid, about a capital letter tall, followed by half an em of space
(about two spaces) and the sentence. The drawn line is not a blockquote (Desktop draws those with a
quotation mark). The logo is always exact
because the agent only writes the plain line. The rewritten answer is what
Hermes stores, so each credited answer carries about 800 characters of LaTeX in
the chat history.

In the terminal chats, the terminal UI (`hermes --tui`) and the classic chat
(`hermes`), the answer streams to the screen before the hook runs, so the line the
agent wrote stays. There the agent is asked for the line without the `Cognee:` label,
and the Cognee logo in braille is printed under it as a signature:

```
> Mika likes Yuji and volleyball, based on dataset agent_sessions.

⡏⠁⡏⡇⡏⡁⡧⣸⢸⠅⡯
⠓⠂⠓⠃⠓⠃⠃⠙⠘⠂⠓
```

Rewriting the line instead would show the answer twice: the classic chat prints a
replaced answer again, and the terminal UI keeps the streamed text and adds the new
answer under it. With streaming off in the classic chat
(`hermes config set display.streaming false`), the line is replaced by the logo with
the sentence beside its first row instead. Messaging apps show the plain line for now.

How it works:

- Every recalled item the agent is given is tagged `m1`, `m2`… with where it came
  from: a Claude Code, Codex, Cursor or Hermes session, a dated session learning,
  a document, graph entities or facts, code-graph facts, or agent guidance. That
  covers the per-prompt memory block and the results of the agent's own
  `cognee_recall` calls. Tags are numbered across the conversation, and a memory
  recalled again keeps its tag.
- On turns where memory was given, an instruction above it says how many memories
  Cognee found and asks the agent to add the credit only when the answer used a
  tagged item, never for what it knows from instruction files, its own memory
  files, the code or the conversation. The line names the dataset, not the tags.
  The first credit in a conversation also says in a few words how it connects to
  the earlier work.
- When the turn is stored, the credit is noted (logged as `cognee memory_credit`
  at INFO) and stripped, so it is never stored back into memory as part of the
  answer.

Turn it off with `COGNEE_PLUGIN_CITE_MEMORY=off` in `~/.hermes/.env`.

### Capture policy and redaction

Each completed turn is stored as a `qa` entry and each of its tool calls as a
`trace` entry (tool name, arguments, result, status), read from the
conversation Hermes hands the provider. The rules are the ones the Claude Code,
Codex and OpenClaw plugins use (`integrations/conformance/capture_policy_cases.json`):

- `capture_tools` / `capture_deny_tools` choose which tool calls are captured:
  a name, an fnmatch glob, or `tool(prefix:*)` matched against the first word of
  a shell command (`terminal(git:*)`), case-insensitively. Separate entries with
  `|`, or give a JSON array.
- A call whose path arguments or shell command name `.env*`, `*.pem`, `*.key`,
  ssh or cloud credentials, `secrets.*` or `credentials.*` (plus
  `capture_deny_paths`) is never captured. Neither is a call to the cognee tools
  themselves, or a shell command that mentions cognee.
- `capture_redact` (on by default) scrubs private keys, bearer and basic tokens,
  database URLs, credential assignments, vendor keys, AWS key ids, bcrypt hashes
  and secret-named fields from captured turns, tool calls and mirrored Hermes
  memory writes, plus `capture_redact_patterns`. Redaction runs before
  truncation (arguments at 4,000 characters, results and turns at 8,000).
- A pattern the plugin cannot parse fails closed: nothing is captured until it is
  fixed, and the Hermes log says why.
- `capture=false` turns automatic capture off. What you store with
  `cognee_remember` is never filtered or redacted.

Tool traces need the HTTP transport (the default); the in-process SDK transport
stores the turns only.

### Writes the server never received

A turn or trace that cannot reach the server because the connection could not
be opened (refused while it boots, host unreachable, DNS failure) or because it
answered 503 is queued in `~/.cognee-plugin/hermes/pending-entries.json` (newest
500, kept a week). The queue is replayed after the next successful write, when a
session starts, and, for the ending session, before its session-end `improve()`.
A write that timed out or was cut off mid-request is not replayed, since the
server may already have stored it. While the circuit breaker is open, writes go
straight to the queue.

### What this plugin runs, stores and sends

- **Runs**: in local mode, a cognee server (`uvicorn cognee.api.client:app`) on
  `127.0.0.1:8011` from the plugin's cognee runtime
  (`~/.cognee-plugin/hermes/runtime/venv`, built only when you ask for it), in
  `~/.cognee-plugin/hermes/`, with its output in
  `~/.cognee-plugin/hermes/server.log`; a detached exit watcher per Hermes
  process that closes the session if Hermes cannot. The server gets its
  settings from Hermes (the active profile's secrets and environment), not from a
  `.env` in the directory you start Hermes from; the watcher gets the API key
  from the plugin when it starts and reads no key file.
- **Stores locally**: `~/.cognee/` (the shared cognee data, unless you move it),
  and under `~/.cognee-plugin/hermes/`: the write queue, dataset
  switches, indexed code graphs, exit-watcher state and the update-check cache.
  Plugin settings are in `$HERMES_HOME/cognee.json`.
- **Sends** to the cognee server it is configured for (local, or
  `COGNEE_BASE_URL`): each turn and its tool calls (after the capture policy and
  redaction), your prompt as the recall query, mirrored Hermes memory writes,
  what you ask it to remember, forget or index, and `improve()` at session end
  and before each context compression. The server, not the plugin, calls your LLM and embedding
  providers; what else the cognee server sends is governed by its own settings.
  The plugin's only other request is the CLI's update check, which asks PyPI
  for the latest release version at most once an hour.

### Server settings and `.env` files

cognee >= 1.6.2 loads the first `.env` it finds walking up from its working
directory and lets it override the environment. The server this plugin starts
therefore runs in `~/.cognee-plugin/hermes/`, so a `.env` in the directory you
start Hermes from never reaches it.

Keys and settings come through Hermes only: `hermes memory setup cognee` stores
`COGNEE_API_KEY`, `LLM_API_KEY` and `COGNEE_BASE_URL` in the profile's `.env`
through Hermes' own writer, and the plugin reads them back with Hermes'
`get_secret`, which picks the right profile when one gateway serves several.
The plugin never opens a `.env` file itself. **`~/.cognee/.env`, the file the
Claude Code, Codex and OpenClaw plugins share, is not read by this plugin:** if
your keys live only there, run `hermes memory setup cognee` once, or put them in
Hermes' `.env`.

## Code graph: index a repository

Repositories are indexed explicitly (Hermes is rarely launched inside a
checkout, so there is no auto-indexing):

```bash
hermes cognee index-repo ~/work/my-service            # local path
hermes cognee index-repo https://github.com/o/repo    # URL (the server clones it)
hermes cognee index-repo ~/work/my-service --wait 120 # block until queryable
```

Each repository gets its own `codebase-<repo>-<digest>` dataset. Indexing is
deterministic — no LLM or embedding calls on either side (add semantic search
over code entities with `--index-vectors`). Requires a cognee server >= 1.5.4.

Once indexed, two things light up in a Hermes session:

- the **`cognee_code_search` tool** — exact structural answers: `query_facts`,
  `explore`, `traverse`, `find_path`, `impact_analysis`, `delta`;
- the **code recall lane** — a prompt naming an identifier-shaped token
  (`process_payment`, `UserService`, `billing/api.py`) while Hermes runs inside
  an indexed repo gets code-graph facts injected alongside the memory block.
  For repos indexed elsewhere, list their datasets in `COGNEE_CODE_DATASETS`.

A locally indexed path reflects the working tree at index time; a URL-indexed
repo reflects the last pushed commit. Re-run `index-repo` after significant
changes — the server's content hashes make re-runs cheap.

## Hermes Commands

When Cognee is the active memory provider:

```bash
hermes cognee status [--check-updates]
hermes cognee version [--check-updates]
hermes cognee setup
hermes cognee config
hermes cognee install
hermes cognee install-runtime [--yes]  # local mode: install or update the private cognee runtime (stops this plugin's server, asking first)
hermes cognee remove-runtime      # local mode: delete it (data in ~/.cognee is kept)
hermes cognee index-repo <path-or-url> [--dataset D] [--index-vectors] [--wait SECONDS]
hermes cognee keys list [--plugin KEY] [--json]      # per-machine keys of your plugin agents, most recently used first
hermes cognee keys revoke <key_id> --plugin KEY      # revoke one (frees a slot on a full agent)
hermes cognee standup  [--since 24h] [--projects P,…] [--all-sessions] [--json]
hermes cognee digest   [--since 7d] [--max-learnings N] [--json]
hermes cognee timeline <topic> [--since 30d] [--json]
```

`standup`, `digest` and `timeline` print a recap of the sessions recorded into
the configured dataset (`--dataset` picks another): what was worked on and left
open, the lessons the server distilled into memory, or a dated timeline of one
topic. Only sessions recorded into that dataset are listed, so on a shared server
a recap never shows other users' conversations; `--all-sessions` adds other
clients' sessions in it (Claude Code, Codex). The server cannot filter sessions
by dataset, so a recap reads up to 2,500 of the newest sessions and says so when
it had more. The same recaps are available to the agent as the `cognee_recap`
tool.

`keys` manages the API keys of your cognee plugin agents (Claude Code, Codex, …):
every machine a plugin runs on holds its own key for that plugin's agent, and the
server caps them. A full agent refuses a new machine a key — that machine then runs on
your personal key and says so at session start — until an old machine's key is revoked,
which you can do from here.

For pip installations, `status` and `version` include an update hint when PyPI has a newer release
(checked at most once per `COGNEE_UPDATE_CHECK_INTERVAL`, never from a live
session): update with `pip install -U cognee-integration-hermes-agent` and then
`cognee-hermes-install`, since Hermes runs the installed copy.

For catalog installations, these commands show the running plugin version and
direct you to `hermes plugins update cognee`. They never query PyPI, even with
`--check-updates`; that flag prints catalog update guidance rather than checking
for a new catalog release. `hermes plugins update cognee` checks the catalog.

## Development

```bash
cd cognee-integrations/integrations/hermes-agent
uv sync --dev
uv run pytest -q
uv run ruff check .
```
