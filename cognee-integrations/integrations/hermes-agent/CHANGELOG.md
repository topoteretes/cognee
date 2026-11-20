# Changelog

All notable changes to the Cognee memory plugin for Hermes Agent are documented
here.

The version must match the `version` field in both `pyproject.toml` and
`plugin.yaml` (the root copy and `cognee_integration_hermes/_plugin_root/`).

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **The plugin's source moved to the cognee repository (SDK-1489).** It now lives in
  `cognee-integrations/integrations/hermes-agent` in
  [topoteretes/cognee](https://github.com/topoteretes/cognee); earlier history is in
  `integrations/hermes-agent` of topoteretes/cognee-integrations. The PyPI package name,
  the entry points and `cognee-hermes-install` are unchanged.

## [1.4.3]

### Added
- **Recall is auto on the default dataset (SDK-1393).** Unless recall is pinned, the
  per-prompt memory lane and `cognee_recall` name no dataset and the server picks the
  best-matching one.
  Routing needs a cognee 1.6.6 server; an older one searches every dataset the user can read.
  Saves still go to the conversation's dataset. While recall is auto, the memory credit says
  "based on Cognee memory" rather than naming the active dataset. The guidance lane,
  `cognee_code_search` and recaps keep their datasets.
- **The recall mode is a setting: `recall_mode` / `COGNEE_PLUGIN_RECALL_MODE`.** `auto` lets
  the server pick the dataset; `pinned` keeps recall on the active dataset (reason
  `configured_pin`). Set it with the env var, `recall_mode` in `cognee.json`, or the setup
  wizard's new recall-mode question. An unknown value is `auto`. With no mode set, the default
  dataset is `auto` and a configured non-default dataset stays `pinned` (reason
  `configured_pin`), so a dataset chosen to keep agents or projects apart keeps recall apart
  too; the wizard offers `pinned` for such a dataset. A set mode wins, and a conversation's own
  pin or `auto` wins over both.
- **A one-time note that auto exists.** A user whose recall stays on a configured dataset only
  because no mode is set gets one line appended to the first reply of a session, once per
  machine and dataset (marker under `~/.cognee-plugin/hermes/notices/`), and logged: "Cognee:
  recall stays on `<dataset>`. Cognee can also pick the best-matching dataset for each prompt
  from all you can read: set `recall_mode: auto` in cognee.json (or
  `COGNEE_PLUGIN_RECALL_MODE=auto`, or rerun `hermes memory setup cognee`), or ask to set recall
  to auto for this conversation." It is never stored with the answer. `hermes cognee status`
  shows the recall mode in effect.
  `COGNEE_PLUGIN_DATASET=auto` (any case) means the default dataset and says nothing about the
  mode; before, saves went to a dataset named `auto`.
- **`cognee_switch_dataset` sets the conversation's recall mode, which wins over the
  configured one.** New `action='pin'` pins recall to the active dataset (no new session); a
  `switch` pins recall to its target, and a `switch` to the dataset in use behaves as `pin`.
  `action='auto'` lets the server pick for that conversation, lifting a configured pin too (the
  reply names it in `lifted`). `reset` goes back to the configured dataset and recall mode.
  `current` reports the mode in effect and, when pinned, a `reason` as the other cognee
  plugins do: `dataset_picked` or `configured_pin`. Every result carries one `message` line
  built from the mode and reason. `switch` refuses the name `auto`. The memory credit names
  the dataset whenever recall is pinned.
- **The mode is kept in `dataset-overrides.json` as a mode.** A pin or `auto` without a switch
  stores only `{"recall": ...}`: recall applies to the active dataset, which follows the
  configuration, so after a change of the configured dataset a pinned conversation is pinned
  to the new one (reason `dataset_picked`). A switch stores its dataset and counter too.
  Entries written before this release (no `recall`) stay pinned, except one on the configured
  dataset (indistinguishable from a reset), which follows the configuration.

### Changed
- **The terminal logo names the dataset.** In `hermes` and `hermes --tui` the braille
  logo the plugin adds under a streamed answer now reads `COGNEE  based on alice_in_wonderland
  dataset` ("a and b datasets" for several), taken from the datasets the credit line names; a
  line crediting only "Cognee memory" keeps the logo alone. In the classic CLI it follows the
  answer's closing line with no blank lines in between.
- **Hermes Desktop sets the credited dataset as code.** The drawn credit line shows
  "based on dataset `live_context_support_agent`" with the name in inline code, so it stands
  out from the sentence. Only a name that ends the line or leads to the next named dataset
  is set; the terminal chats and messaging apps keep the line as the agent wrote it.
- **The memory credit names the dataset an auto recall came from.** While recall is auto the
  credit said "based on Cognee memory", because the server picks the datasets. Each recalled
  memory now carries the dataset the server reports for it (`dataset_name` on the search
  result) in its tag, `[m1 · Cognee memory · dataset shop_support]`, and the credit names the
  datasets of the items the answer used ("based on dataset shop_support and dataset returns").
  The same goes for `cognee_recall` results. When a result names no dataset (an older server),
  the credit says "based on Cognee memory" as before.
- **The setup wizard asks about recall once.** `hermes memory setup` no longer asks
  "Auto-route recall" right after the recall mode, where it read as the same question twice.
  The two are different settings: `recall_mode` is which datasets recall reads, `auto_route`
  is whether Cognee picks the search type for each query. `auto_route` keeps its saved value
  and is set in `cognee.json` or with `COGNEE_AUTO_ROUTE`.
- **Local mode installs cognee 1.6.6** (was 1.6.5), and a runtime installed for 1.6.5 keeps
  local mode off until `hermes cognee install-runtime` installs 1.6.6: the plugin never starts
  a server on a cognee it was not written against, and never downloads one behind the user's
  back. Hermes reports Cognee as unavailable with that command, and `hermes cognee status`
  shows the runtime as outdated. `install-runtime` upgrades `runtime/venv` in place; a failed
  or interrupted install leaves no ready marker, so local mode stays off and the next
  `install-runtime` repairs the venv. Then start a new Hermes session.
- **`install-runtime` stops the local server this plugin started before it installs, and
  asks first.** A server still running from the venv would load modules of the new version
  into the old process. The plugin now writes a pidfile when it starts a server
  (`~/.cognee-plugin/hermes/runtime/servers/<port>.pid`) and stops servers by it, checking the
  pid is alive and still runs `cognee.api.client:app` (a reused pid is left alone). It says
  "This stops the local cognee server (pid N) on port 8011; other cognee plugins using it will
  restart it on their next request." and asks `Continue? (y/n)` (default yes); without a
  terminal it goes ahead only with the new `--yes` flag. The setup wizard's install question
  names the pid when a server is running. The server gets SIGTERM only, never SIGKILL, and up
  to 60 seconds; one still running after that is left running and the install stops with the
  ready marker and the venv untouched. Other cognee plugins sharing that server start a new
  one on their next request. A session that would start a server while `install-runtime`
  holds its lock reports Cognee as unavailable instead of starting one on a half-upgraded venv.
- **`install-runtime` changes nothing while an outdated cognee server still answers on the
  local port.** When the server left on the port reports a cognee older than the pin, or no
  version (one 1.4.2 or earlier started without a pidfile, where agent mode makes it exit once
  idle; one another plugin started on an older cognee; any server on Windows, where nothing is
  stopped for you), the install stops before it touches the ready marker or the venv, and
  names the port, the version and, when `lsof` is installed, the pid to stop before re-running
  it. A server on the pinned cognee or newer, such as another plugin's current one, does not
  block the install. The outdated-server and
  outdated-runtime messages no longer promise that `install-runtime` stops every server.
- **Embedded mode needs the pinned cognee in Hermes's own environment.** Embedded mode
  (`COGNEE_EMBEDDED`, `embedded` in `cognee.json`) runs cognee in the Hermes process, so it is
  available only when that environment holds cognee 1.6.6 or newer; the runtime venv never
  counts for it. With an older one, Hermes's warning names the found version and the upgrade
  command (`<python> -m pip install cognee==1.6.6`), or says to unset `COGNEE_EMBEDDED`; it
  never points to `install-runtime`, which embedded mode does not use.
- **A cognee older than the pin in Hermes's own environment is never started.** 1.3.x
  installed cognee 1.6.5 as a dependency, and `pip install -U` leaves it behind; local mode
  used to start the server from it, so recall was never routed and nothing said why. It is now
  used only at the pinned version or newer; otherwise local mode stays off with the
  `install-runtime` message, and `hermes cognee status` shows it as outdated.
- **A running local server on an older cognee is not used.** The server on the local port is
  shared with the other cognee plugins and may be one of them running an older cognee. When
  its `/health` version is older than the pin, Hermes reports Cognee as unavailable, naming
  the port, until it is replaced. A server that reports no version is used.
- **`COGNEE_PLUGIN_RECALL_MODE` overrides `recall_mode` in `cognee.json`.** The setup wizard
  always writes `recall_mode`, so the exported variable did nothing after setup. It now wins,
  as in the hook-family plugins (every other setting still lets `cognee.json` win), and a
  configured pin names where it was set: `COGNEE_PLUGIN_RECALL_MODE=pinned` or
  `recall_mode "pinned" in cognee.json`.

## [1.4.2]

### Fixed
- **A secret used as a dict key is redacted too.** `redact()` checked a key only to
  decide whether to mask its value, so a secret that was itself the key (a map keyed by
  DSN or token in a tool's structured output) was stored as is, while the same string
  as a value was redacted. Keys now go through the same patterns as strings; a
  secret-named key (`api_key`, `*_token`, …) still masks its value. Keys that redact to
  the same text collapse into one, and the last value wins. Matches Claude Code, Codex,
  Antigravity, Cursor and OpenClaw (shared conformance table).

## [1.4.1]

### Added
- **The memory credit carries the Cognee logo.** A `transform_llm_output` hook draws it, so
  the logo is always exact: the agent only writes a plain line. (Asking the agent to copy a
  braille logo itself failed live: it dropped cells, lost the colour markup and copied the
  placeholder.)
  - **Hermes Desktop:** `Cognee:` in the agent's `> Cognee: …` line becomes the Cognee pixel
    logo in the brand purple (`#BC9BFF`), drawn in KaTeX as coloured rules on the logo's
    five-row grid, followed by half an em of space and the sentence. The quote marker is
    dropped (Desktop draws a blockquote with a quotation mark).
  - **Terminal chats (`hermes --tui` and `hermes`):** the answer streams before the hook runs
    and a rewritten answer would be shown twice, so the agent is asked for the line without
    the `Cognee:` label (`> <what it remembered>, based on dataset <name>.`) and the Cognee
    logo in braille is appended below it as a signature. In the classic chat with
    `display.streaming` off, the logo replaces the quote marker instead, with the sentence
    beside its first row.
  - Messaging apps keep the plain `> Cognee:` line for now.
  - In the terminal the unlabelled line is recognised by where it is and what it names: the
    answer's last line, quoted, naming `dataset <name>` or `Cognee memory`, however the rest is
    worded (seen live: "stored in dataset agent_sessions", not "based on").
  - Every form of the credit, drawn or plain, is stripped before the turn is stored.

### Fixed
- **Local mode without its runtime now says how to fix it.** Hermes never calls
  `initialize()` for a provider that reports unavailable, so the "run
  `hermes cognee install-runtime`" error never reached the user: they saw only Hermes's
  generic "check credentials/config" warning. The provider now implements
  `unavailable_reason()`, which Hermes appends to that warning (the embedded-mode wording
  when `COGNEE_EMBEDDED=true` and cognee is not in Hermes's environment).
- **Upgrading from 1.3.x in local mode needs `hermes cognee install-runtime`.** 1.3.x
  installed cognee into Hermes's environment as a dependency; Hermes drops it from the first
  environment it builds after the upgrade. The README now says so, and corrects 1.4.0's
  claims: a server another cognee plugin runs on the port is used only once the runtime is
  installed (availability is decided before anything connects), and a cognee installed into
  Hermes's environment by hand does not survive Hermes rebuilding that environment.
- **`hermes plugins validate` passes.** `plugin.yaml` declares the `transform_llm_output`
  hook under `provides_hooks`; it was registered but undeclared, which the catalog check
  rejects.

## [1.4.0]

### Changed
- **cognee is no longer a dependency, so the plugin installs on any Hermes.** Hermes resolves
  every plugin into one shared environment, and cognee 1.6.5's pins conflict with Hermes
  extras there (`huggingface-hub<1` against `hermes-agent[trace-upload]`, and litellm's
  `boto3` floor against `hermes-agent[bedrock]` on older Hermes builds), so
  `hermes plugins enable cognee` was refused for everyone, Cognee Cloud users included.
  `pyproject.toml` now declares no dependencies and `plugin.yaml` no `pip_dependencies`.
  Remote mode is plain HTTP and never needed cognee.
- **Local mode runs from a private cognee runtime.** The plugin installs `cognee==1.6.5` into
  a venv of its own, `~/.cognee-plugin/hermes/runtime/venv` (about 1 GB from PyPI), and starts
  the local server from it, so cognee never enters Hermes's environment. The venv is built from
  the interpreter running Hermes, with uv when it is on PATH (never downloading an interpreter)
  or stdlib `venv` + pip otherwise. It is installed only on request: `hermes memory setup
  cognee` offers it when you choose local mode, `hermes cognee install-runtime` installs it or
  moves it to the plugin's pinned cognee after an update, and `hermes cognee remove-runtime`
  deletes it; nothing is downloaded when a session starts. One runtime serves every Hermes
  profile, as the server does. `hermes cognee status` shows a "Local runtime" line. A server
  another cognee plugin already runs on the port is still used, and a cognee installed into
  Hermes's environment by hand still works. Embedded mode (`COGNEE_EMBEDDED=true`) needs cognee
  in Hermes's own environment.
- **Setup defaults to remote mode.** `hermes memory setup cognee` lists remote (Cognee Cloud or
  another cognee server) first and offers it as the default; an install that chose local sees
  "Current mode: local" above the prompt. Choosing local clears `COGNEE_API_KEY` along with
  `COGNEE_BASE_URL`, so a Cognee Cloud key is no longer sent to the local server. A
  `COGNEE_API_KEY` from the environment that a *local* server rejects (401/403, e.g. one left
  by an earlier setup) is replaced by the local server's own key; a key it accepts, or that
  cannot be checked, is used as before.
- **Availability.** The provider is available with a service URL (remote), or once the local
  runtime is ready (local). It no longer also requires `LLM_API_KEY`: that refused keyless
  providers (Ollama), and the server bootstrap already warns by name when it starts without a
  key.
- **A local server that is still starting no longer holds the session for up to 10 minutes.**
  Session start waits 90 seconds; if the server it started is alive but not answering yet (a
  first start migrating an existing `~/.cognee`), the session starts anyway, Cognee's tools
  answer that the server is still starting and where its log is, and the provider connects in
  the background once `/health` answers (within the same 10-minute boot deadline, never by
  starting a second server). Other failures report their own cause (exit code, log path)
  instead of a generic list of port, dependency and `COGNEE_EMBEDDED` suggestions.
- **Setup instructions are one path everywhere.** Installing or enabling the plugin
  (`hermes plugins enable cognee`, the Desktop toggle) does not make it Hermes's memory
  provider, so memory stayed off when the old "install, enable, setup" steps stopped early.
  The plugin description, `after-install.md`, the README and the `hermes cognee install` /
  `cognee-hermes-install` hints now give the same steps: install, run
  `hermes memory setup cognee` in a terminal, start a new chat.
- **Cognee first, Hermes' built-in memory as the fallback.** An agent shown both a recalled
  Cognee memory and one of Hermes' own memory notes answered from the (stale) note. The
  system prompt now says Cognee is the primary memory, that `MEMORY.md` / `USER.md` are the
  fallback for when Cognee has nothing, and that when they disagree it goes with Cognee and
  points out the difference; each prompt's Cognee memory block repeats that in one line,
  next to the memory. While Cognee is unavailable the plugin adds nothing, so Hermes' memory
  is used as before. `COGNEE_MEMORY_STEER=false` turns both off.
- **Memory credit in the answer**, as in the Cursor and Claude Code plugins: recalled items
  (the per-prompt memory block and `cognee_recall` results) are tagged `[mN · where it came
  from]` with one numbering per conversation, and an instruction asks the agent to end an
  answer that used them with one plain line, `> Cognee: <what it remembered>, based on
  dataset <name>`. The credit is noted and stripped before the turn is stored. On by default;
  `COGNEE_PLUGIN_CITE_MEMORY=off` turns it off.
- **Local mode works without an LLM key.** cognee 1.6.5 then builds memory with local
  models (GLiNER extraction, fastembed embeddings) installed into the runtime and downloaded
  on the first build (about 1.5 GB); recall returns the matching stored text. The setup's LLM
  key prompt says a blank answer means local models, the server's no-key message says the
  same instead of "memory will not be built", and the README explains that the embedder
  sticks to the store (with how to add a key later).
- **The local server no longer reads `~/.cognee-plugin/.env`.** cognee >= 1.6.2 loads the
  first `.env` found walking up from its working directory, with that file overriding the
  environment; from the plugin's state directory the walk reached `~/.cognee-plugin/.env`,
  the file the other cognee plugins keep their keys in, so the server ran on their LLM key and
  settings instead of the ones Hermes passed in. The server now starts with `COGNEE_ENV_FILE`
  pointing at an empty file of the plugin's own (`~/.cognee-plugin/hermes/cognee-server.env`);
  a `COGNEE_ENV_FILE` you set yourself still wins.
- CI installs `local-mode-requirements.txt` (the same pin) for the tests that run against
  cognee, and runs them with `uv run --no-sync` so the install is kept.

## [1.3.3]

### Added
- **`hermes cognee keys list` / `hermes cognee keys revoke <id> --plugin <key>`** manage
  the per-machine API keys of your cognee plugin agents (Claude Code, Codex, …). Cognee's
  provision route now adds one key per machine and caps them (cognee#6113); a full agent
  refuses a new machine a key, and this is how a slot is freed from Hermes.

## [1.3.2]

### Changed
- **Pinned cognee is 1.6.5** (was 1.6.3; `pyproject.toml` and `plugin.yaml`). The
  local server the plugin starts runs that release's one new migration on its next
  start: additive columns that count embedding tokens apart from LLM tokens. 1.6.5
  lets typed session entries carry a `node_set` and refuses a typed entry that names
  a different dataset than the one its session is bound to. The Claude Code, Codex,
  Antigravity and OpenClaw plugins move to 1.6.5 at the same time.

## [1.3.1]

Credentials come through Hermes only, as the plugin catalog's rule 11 requires:
the plugin no longer opens a `.env` file or a key file on its own.

### Changed
- **`~/.cognee/.env` is no longer read.** 1.3.0 handed that file's values to the
  local server it starts. The server now gets its settings from Hermes: its LLM
  key through Hermes' `get_secret` (the active profile's), the rest from the
  environment Hermes runs in. It still starts in `~/.cognee-plugin/hermes`, so a
  `.env` in the directory Hermes was started from still never reaches it. **If
  your keys live only in `~/.cognee/.env` (shared with the Claude Code, Codex and
  OpenClaw plugins), run `hermes memory setup` once or put them in Hermes'
  `.env`.**
- **Keys are read through Hermes' `get_secret`.** `COGNEE_API_KEY`,
  `LLM_API_KEY`, `COGNEE_BASE_URL` (and its `COGNEE_SERVICE_URL` alias) and
  `COGNEE_USER_EMAIL` / `COGNEE_USER_PASSWORD` were read straight from
  `os.environ`, which under a multi-profile gateway can hold another profile's
  key. On a Hermes without `get_secret` the environment is read as before.
- **`hermes memory setup` saves through Hermes' own `.env` writer**
  (`hermes_cli.config.save_env_value`, under the profile's home) instead of
  rewriting the file itself, so names and values get Hermes' validation and the
  0600 atomic write.
- **The exit watcher reads no key file.** It used the key in its environment and
  fell back to `~/.cognee-plugin/api_key.json`. It now uses only the key the
  plugin resolved and handed it at spawn time, and never one inherited from
  Hermes' environment. The plugin arms the watcher after connecting, so that key
  is the one the session uses; when the session has none (auth off, or a failed
  mint) the watcher logs that its close may be rejected, in `exit-watcher.log`.
- **No API key file is read or written.** For a local server the plugin cached
  the key it minted in `~/.cognee-plugin/api_key.json` and reused a key another
  cognee plugin had left there. It now keeps the key in memory for the process
  and, at a process's first connect, logs in to the local server and reuses the
  key the server already has (creating one only when none exists), so it still
  lands on the same key as the Claude Code, Codex and OpenClaw plugins and keys
  do not pile up. The remembered key is tied to the server and the login
  (`COGNEE_USER_EMAIL` plus a digest of the password, never the password), so a
  gateway serving profiles with different cognee users never hands one profile
  another's key, and a wrong password is not skipped past. The login and key
  lookup are capped at 10 s each, so a stalled server cannot hold session start
  for the full connect timeout. **A remote server or Cognee Cloud now always needs `COGNEE_API_KEY`
  configured in Hermes**; a key cached by another plugin no longer stands in.
- **The spawned server never inherits another profile's API keys.** Each key
  cognee's LLM and embedding clients read (`LLM_API_KEY`, `EMBEDDING_API_KEY`,
  `LLM_EXTRACTION_API_KEY`, `LLM_QUERY_API_KEY`, `LLM_SUMMARIZATION_API_KEY`) is
  dropped from the inherited environment and set from the profile's own scope
  (`get_secret`), so a profile no longer boots the server on, or bills its
  embeddings to, a key the gateway holds for another profile. Non-secret
  settings (provider, model, endpoint) are inherited as before.
- **Starting the server without an LLM key is logged**, once per process, as a
  warning naming `hermes memory setup`; before, the first sign was a failed
  cognify in `server.log`. A keyless provider (`LLM_PROVIDER=ollama`) is not
  warned about.

### Removed
- `cognee_integration_hermes/env_file.py`.

## [1.3.0]

Brings the plugin to cognee 1.6.3 and to parity with the Claude Code, Codex and
OpenClaw plugins: recall trimmed to what Hermes does not already have, tool-call
capture under the shared capture policy with secret redaction, a write buffer
for turns captured while the server is away, the recap tool and commands, and a
local server that no longer reads a stray `.env`. Also folds in two community
contributions: the pre-compression improve (#487) and background permanent
writes (#479).

### Added
- **Tool calls are captured as traces, under the capture policy, with secrets
  redacted.** Until now only the turn's text was stored. Each turn's tool calls,
  read from the conversation Hermes passes to `sync_turn`, are now stored as
  `trace` entries (tool, arguments, result, status) next to the turn, which the
  session-end `improve()` distils with the rest. The rules are the Claude Code,
  Codex and OpenClaw `COGNEE_CAPTURE_*` ones: `capture_tools` /
  `capture_deny_tools` take a name, an fnmatch glob, or `tool(prefix:*)` matched
  against the first word of a shell command, case-insensitively; calls touching
  `.env*`, `*.pem`, `*.key`, ssh and cloud credentials, `secrets.*` or
  `credentials.*` (plus `capture_deny_paths`) are never captured, nor are calls to
  the cognee tools or shell commands that mention cognee. `capture_redact` (on by
  default) scrubs private keys, bearer and basic tokens, DB URLs, credential
  assignments, vendor keys, AWS key ids, bcrypt hashes and secret-named fields from
  captured turns, traces and mirrored Hermes memory writes, plus
  `capture_redact_patterns`; redaction runs before truncation. A pattern the
  plugin cannot parse fails closed: nothing is captured until it is fixed, and the
  log says why. `capture=false` turns automatic capture off; `cognee_remember` is
  never filtered. Each setting falls back to its `COGNEE_CAPTURE_*` variable.
  Traces need the HTTP transport (the default); the in-process SDK transport keeps
  storing the turns only. A turn's writes share one `write_timeout` deadline and give the circuit
  breaker one verdict: a server that stops answering costs one timeout, not one per
  tool call, and the turn's unsent entries are queued for replay. Shared cases:
  `integrations/conformance/capture_policy_cases.json`.
- **Turns captured while the server is away are replayed.** A turn or trace that
  never reached the server (the connection could not be opened: refused while it
  boots, host unreachable, DNS failure; or a 503) is queued in
  `~/.cognee-plugin/hermes/pending-entries.json` (newest 500, kept a week) and
  replayed after the next successful write, when a session starts, and, for the
  ending session, before its session-end `improve()`. While the circuit breaker is
  open, writes go straight to the queue instead of being dropped. A write that
  timed out, was cut off mid-request, or failed otherwise is not replayed, since
  the server may already have stored it. The file is shared by every Hermes
  process and locked around each change, and a replay claims each entry before
  sending it, so two processes replaying at once never send the same one twice.
  A replay's budget bounds it as a whole (each send gets what is left of it), a
  session-end replay waits at most 5 s for one already running, and the replay
  after a successful write runs on its own thread instead of the turn's. Until now such writes were logged and
  lost, and every turn of a cold boot with them.
- **`cognee_recap` tool and `hermes cognee standup|digest|timeline`**, a port of
  the Claude Code and Codex recap skills (and OpenClaw's `memory_recap`): what was
  worked on and left open, the lessons distilled into memory, or a dated timeline
  of one topic, built from the sessions Cognee recorded and framed as data for the
  model to summarize. Sessions are grouped by the repository their edited files
  live in. A recap only covers sessions recorded into the active dataset, so on a
  shared server it never shows other users' conversations (`all_sessions` adds
  other clients' sessions in that dataset, never other datasets). The server
  cannot filter sessions by dataset, so a recap reads up to 2,500 of the newest
  sessions and says so when the server had more, or when `max_sessions` left some
  out. `recap_tool: false` (`COGNEE_RECAP_TOOL`) leaves the tool out.
- **Session guidance** is injected as `<cognee_guidance>`, from its own
  `scope=["session_context"]` request with the session id; see Changed.
- **`recall_passage_chars` / `recall_context_chars`** (defaults 2000 / 12000, or
  `COGNEE_RECALL_PASSAGE_CHARS` / `COGNEE_RECALL_CONTEXT_CHARS`; `0` disables)
  bound the injected recall; see Changed.
- **Slow-server probe.** After three prompts in a row whose recall only timed
  out, each prompt first spends a 3 s `/health` probe and skips recall while the
  server does not answer, instead of waiting out the whole budget. Any answered
  recall ends it.
- **`chunk_size` setting** (`COGNEE_CHUNK_SIZE`, [#428](https://github.com/topoteretes/cognee-integrations/issues/428)).
  Sent as `chunk_size` on `cognee_remember` writes, for embedding models whose
  input limit is below the server's default chunk size. Opt-in: unset sends
  nothing, and the server default applies as before.
- **The session is bridged into the graph before context compression.** Hermes
  compresses long conversations lossily and calls `on_pre_compress(messages)`
  first; the plugin did not implement it, so `improve()` ran only at session end
  and turns compressed away earlier could be lost before reaching the graph if
  the session never closed cleanly. `on_pre_compress` now starts the same
  session improve (same dataset and cognee session id, same remote/local
  transport) on a worker thread, after replaying the session's buffered writes.
  It is best-effort (Hermes pre-compress checkpoint API v1): compression never
  waits on it, a failure is logged as a warning and never raised, a compaction
  while the previous one's improve is still running does not start a second, and
  it contributes no text to the summary prompt. Gated like the session-end
  improve (`improve_on_end`, writes enabled, breaker closed); the session-end
  improve still runs. `on_pre_compress` is declared in both `plugin.yaml` files.
  Like the Claude Code plugin's, a pre-compress or session-end improve is skipped
  when nothing was stored for the session since its last improve, so a session
  that compacted and then ended pays for one improve, not two; the close still
  unregisters. Contributed by @rtheron (#487).
- **"What this plugin runs, stores and sends"** section in the README.

### Changed
- **Pinned cognee is 1.6.3** (was 1.5.4) in `pyproject.toml`, both `plugin.yaml`
  copies and `uv.lock`. The installed package doubles as the local server the
  plugin spawns, so a `pip install -U` upgrades that server too and its next boot
  runs the migrations of 1.6.0 through 1.6.3. 1.6.0 made `fastembed` and
  `onnxruntime` core dependencies, so the install is larger; 1.6.1 carries chunk
  metadata into hybrid retrieval and makes `dlt` a core dependency; 1.6.2 sizes
  chunks to the embedding model's input limit and explains a bare 401; 1.6.3 adds a
  capability layer for permissions and moves LanceDB to 0.39.
- **Recall injects the retrieved context only, not the conversation Hermes
  already has (SDK-741, SDK-904).** The per-prompt recall no longer fans out over
  the `session_memory`, `trace_lessons` and `agent_guidance` lanes. It makes one
  `scope=["graph"]`, `HYBRID_COMPLETION`, `only_context=True` request, sent
  without the session id: on cognee >= 1.6.0 the item's `text` is the full LLM
  input, and with a session id that input also carries this conversation's recent
  turns, which were most of a mid-session payload. The item is then cut down to
  its retrieved context: the question template goes, passages recorded in this
  session go, each passage is capped at `recall_passage_chars` (cut at a
  paragraph break), and entity lines that say nothing are pruned. Text that does
  not parse (an older server's bare context) is injected whole. The block is
  `<cognee_memory>` (was `<graph_memory>`); the item's `system_prompt` is never
  read. The session's guidance comes from its own `scope=["session_context"]`
  request, with the session id, against the active dataset, as
  `<cognee_guidance>`. `recall_context_chars` then bounds the whole injection
  (guidance first, then memory, then code facts; the block that crosses it is
  cut at a line and marked). The memory, guidance and code lanes are dispatched
  together under one deadline and are independent: a lane that fails or misses
  the deadline is dropped without costing the others, and a failed guidance
  request never feeds the breaker. The same rules as the Claude Code, Codex and
  OpenClaw plugins, shared through `integrations/conformance/recall_text_cases.json`.
  The hit header no longer carries a "beyond this session" count.
- **Recall reads the knowledge graph, the code graph and the session guidance
  only.** The raw session-cache scopes (`session`, `trace`) and the `auto` scope
  that folds them in are no longer requested anywhere. The explicit
  `cognee_recall` tool lost its `scope` argument (`auto` | `session` | `graph`; a
  caller still passing one is ignored) and sends `scope=["graph"]` with the
  dataset and the caller's `search_type` (or none, for the query classifier), and
  no session id, like OpenClaw's `memory_search`. The legacy single `auto`-scope
  prefetch and its `recall_session_layers` / `COGNEE_RECALL_LAYERS` toggle are
  gone. Both transports state the scope outright instead of letting the server
  default to `auto`, and the in-process SDK backend forwards `scope` and
  `only_context` to `cognee.recall`. Turns are still written and still promoted
  into the graph by `improve()`; they are just not searched as raw entries.
- **A turn is stored with its prompt as the question.** Over HTTP the session
  cache got each turn as one `User: … / Assistant: …` text in the `answer` field
  with an empty `question`, so recaps listed sessions with no prompts and a
  timeline could not find a topic in what was asked. Turns are now typed QA
  entries with the prompt and the answer in their own fields, as the other
  plugins store them. The in-process SDK transport keeps the framed text.
- **Captured turns are capped**: each side of a turn at 8,000 characters, tool
  arguments at 4,000 and tool results at 8,000, after redaction, as in OpenClaw.
- **The local server starts in `~/.cognee-plugin/hermes`, and reads its settings
  from `~/.cognee/.env`.** cognee >= 1.6.2 loads the first `.env` it finds walking
  up from its working directory and lets it override the process environment, so
  a `.env` in the directory Hermes was started from could replace the default
  user, data directories or auth settings the plugin passes. The server now runs
  from the plugin's state directory, and gets the values of `~/.cognee/.env` (or
  `COGNEE_ENV_FILE`) the way the Claude Code, Codex and OpenClaw plugins do: they
  act like exports, a variable already in the environment wins, and names such as
  `PATH` or `LD_*` are skipped. **If you kept `LLM_API_KEY` (or other server
  settings) in a `.env` in the directory you start Hermes from, move it to
  `~/.cognee/.env`** or to Hermes' own `.env`.

### Fixed
- **Dataset names are sanitized for cognee.** cognee rejects a dataset name
  containing a space or a dot; a configured `dataset` like `my project` now has
  spaces and dots replaced with `_` (with a warning), while every name the server
  already accepts stays exactly as it is. `cognee_switch_dataset` refuses such a
  name and suggests the sanitized form. Code-graph datasets for repositories with
  a dot in their name are now dot-free and can be indexed. Shared rule:
  `integrations/conformance/dataset_name_cases.json`. First implemented by
  @eiza763 (#226).
- **Permanent writes are no longer rejected as duplicate documents on cognee
  1.6.0.** Every permanent write uploaded its text as `memory.txt`, and 1.6.0
  introduced a document-identity rule — `add()` answers **HTTP 409** when a name
  already exists in the dataset with different content — so the first write to a
  dataset succeeded and every later one was refused and lost. The upload is now
  named after the content hash, which keeps a fresh name per text and keeps the
  server's documented no-op for identical content. Contributed by @firefloc
  (#431).
- **Recall items with no text are no longer injected as memories.** When a recall
  item carried no readable text (an empty completion, `text: null`, an error
  payload such as `{'error': 'Unauthorized', 'status': 401}`), hermes fell back to
  the item's `str()` and injected the whole response envelope into the prompt as
  a memory, untruncated, and counted it as a hit, so the hit numbers looked
  healthy while nothing useful arrived. Only string text counts now: empty and
  non-string slots are skipped in the usual key order (`answer`, `text`,
  `content`, `chunk_text`, `summary`), items with nothing readable are dropped
  from the per-prompt block and the `cognee_recall` tool, and an unconvertible
  object no longer turns into its repr. Reported by @Diddykonga (#457).
- **Fresh installs against cognee 1.6.0 could not mint their owner API key
  (SDK-740).** cognee 1.6.0 stopped baking `default_password` into the default user:
  the server creates that user at startup only when `DEFAULT_USER_PASSWORD` is set,
  sets the password once, and never rewrites a stored one, so the login the key
  bootstrap relies on answered `400 "This user does not have a password"`. The server
  this plugin spawns is now started with `DEFAULT_USER_EMAIL=default_user@example.com`
  and `DEFAULT_USER_PASSWORD=default_password` (the literals every Cognee plugin
  shares, since they all share one server and one database), `setdefault` so an
  operator's own `DEFAULT_USER_*` export wins. `COGNEE_USER_EMAIL`/`COGNEE_USER_PASSWORD`
  still select which user the plugin logs in as; a non-default user must already
  exist. Against a server the plugin did not start, a rejected login now logs a
  warning that says what to do (start the server with `DEFAULT_USER_PASSWORD`
  matching `COGNEE_USER_PASSWORD`, or set `COGNEE_API_KEY`), and the first `401`
  that follows carries the same hint. The README documents it.
- **`cognee_remember` no longer reports a timeout for a write that succeeded.**
  Permanent writes went to `/api/v1/remember` without `run_in_background`, so the
  server ran the whole cognify (several LLM calls per document) before answering;
  on a slow or rate-limited LLM that exceeded `COGNEE_WRITE_TIMEOUT` and the agent
  saw `Cognee remember failed: ... timed out` for a write the server completed.
  The write now runs in the background and returns `status: "running"`, the fix
  the Claude Code plugin got in #96. Contributed by @jstevenjacobs (#479).
- **Session-end improve says why on an exhausted LLM budget.** cognee 1.6.3
  answers 402 when the server's LLM budget runs out. The exit watcher and the
  in-process close now log that the budget is exhausted and that nothing will
  retry, and the in-process close no longer counts it as a circuit-breaker
  failure (the server answered).

## [1.2.2]

### Added
- **Catalog installation support.** CLI status and version commands recognize
  Hermes' catalog marker, report the running plugin version, and direct updates
  through `hermes plugins update cognee`. Catalog copies skip PyPI checks even
  with `--check-updates`, and `cognee-hermes-install` refuses to overwrite them,
  including copies with damaged catalog metadata. Installation and setup docs
  now distinguish catalog and pip workflows; catalog-name installation is marked
  as available after acceptance.

### Changed
- **The per-prompt recall prefetch dispatches every lane at once.** The layered
  lanes (`session`, `trace`, `session_context`, `graph`, plus `code` when armed)
  ran one after another, cheap lanes first, so every cheap lane was a full round
  trip on top of the graph search and an armed code lane could burn seconds
  before graph even started. All lanes are now in flight together under one
  shared deadline — `recall_timeout` clamped to `recall_budget` — and the
  prefetch costs the slowest lane, not the sum. The rendered blocks keep their
  canonical order whichever lane answers first; a failing lane still never
  discards the others, and the breaker still sees one verdict per turn.
- **cognee pinned to exactly 1.5.4** (`pyproject.toml`, `plugin.yaml`), up from 1.5.3.
  The floor moves because code-graph indexing only reads `raw_data` from 1.5.4 on;
  `content_type="code"`, the `code` recall scope and targeted session invalidation on
  document delete still date from 1.5.3. The installed package doubles as the local
  server this plugin spawns, so the pin and the wire contract have to move together.
- **Connections identify Hermes explicitly.** Agent registration now sends
  `type: hermes_agent` instead of the generic `type: api`, so the Cognee server
  can distinguish Hermes connections from other API clients.
- **Package metadata declares Apache-2.0.** `pyproject.toml` now includes the
  license identifier.
- **The README links Hermes' installation guide** instead of inlining its
  `curl … | bash` one-liner. The catalog's install scanner flags a piped shell
  script even inside a README, and `hermes plugins validate` now reports
  `security scan — safe` with no warnings.
- **The README makes the existing Python 3.10+ requirement explicit**, including
  guidance for macOS users whose Xcode Command Line Tools provide Python 3.9.
  The minimum supported Python version has not changed.

### Fixed
- **Repo indexing submitted the repository under a field the server had stopped
  reading, so every index 400'd ([#420](https://github.com/topoteretes/cognee-integrations/issues/420)).**
  cognee 1.5.4 renamed the form field that carries the repository spec on
  `POST /api/v1/remember` with `content_type=code` from `repositories` to
  `raw_data`. `hermes cognee index-repo` still sent the old name, and an unrecognised multipart part is
  dropped by the server rather than refused — so each request arrived naming no
  repository at all and came back `HTTP 400: content_type='code' requires at least
  one repository path or git URL in 'raw_data'`. Local paths and git URLs failed
  alike. The spec now goes in `raw_data`.
- **A 400 the server could explain was reported as "your server is too old".** The
  error branch treated any 400 whose body mentioned `content_type` as a server
  predating `content_type='code'`. Every 400 the server's code branch raises names
  that field — including the one above — so the actionable message was overwritten
  with advice to upgrade a deployment that was already new enough, and the reporter
  of #420 spent the session chasing the wrong problem. Only the server's own
  "Unsupported content_type" wording counts as a version problem now; every other
  400 is passed through verbatim.

## [1.2.1]

Fixes [#382](https://github.com/topoteretes/cognee-integrations/issues/382):
**every pip install of 1.0.0–1.2.0 produced a plugin Hermes silently ignored.**
`pip install` + `cognee-hermes-install` reported success, `hermes plugins show
cognee` said "enabled", but the memory provider — recall, remember, forget,
the session-end bridge — was never wired up; `hermes memory status` ("Plugin:
NOT installed") was the only symptom. Installs copied from a repo checkout
(the path the plugin was developed and live-tested through) were unaffected,
which is how this shipped three times.

### Fixed

- **`cognee-hermes-install` now lays down the plugin root's `__init__.py`.**
  Hermes' memory-provider loader (`plugins.memory._is_memory_provider_dir`)
  skips — silently — any plugin directory without an `__init__.py` naming
  `MemoryProvider` or `register_memory_provider`. The repo always had the
  right file (it is what made checkout installs work); the installer's file
  list simply never included it. It ships in the wheel as
  `_plugin_root/plugin_init.py` (an `__init__.py` inside `_plugin_root/`
  would make it an importable subpackage) and is installed under its real
  name. After `pip install -U`, re-run `cognee-hermes-install` — that now
  also repairs an existing broken install in place.
- **The pip entry point is now in the group Hermes actually reads.** The
  memory loader scans `hermes_agent.memory_providers`; the package declared
  only the generic `hermes_agent.plugins` group (kept for compatibility),
  which is why Hermes reported the plugin "enabled" with nothing registered.
  With the correct group, a plain `pip install` activates the provider via
  `register(ctx)` even before `cognee-hermes-install` runs — the installer
  remains recommended (the directory install carries the `hermes cognee` CLI
  and the dashboard config panel at full fidelity).
- Regression tests: the installer's output directory must pass Hermes'
  discovery heuristic verbatim, and the pyproject must declare the
  `hermes_agent.memory_providers` entry point.

### Corrections to earlier notes

- 1.0.0's known-limitations claim that "Hermes discovers memory providers by
  directory scan only" was wrong: Hermes has always had a pip entry-point
  path for memory providers — this package was registered in the wrong group.

## [1.2.0]

Parity release: brings the Hermes plugin up to the claude-code/codex
integrations (and the OpenClaw 2026.8.27 parity release) on every user-facing
memory operation, expressed the way Hermes consumes capabilities — as memory
provider tools plus `hermes cognee` CLI commands.

### Added

- **Recall session layers.** The per-prompt recall used a single
  `scope: auto` call, and with `datasets` + an explicit `search_type` in the
  request the server resolves that to graph-only — so cached Q&A turns,
  tool-call trace lessons and distilled agent guidance never reached the
  prompt. Recall now fans out one bounded call per scope, cheap lanes first
  (`session → trace → session_context → graph`, the graph lane on
  `HYBRID_COMPLETION` with `only_context`), each rendered as its own block
  (`<session_memory>`, `<trace_lessons>`, `<agent_guidance>`,
  `<graph_memory>`), under a shared budget (`COGNEE_RECALL_BUDGET`, 20s) so a
  slow graph search can never starve the cheap lanes. A failure in one lane
  never discards the others; a 404 on the graph lane (dataset not yet
  cognified — routine on a fresh install) is benign, not a breaker failure.
  The `cognee_recall` tool's `scope=session` now covers all three session
  layers. Opt out with `COGNEE_RECALL_LAYERS=false`.
- **Per-document forget.** `cognee_forget` is now two-phase and user-directed
  ("forget what we said about tennis"): `action=find` lists candidate
  documents with raw-text previews and matched terms; `action=forget` deletes
  only the listed `data_ids`, one `POST /forget` (`datasetId`/`dataId`) each,
  and only with `confirm=true`. Ids are validated as UUIDs before they touch a
  request; whole-dataset deletion requires an explicit
  `everything_in_dataset=true`, and the all-datasets wipe is not expressible
  from the tool by construction. Targeted session-cache invalidation on delete
  needs cognee >= 1.5.3.
- **`cognee_switch_dataset` tool** — move one conversation to another dataset:
  `list` / `current` / `switch` / `reset`. A switch flushes the pending turn,
  bridges the current session into its old dataset (`improve` on the server's
  background pipeline — strict: it aborts on failure unless `force=true`, in
  which case the un-bridged session is recorded and re-submitted at session
  end), ensures the target, then re-points capture, recall and the session-end
  improve (and the crash-safe exit watcher) under a fresh cognee session id
  (`hermes_<id>__N`). Overrides persist per conversation in
  `~/.cognee-plugin/hermes/dataset-overrides.json` and are re-applied when the
  same Hermes session initializes again. `COGNEE_DATASET_SWITCH_TOOL=false`
  removes the tool.
- **Code graph.** `hermes cognee index-repo <path|url> [--dataset]
  [--index-vectors] [--wait <s>]` indexes a repository into cognee's
  deterministic code graph (enola pipeline, no LLM/embedding calls, one
  `codebase-<repo>-<digest>` dataset per repo — the path digest keeps
  same-basename checkouts from sharing a graph); `cognee_code_search` answers
  structural questions exactly (`query_facts`, `explore`, `traverse`,
  `find_path`, `impact_analysis`, `delta`); an additive `code` recall lane
  fires only when a prompt names an identifier-shaped token AND the launch
  directory sits inside an indexed repo (or `COGNEE_CODE_DATASETS` names one).
  Autoindex and per-turn re-ingest are deliberately not ported — Hermes is
  rarely launched inside a checkout (the OpenClaw plugin made the same call).
  Indexed repos are recorded under `~/.cognee-plugin/hermes/code-graph/`.
  Requires cognee >= 1.5.3. `COGNEE_CODE_SEARCH_TOOL` / `COGNEE_CODE_GRAPH_RECALL`
  opt out.
- **Memory steer.** The system-prompt block now asserts Cognee as the
  preferred, authoritative long-term memory over Hermes' built-in memory — the
  counterpart of claude-code's `COGNEE_PREFER_MEMORY` and openclaw's
  `memorySteer`. `COGNEE_MEMORY_STEER=false` disables it,
  `COGNEE_MEMORY_STEER_TEXT` replaces the wording.
- **Memory-hit visibility.** The injected recall block opens with plain words
  on what memory just did — `4 memory hits this turn (2 beyond this session) ·
  3/7 turns had hits this session` — with per-session totals that reset when a
  conversation resets. `COGNEE_MEMORY_HITS=false` hides it.
- **Version display + update check.** `hermes cognee version`, and both it and
  `hermes cognee status` now carry an "update available" hint from a
  rate-limited, fail-silent PyPI check (cached in
  `~/.cognee-plugin/hermes/update-check.json`; `COGNEE_UPDATE_CHECK`,
  `COGNEE_UPDATE_CHECK_INTERVAL`, `--check-updates` forces a live check).
  CLI-only — the session path never checks. The nudge stays two-step on
  purpose: `pip install -U` then `cognee-hermes-install`, because Hermes runs
  the installed copy.

### Changed

- **Breaking — cognee is now pinned to exactly 1.5.3** (`cognee==1.5.3`; the
  repo's pin checker accepts exact pins as bounding both sides). The installed
  package doubles as the local server this plugin spawns, and this release's
  features set a hard floor there: `content_type="code"` repo indexing, the
  `code` recall scope, and targeted session invalidation on document delete
  all first shipped in 1.5.3, so a 1.4.x resolve would silently strip the code
  graph and let a later sync re-persist forgotten content. The claude-code,
  codex and openclaw plugins pin their server to exactly 1.5.3 for the same
  reason; a *remote* server (`COGNEE_BASE_URL`) is unaffected by the pin but
  needs >= 1.5.3 for the same features, and an older one now gets clear
  errors on the code paths instead of quiet degradation.
- The HTTP transport grew the endpoints the new features stand on:
  `GET /datasets`, `GET /datasets/{id}/data`, `GET /datasets/{id}/data/{id}/raw`,
  single-document `POST /forget`, code indexing via `POST /remember`
  (`content_type=code`), `GET /datasets/status`, and `recall` now carries
  `scope` lists, `context_profile`, `code_query` and `only_context`. The SDK
  transport reports these operations as HTTP-only instead of failing obscurely.

### Known limitations

- Turns from the live conversation become deletable documents only after the
  session is bridged to the graph; `cognee_forget` says so in its envelope.
- The dataset listing in `cognee_switch_dataset` cannot tell read-only
  datasets apart client-side; a switch to one fails at the ensure step with a
  clear error instead.

## [1.1.0]

Hardens the plugin against silently corrupted search indexes with local Ollama
embedding models (live-diagnosed on a meeting-notes ingestion: the pipeline
reported success on every document while all retrieval timed out). cognee sizes
its chunks from `EMBEDDING_MAX_COMPLETION_TOKENS`, whose 8191 default is far
above any local embedding model's context; every substantial text overflowed
Ollama, and cognee mean-pooled the pieces into lossy vectors while still
reporting success — the only trace an `Ollama embedding error` line in the
server log.

### Added

- **Safe embedding defaults at server spawn.** With `EMBEDDING_PROVIDER=ollama`,
  the spawned server (and embedded mode) now gets a context-matched
  `EMBEDDING_MAX_COMPLETION_TOKENS` for recognized models (`all-minilm`,
  `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, and others — a conservative
  512 for anything unrecognized) plus the matching `HUGGINGFACE_TOKENIZER`, so
  cognee chunks within the model's real limits. Explicit values always win; no
  effect on other providers.
- **Overflow surfacing.** The HTTP transport tails the spawned server's log
  between calls; a fresh embedding-context overflow now lands in the tool
  envelope — a `warning` on results/successful writes, an `error` on an empty
  recall — naming the env levers and the recovery runbook, instead of degrading
  silently.
- **Timeout advice.** A timed-out recall error now says why it is likely slow
  (GRAPH_COMPLETION runs an LLM per query) and what to do (`search_type=CHUNKS`,
  `scope=session`, or `COGNEE_RECALL_TIMEOUT`); the `search_type` tool schema
  explains the CHUNKS/GRAPH_COMPLETION trade-off so the model can self-route.
- **Docs.** `RUNBOOK.md` (pause ingestion → fix env → restart server → rebuild
  dataset → verify → resume), `.env.example`, a README section on Ollama
  embedding settings, and README rows for the previously undocumented
  `COGNEE_RECALL_TIMEOUT` / `COGNEE_WRITE_TIMEOUT` / `COGNEE_IMPROVE_TIMEOUT`.

## [1.0.0]

First stable release, published to PyPI. Moves the provider onto cognee's REST
API, the way the Claude Code, Codex and OpenClaw plugins already talk to it. The local cognee server was always there —
what changed is the client: routing through the SDK's `cognee.serve()` /
`CloudClient` silently dropped fields the server accepts.

### Changed

- **Breaking — the transport is now direct HTTP.** Requests are built and sent by
  the plugin (stdlib `urllib` only) instead of going through `cognee.serve()`.
  `COGNEE_TRANSPORT=sdk` restores the old client; `COGNEE_EMBEDDED=true` still uses
  the in-process SDK, which is the only transport that can run without a server.
- **Breaking — the local server moved from port 8000 to 8011.** This matches the
  other cognee agent plugins and leaves cognee's own default of 8000 to servers you
  run yourself, so the plugin no longer attaches to one by accident. Memory is
  unaffected — where it lives depends on the server's data directory, never the
  port. Set `COGNEE_LOCAL_PORT=8000` to keep the old behaviour, and stop any old
  plugin-started server on 8000 so two servers do not share one data directory.
- **Breaking — Hermes joins the shared brain.** The default dataset is now
  `agent_sessions` and local storage defaults to `~/.cognee/{data,system}` — the
  exact names and roots the Claude Code, Codex and OpenClaw plugins pin — so
  memory is shared across all cognee agent plugins on the machine, no matter
  which of them booted the server. `COGNEE_PLUGIN_DATASET` (the name the other
  plugins read) is the canonical dataset variable; `COGNEE_DATASET` stays as a
  lower-precedence alias. To keep a Hermes profile apart, set its own dataset, or
  its own `COGNEE_DATA_ROOT` / `COGNEE_SYSTEM_ROOT` *plus* `COGNEE_LOCAL_PORT`
  for full isolation. Old 0.1.x memory (dataset `hermes`, cognee's default
  roots) is not deleted but is no longer found by default —
  `COGNEE_DATASET=hermes` restores it. Previously roots were only set in embedded
  mode, so the spawned server fell back to cognee's global default location.
- **The minted API key is shared too.** It now lives at
  `~/.cognee-plugin/api_key.json` in the other plugins' exact format (previously
  `$HERMES_HOME/cognee-api-key.json`): whichever plugin mints first, the rest
  reuse. The spawned server's environment also mirrors the other plugins'
  bootstraps (`CACHE_ROOT_DIRECTORY`, `LLM_INSTRUCTOR_MODE=json_schema_mode`,
  `COGNEE_IMPROVE_SUBMIT_TIMEOUT=420`), and its log moved to
  `~/.cognee-plugin/hermes/server.log`, so the server behaves identically no
  matter which plugin starts it.

### Fixed

- **Session memory now reaches the permanent graph.** `improve()` is called with
  `session_ids`, which `CloudClient.improve` never forwarded — the session-to-graph
  bridge had silently become a dataset-wide improve.
- **Recall can see the session cache.** Over HTTP, `search_type` is now always
  sent — as an explicit `null` when auto-routing — and the scope travels by name
  in cognee's own `scope` field. `/api/v1/recall` defaults a *missing*
  `search_type` to `GRAPH_COMPLETION`, and cognee folds the session cache into an
  `auto` scope only while the search type is null, so leaving the key out
  resolved every scope — `session`, `auto` and `graph` alike — to graph-only.
  Turns were being written to a cache that nothing could read until `improve()`
  promoted them at session end.
- **`COGNEE_AUTO_ROUTE` is honoured over HTTP.** Same root cause: with the key
  omitted, `true` and `false` produced byte-identical requests and the query
  classifier never ran.
- **The session-end improve is no longer killed by the server it runs on, and no
  longer delays your exit.** Closing a session is now handed to the detached
  worker that already covered crashes: it runs `improve()` to completion and only
  then unregisters, while Hermes exits immediately. Previously the improve was
  submitted in the background and `shutdown()` unregistered straight after, which
  drops the agent count to zero — and `COGNEE_AGENT_MODE`'s watchdog SIGTERMs the
  server within 60s of that, with no regard for running pipelines, so a graph
  build (routinely longer than 60s) was liable to be cut in half. Doing it
  synchronously in-process would have fixed the ordering by making the user wait,
  so it happens out of process instead, matching how the claude-code, codex and
  openclaw plugins close a session. Both paths claim a once-marker, so a session
  is closed exactly once. `COGNEE_IMPROVE_BACKGROUND` now opts out of the handoff
  and keeps the work in-process.
- **An unreachable `COGNEE_BASE_URL` now fails at startup.** `cognee.serve()` logged
  a warning and returned a client regardless, so a bad URL only surfaced on the
  first real call.
- **A remote `COGNEE_BASE_URL` without `COGNEE_API_KEY` also fails at startup.**
  Key minting is local-only — remote servers (Cognee Cloud included) expose no
  login route to mint from, and the claude-code/openclaw plugins require a key
  for remote targets for the same reason. Previously the plugin continued
  unauthenticated and every call 401'd one at a time; it also no longer sends
  the default-user login credentials to non-local hosts.
- **The spawned server no longer leaks.** The plugin registers an agent connection
  on connect and unregisters on shutdown, so cognee's idle watchdog can stop a
  server nobody is using.
- **The exit watcher is safe on Windows.** Its liveness probe no longer uses
  `os.kill(pid, 0)` there — CPython implements non-console "signals" on Windows
  as `TerminateProcess`, so probing would have killed the running Hermes — and
  queries the process handle (`OpenProcess`/`GetExitCodeProcess`) instead.
  Group/broadcast pids (`<= 0`) are also never treated as watchable.
- **A crashed Hermes no longer loses its session or strands the server.** In
  server/remote mode the plugin arms a small detached exit watcher (the pattern
  the claude-code/codex/openclaw plugins use) that polls the Hermes PID; on an
  unclean death it runs `improve(session_ids=[...])` synchronously — improve
  *before* unregister, so the idle watchdog cannot tear the server down
  mid-promotion — then unregisters the agent connection. A clean shutdown
  disarms it first, so nothing double-fires; a session end that already bridged
  stands down only the improve half. State and log live under
  `~/.cognee-plugin/hermes/`. Embedded mode gets no watcher — there is no server
  to outlive the process.
- **Local servers are authenticated.** An owner API key is minted once and cached
  at `~/.cognee-plugin/api_key.json`, instead of relying on the server having
  auth disabled.
- **A failed initialization now fails closed.** Hermes logs a provider's
  initialization error and starts anyway, so every entry point now refuses to run
  rather than operating an unconnected transport — which previously meant quietly
  falling back to in-process cognee and its single-writer databases.
- **No more cross-conversation context leak on `/reset`.** A recall still in flight
  for the previous conversation is discarded instead of landing in the fresh one.
- **Write gating is uniform.** `on_memory_write` now honours `agent_context`, so a
  subagent's built-in memory write is suppressed like its conversation turns.
- **`hermes backup` sees cognee storage.** `backup_paths()` reports roots pointed
  outside `HERMES_HOME`.
- **`COGNEE_AUTO_ROUTE=false` is honoured over HTTP**, translated to an explicit
  `GRAPH_COMPLETION` search type since the endpoint has no `auto_route` field.
- **Session turns actually reach the session cache.** They are stored as a typed
  QA entry via `/api/v1/remember/entry`, the endpoint the Claude Code plugin uses
  for the same purpose. Sending them as a document to `/api/v1/remember` with a
  `session_id` looked like it worked — the API answered
  `status: "session_stored"` — but the payload arrives as a multipart file, which
  cognee coerces to a `[UploadFile]` placeholder and deliberately *skips* for the
  session cache. Every turn was silently discarded and `improve()` had nothing to
  promote into the graph. Found by the live round-trip test; no mock could catch it,
  because the server reported success.
- The spawned server is also started with `CACHING=true` and `AUTO_FEEDBACK=true`,
  matching the Claude Code and OpenClaw plugins. Both are already cognee's defaults,
  so this is insurance against a future default change silently disabling the session
  tier, not a fix in itself.
- **A cold first boot no longer times out.** `COGNEE_SERVER_BOOT_TIMEOUT` now
  defaults to 600s (was 30s), matching the other plugins' boot deadline — a first
  boot runs DB migrations and can take minutes, and giving up early left memory
  off for the whole session. The long deadline is safe because the bootstrap now
  **fails fast when the spawned server dies**: a crashed child with nothing
  listening on the port raises immediately (pointing at the server log) instead
  of stalling the session start, while a child that merely lost the port-bind
  race to a concurrent starter still waits for the winner to become healthy.
- **cognee is now bounded to `>=1.2.1,<=1.4.0`** (was the loose
  `>=1.0.0,<2.0.0`). The wire contract this plugin depends on —
  `/api/v1/remember/entry` and `improve(session_ids)` — first shipped in 1.2.1,
  and a 1.0.x server would accept some of these requests and silently do the
  wrong thing. The 1.4.0 cap is the newest version verified against the full
  test suite and a live server boot (health, key minting, agent
  register/unregister); the lockfile resolves to 1.4.0.

### Added

- **pip is now a working install channel.** The wheel ships the plugin-root
  files (`plugin.yaml`, the `cli.py` shim, `after-install.md`) as package data,
  and a new `cognee-hermes-install` console script copies the plugin into
  `$HERMES_HOME/plugins/cognee` — the only place Hermes' directory scan looks.
  `hermes cognee status` now shows the installed plugin version and warns when
  the pip package is newer than the installed copy (after `pip install -U`,
  re-run `cognee-hermes-install`). Releases publish to PyPI from CI on
  `hermes-agent-v*` tags.
- **The dataset is ensured at startup** (idempotent `POST /api/v1/datasets`, the
  other plugins' bootstrap call), so a session that opens with a recall on a
  fresh server or cloud tenant no longer hits a missing dataset. Best-effort:
  writes create the dataset implicitly anyway.
- `COGNEE_TRANSPORT` — `http` (default) or `sdk`.
- A test suite grown from 34 to 270, including wire-format tests per transport and
  an opt-in live round trip against a real cognee server
  (`COGNEE_RUN_INTEGRATION=1`).

### Known limitations

- A *permanent* write cannot be linked to its originating session over HTTP:
  `/api/v1/remember` has no `session_ids` field. The session-to-graph bridge is
  unaffected. Logged once per session rather than dropped silently.
- `pip install` alone cannot activate the provider — Hermes discovers memory
  providers by directory scan only — which is why `cognee-hermes-install`
  exists, and why it must be re-run after every `pip install -U`.

## [0.1.0]

Initial standalone plugin: session-aware graph memory with recall, remember,
forget, and a session-end improve, over the cognee SDK.
