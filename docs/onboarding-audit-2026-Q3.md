# Onboarding audit, 2026-Q3

Written for issue [#3605](https://github.com/topoteretes/cognee/issues/3605). Part 1 of that ticket: walk the real first-run paths as a new user, capture the friction verbatim, and rank it. The proposal (Part 2) and quick-win implementation (Part 3) build on this document and land in the same PR series.

## Setup

- Auditor: Labi-Joy
- Date: 2026-07-09
- Baseline: Ubuntu 22.04, Python 3.10.12 via `uv`, no prior cognee config on disk, no `LLM_API_KEY` in the environment at `t = 0`
- Cognee version: `1.2.2-local` on `dev` at `89eaa725b`
- Entry points walked: Python SDK, `cognee-cli`, `cognee-cli -ui`, `docker compose up`

## Executive summary

Six findings ranked by user-time-cost times frequency. Two more (F7, F8) are called out for cross-reference to sibling issues.

| Rank | Finding | Impact |
| ---- | ------- | ------ |
| F1 | Invalid `LLM_API_KEY` retries with exponential backoff for ~2 minutes before erroring | **Critical**. Every new user with a typo, quota-exhausted key, or wrong provider hits this. Two minutes of ambiguous log spam before a 401 is finally surfaced. |
| F2 | Errors leak upstream implementation details (`Status code: 422`, OpenAI 401 body, aiohttp `Unclosed client session`) | **High**. Users cannot act on messages that reference HTTP status codes and library internals when the CLI is not itself an HTTP client. |
| F3 | Primary CLI commands emit no "what now?" hint after success | **High**. A user who runs `remember` sees a success line and no pointer to `recall`. Every command is a dead end. |
| F4 | Excellent Docker preflight exists in `-ui`, none in the primary commands | **Medium**. Parity gap. The `-ui` path fails calmly with actionable fixes; every other command charges into the same failure and blames the log. |
| F5 | No offline sample-data path, so newcomers commit real tokens on first successful run | **Medium**. The quickstart promises a fast result, but that result costs money the user did not agree to. |
| F6 | The four common first-run failure modes have no unified preflight command | **Medium**. Same signals are checked in different places; no `cognee doctor`-style single command to answer "is my setup right?". |
| F7 | `session_lifecycle/usage_tracking.py` reports `chars/4` as the token estimate | Cross-ref: issue [#3606](https://github.com/topoteretes/cognee/issues/3606) covers accurate token capture. Noted here so the audit points at it. |
| F8 | `_PRICING_PER_M_TOKENS` is "conservative and incomplete" per its own inline note | Cross-ref: same as F7. |

## F1. Invalid `LLM_API_KEY` retries for ~2 minutes before erroring

**Reproducer:**

```bash
export LLM_API_KEY=sk-fake-just-to-pass-the-key-check
cognee-cli recall "what did we talk about?"
```

**Observed:** four exponential-backoff retries at 8.66s, 16.1s, 32.6s, 64.7s. Total wall-clock before the final error is roughly two minutes. Each retry re-logs the full error trace and a warning.

**Root cause:** `LiteLLMEmbeddingEngine.embed_text` is wrapped in `tenacity` retry logic that treats a 401 authentication error the same as a transient 5xx. Authentication errors are terminal for the current process; retrying them wastes user time and confuses the log.

**Verbatim tail of the failure output** (trimmed for length):

```
2026-07-09 21:14:15 [error] Error embedding text: litellm.AuthenticationError:
    AuthenticationError: OpenAIException - Error code: 401 - {'error':
    {'message': 'Incorrect API key provided: sk-fake-****heck.
    You can find your API key at https://platform.openai.com/account/api-keys.',
    'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None},
    'status': 401}. EMBEDDING_ENDPOINT='None'.
2026-07-09 21:14:15 [warning] Retrying LiteLLMEmbeddingEngine.embed_text in 16.1s ...
2026-07-09 21:14:32 [error] Error embedding text: ... (same again)
2026-07-09 21:14:32 [warning] Retrying LiteLLMEmbeddingEngine.embed_text in 32.6s ...
... two more retries at 32.6s and 64.7s ...
Error: Failed to recall: litellm.AuthenticationError: OpenAIException -
    Incorrect API key provided: sk-fake-****heck.
Note: Please refer to our docs at 'https://docs.cognee.ai' for further assistance.
```

**What a user needs instead:** on authentication errors specifically, no retry. Fail fast with a one-line message that names the problem and one action:

```
Cognee could not authenticate with the LLM provider.
  Your LLM_API_KEY is set but the provider rejected it (401).
  Check that the key is valid and has not expired at your provider's console,
  then retry.
```

## F2. Errors leak upstream implementation details

Three specific instances observed across the two failure paths tested.

**F2a. `Status code: 422` in CLI output when the CLI is not an HTTP client.** The number is a leaked HTTP status from an internal HTTP error class. To a user, a CLI reporting HTTP status codes is confusing at best.

Sample:

```
Error: Failed to remember: LLMAPIKeyNotSetError: LLM API key is not set. (Status code: 422)
```

**F2b. OpenAI-shaped 401 body in the recall error message.** When the LLM provider rejects the key, cognee surfaces the provider's raw JSON body verbatim, including the masked key hint and the URL to the OpenAI account page. This is a useful upstream detail for a developer debugging the provider, but a user driving cognee is not the target audience for that message.

**F2c. Async cleanup warnings shown to the user.** After every LLM failure the CLI prints:

```
Unclosed client session
client_session: <aiohttp.client.ClientSession object at 0x...>

Unclosed connector
connections: ['deque([(<aiohttp.client_proto.ResponseHandler object at 0x...>, ...)])']
connector: <aiohttp.connector.TCPConnector object at 0x...>
```

That is a Python `warnings.warn` from `aiohttp` at process teardown. It should be captured or suppressed at the CLI boundary rather than piped to the user's terminal.

## F3. Primary CLI commands emit no "what now?" hint after success

Observed by running `cognee-cli remember "cognee turns documents into memory"` with a valid `LLM_API_KEY`. Output ends with:

```
Data ingested and knowledge graph built successfully!
  Dataset ID: 8f...
  Items processed: 1
  Content hash: a3...
  Elapsed: 4.2s
```

Then the shell prompt. The user has no pointer to the next command in the flow (`recall`), no hint that a session id would let them scope queries, and no confirmation that they can now query the graph. The four primary commands have this same shape: success, some metadata, no continuation.

The dropped-thread pattern shows up in the code at `cognee/cli/commands/remember_command.py` around L110-L120 and mirrors in `recall_command.py`, `cognify_command.py`, and `forget_command.py`. All four write metadata via `fmt.echo` and stop.

**What a user needs instead:** a single hint line at the end of each success block that names the next natural command with a copy-paste example.

Sketch:

```
Data ingested and knowledge graph built successfully!
  Dataset ID: 8f...
  Items processed: 1
  Content hash: a3...
  Elapsed: 4.2s

Next: cognee-cli recall "your question" -d main_dataset
```

## F4. `-ui` has excellent Docker preflight, primary commands have none

`cognee/api/v1/ui/ui.py` at `_check_docker_available()` (L24-L76) is a model of the pattern. It checks for the docker binary, runs `docker info` with a 15s timeout, distinguishes "no CLI installed", "CLI installed but daemon down", and "call timed out", and emits a message that names three concrete fixes per case (Docker Desktop, Colima, `sudo systemctl start docker`).

The gap is that the same care does not exist in `cognee-cli remember`, `recall`, `cognify`, `forget`. Those commands go straight to the LLM client and fail with the messages surfaced in F1 and F2.

The `-ui` preflight is the north star. Extract the pattern into a shared `cognee doctor`-style module that the primary commands and a new preflight subcommand both call.

## F5. No offline sample-data path, so newcomers commit real tokens on first successful run

The README's Quickstart at L105-L182 walks the user from `uv pip install cognee` to `await cognee.remember(...)`. The first successful `remember` runs `cognify`, which calls the LLM to extract entities, then generates embeddings, then persists the graph. All three cost tokens. Nothing in the flow warns the user of this or offers a way to see a result without spending.

Two consequences:

- Users who try cognee "just to see what it does" pay for it. Small dollar amount, but real, and unexpected.
- Users without an OpenAI account cannot try cognee at all without signing up for a paid provider first.

**What a user needs instead:** two things, landing in stages. First (shipped here) a `--sample-data` flag that ingests a bundled fixture so a newcomer reaches a `recall` without composing input. Second (a follow-up, gated on the #3601 mocked harness) a fully offline variant that runs the cycle against a stubbed LLM with no API calls, so a user can see a result before committing a key at all.

## F6. No unified `cognee doctor` preflight

Multiple concerns overlap in F1, F2, F4: is Python the right version, is `LLM_API_KEY` set, does the LLM provider actually accept it, is Docker running (only for `-ui`), is the file-based storage writable. Every command checks a subset in-line and reports failures differently.

A single `cognee doctor` command that runs every preflight and reports a bullet list of pass/fail with the same "actionable message" contract used by `-ui` (see F4) is the smallest change that consolidates the pattern.

## F7 and F8. Cross-references

These findings are surfaced by inspection but their scope belongs to issue [#3606](https://github.com/topoteretes/cognee/issues/3606) (Expansion Signals + Usage Metering). Noted here so this audit is honest about them and so the two issues stay coordinated.

- F7: `session_lifecycle/usage_tracking.py` uses `chars / 4` as the token count, with an inline comment acknowledging the estimate. The first-run user cannot see accurate cost.
- F8: `_PRICING_PER_M_TOKENS` is described in its own docstring as "conservative and incomplete". "What did this cost me?" has no honest answer today.

## Measurements

**Time-to-first-recall on the SDK path**, valid `LLM_API_KEY`:

- Cold start (import + migrations + first LLM call for cognify): observed ~28s on a warm venv, first ever `remember` of a one-sentence input
- Second `recall` in the same process: ~2s

The dominant cost on first `remember` is the LLM call inside cognify to extract entities. Not addressable in this audit; noted for expectation-setting in the proposal.

**Decision count before step 1 by entry point:**

| Entry point | Decisions to make before running command 1 |
| ----------- | ------------------------------------------ |
| Python SDK  | 2: which install tool (pip / uv / poetry), where to put `LLM_API_KEY` (env var or `.env`) |
| `cognee-cli` | Same 2 |
| `cognee-cli -ui` | Same 2 plus install Docker / Colima if absent |
| `docker compose up` | Same 2 plus decide which profiles to enable (`ui`, `mcp`, `postgres`, `neo4j`) |

The `-ui` and `docker compose` paths compound the decision cost. The SDK and CLI paths are already close to minimum-viable.

**Verbatim error text for the failure modes tested:**

Reproducers and full traces are inlined under F1 and F2. Two additional modes were not walked to a fresh reproducer because F1 and F2 already establish the shape:

- Wrong Python version: `pip install cognee` fails on `requires-python = ">=3.10,<3.15"` before any cognee code runs. The pip error text names the version constraint. No cognee-specific message; not obviously broken.
- No Docker for `-ui`: the `_check_docker_available()` path already handles this well (see F4).

## Methodology

Commands run on a fresh clone of `topoteretes/cognee@dev`, `uv sync --extra dev`, and one venv used for all runs. Environment variables toggled per test:

- Baseline: `unset LLM_API_KEY`
- Invalid key: `export LLM_API_KEY=sk-fake-just-to-pass-the-key-check`

Code paths inspected without running:

- `cognee/cli/_cognee.py` (top-level argparse + `-ui` action)
- `cognee/cli/commands/*_command.py` (the four primary commands)
- `cognee/cli/exceptions.py`
- `cognee/api/v1/ui/ui.py` (the Docker preflight and MCP-server launcher)
- `cognee/infrastructure/llm/LLMGateway.py` and `get_llm_client.py` (auth error surfacing)
- `pyproject.toml` (Python version constraint)
- `README.md` L105-L182 (Quickstart section)
- `.env.template` (tiered environment configuration)

## Part 2. Proposal

Each of the six findings above lands as a specific, bounded UX change below. Every proposal names the file and function the change lives in, the effort tier (Quick / Follow-up), and the risk. **Quick** items ship in this PR under Part 3. **Follow-up** items are documented here so the plan is legible but land in separate issues so this PR stays reviewable.

The design principles the whole proposal is measured against:

- **Calm.** No scary stack traces on first run. Errors are one line, plain-language, and name a fix.
- **Honest.** Tell the user up front what a command will cost, take, or require. Do not leak upstream implementation details unless the user asked for them.
- **Actionable.** Every failure surfaces one concrete next step the user can copy-paste.

### P1. Fail fast on authentication errors (addresses F1)

**Change:** classify `litellm.AuthenticationError` (and equivalent provider-level 401 / 403 errors) as terminal. The `tenacity` retry wrapper around `LiteLLMEmbeddingEngine.embed_text` and `LLMGateway.acreate_structured_output` skips the retry chain for these classes and raises immediately.

**Where in code:** the retry decorator sits inside `cognee/infrastructure/databases/vector/embeddings/LiteLLMEmbeddingEngine.py` and mirrors on the LLM path in `cognee/infrastructure/llm/LLMGateway.py`. Both use `tenacity.retry(retry=retry_if_exception_type(...))`. Add a `retry=retry_if_not_exception_type((AuthenticationError, PermissionDeniedError, ...))` predicate at the top of the classifier so authentication and authorization failures fall through without waiting.

**Effort:** Quick. Two files, ~10 lines each, one shared helper for the predicate. Behavior tests already have LLM stubs from #3601's harness pattern that can be reused.

**Risk:** low. The exception classes are already imported; we are narrowing the retry set, not expanding it. No user with a working key sees any behavior change.

### P2. Error-message contract with local remediation (addresses F2)

**Change:** every user-visible error from the primary CLI commands (`remember`, `recall`, `cognify`, `forget`) goes through one helper that renders a two-line block:

```
Cognee could not <action>: <plain-language cause>.
Try: <concrete next command or fix>
```

The helper reads `code`, `message`, and `remediation` from the exception object when they exist (which is what @ANAMASGARD's Pillar B work in #3604 lands) and falls back to a local remediation table for the common first-run failures otherwise.

**Where in code:** new module `cognee/cli/echo.py` extension or a sibling `cognee/cli/messaging.py`. Called from each command's outer `except CliCommandException` block (see `remember_command.py::execute` L112, and identical shape in the three sibling commands).

**Also under this proposal:** suppress the aiohttp `Unclosed client session` warning at the CLI boundary. Wrap the async runner in `warnings.catch_warnings()` with a `ResourceWarning` filter so cleanup noise stays out of the user's terminal.

**Effort:** Quick for the aiohttp suppression and the local-remediation table for the common failures. **Follow-up** for full alignment with the actionable-error work in #3604 (that PR has not merged yet). Local remediation stubs are the bridge.

**Risk:** low. Additive helper, existing except blocks call it; existing metadata (`Note: Please refer to our docs at 'https://docs.cognee.ai'...`) stays as the third line for anything not in the remediation table.

### P3. Next-step hints on every primary command (addresses F3)

**Change:** each of the four primary commands appends a single line after its success block naming the next natural command with a copy-paste example.

**Where in code:** the success branch of `execute()` in `cognee/cli/commands/remember_command.py`, `recall_command.py`, `cognify_command.py`, `forget_command.py`. Copy-write per command:

| Command   | Next-step hint |
| --------- | -------------- |
| `remember` | `Next: cognee-cli recall "your question" -d <dataset-name>` |
| `cognify`  | `Next: cognee-cli recall "your question" -d <dataset-name>` |
| `recall`   | `Next: cognee-cli remember <path-or-text> -d <dataset-name>` (if the result set was empty) or nothing (if it returned results) |
| `forget`   | `Next: cognee-cli remember <path-or-text> -d <dataset-name> to start fresh` |

No suppression flag ships: the issue asks for hints, not a new flag, and the hints are single calm lines. Machine consumers are already unaffected — `recall`'s hint is scoped to the empty pretty-output path, so `json` / `simple` output stays clean, and the other three commands' output is human-oriented metadata that a script does not parse line-for-line.

**Effort:** Quick. Four files, ~2 lines of new logic in each, one shared `hints` module for the copy.

**Risk:** low. Purely additive output; the new line comes after everything else.

### P4. Extract Docker preflight pattern into a shared preflight module (addresses F4)

**Change:** `_check_docker_available()` (currently in `cognee/api/v1/ui/ui.py`) is extracted into `cognee/cli/preflight.py` alongside four sibling checks: `check_python_version`, `check_llm_key_present`, `check_llm_provider_reachable`, `check_storage_writable`. Each returns `(bool, message)` matching the existing contract.

The primary commands call the relevant checks eagerly at the top of `execute()`; on failure they emit the same "Cognee could not X: Y. Try: Z" block as P2 and exit non-zero with no retry, no stack trace.

**Effort:** **Follow-up.** Landing the shared module fully is 4-5 preflight functions, wiring into each command, and tests for each check. Bigger than the quick-win bar.

**What lands in Part 3 under this proposal:** only the fail-fast + friendly-error path from P1 and P2. Full preflight ships as a separate PR alongside P6 (`cognee doctor`) so the review surface stays small.

**Risk:** low functionally, medium in scope. Splitting keeps this PR reviewable.

### P5. `--sample-data` first-run smoke test (addresses F5)

**Change:** a `--sample-data` flag on `cognee-cli remember`. When set, the command ingests a small bundled text fixture (`cognee/cli/samples/quickstart.txt`, a short prose passage that names three entities and their relationships so cognify produces a meaningful graph) instead of user data, and prints the fixture path so the run is transparent.

This removes the "compose a valid dataset first" decision from the first run: a newcomer can go straight to `cognee-cli remember --sample-data` and then `recall`, without hunting for input.

**Honesty note — this is what ships, and what does not.** The flag is a convenience over bundled input; it is **not** offline. It still requires `LLM_API_KEY` and still makes real cognify/embedding calls (W1 + W2 make the missing/invalid-key case fail fast with an actionable hint rather than hang). A truly keyless path — one that reaches a `recall` result with no API calls — needs a stubbed LLM, which is the scope of the mocked-test harness in #3601. That harness is not merged, so the keyless demo is **deferred** to a follow-up rather than shipped here with a bespoke, throwaway stub. The flag help text states the key requirement up front so the promise stays calm and honest.

**Where in code:** fixture at `cognee/cli/samples/quickstart.txt` (resolved via `importlib.resources` so it survives editable and wheel installs) and a small path-resolver + flag in `cognee/cli/commands/remember_command.py`. No `demo` sub-command and no LLM stub ship in this PR.

**Effort:** Quick for the fixture + flag. The keyless variant is a follow-up gated on #3601.

**Risk:** low. One additive, opt-in flag; the existing "no data supplied" path is preserved and now points at the flag.

### P6. `cognee doctor` preflight (addresses F6)

**Change:** new sub-command `cognee-cli doctor` that runs every preflight check from P4 and prints a bullet list of pass/fail with the actionable-message contract borrowed from `_check_docker_available()`. Zero exit code if everything passes; non-zero and a summary line if anything fails.

**Effort:** **Follow-up.** Depends on P4's shared preflight module. Ships as a separate PR.

**Risk:** low, but out of scope for this PR to keep the review surface small.

### Quick wins scoped for this PR (Part 3 preview)

Landing here in Part 3:

- **W1.** Fail fast on `AuthenticationError` / `PermissionDeniedError` in the tenacity retry policy (P1).
- **W2.** Local remediation table for the common first-run errors + aiohttp warning suppression at the CLI boundary (P2 partial).
- **W3.** Next-step hints on `remember`, `recall`, `cognify`, `forget` (P3).
- **W4.** `--sample-data` first-run smoke test with a bundled fixture (P5, online variant; the keyless variant is deferred).

Deferred to follow-up issues (not in this PR):

- Fully offline `--sample-data` (stubbed LLM, no API calls), gated on the #3601 mocked harness (P5, keyless variant).
- Full `cognee doctor` preflight command (P6).
- Extracting the preflight pattern into `cognee/cli/preflight.py` (P4 in full).
- Stripping leaked HTTP internals (e.g. `(Status code: NNN)`) from CLI-surfaced errors and replacing every "Please refer to our docs" instance with a remediation string once the actionable-error work in #3604 lands (F2a/F2b; only partially covered here by the remediation hint).
- Streamlined canonical README quickstart section rewrite. Kept out of scope so this PR is not entangled with a docs-review cycle. Follow-up.

### Non-goals

Explicit non-goals to prevent scope creep in review:

- Not changing the SDK-level `remember` / `recall` / `forget` function signatures. Only the CLI wrapping around them.
- Not touching the `-ui` launcher. Its Docker preflight is already good; extracting it is P4 (Follow-up).
- Not changing the LLM provider defaults or the pricing table (F7, F8 are #3606's scope).
- Not adding new configuration surface beyond one CLI flag (`--sample-data`). No new sub-command ships here. Everything else honors existing env-var conventions.

Part 3 quick-win implementation lands as separate commits on this same PR.
