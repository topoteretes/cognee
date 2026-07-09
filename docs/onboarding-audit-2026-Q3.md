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

**What a user needs instead:** a single hint line at the end of each success block that names the next natural command with a copy-paste example. Suppressible via a `--quiet` flag for scripts.

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

**What a user needs instead:** a `--sample-data` or `cognee-cli demo` path that runs a full remember-cognify-recall cycle against bundled fixtures with the LLM stubbed. This is complementary to the mocked-test harness from issue #3601; the harness is for CI, the sample-data path is for humans.

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

## What Part 2 and Part 3 build

Part 2 (Proposal) turns the six findings into concrete UX changes. Part 3 (Quick wins) lands the low-risk half of the proposal in this PR. See the acceptance criteria on issue #3605 and the follow-up sections of this document once Part 2 lands.
