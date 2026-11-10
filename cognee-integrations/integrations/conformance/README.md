> **Copy.** These case tables are shared by every Cognee agent plugin. Only the
> Hermes plugin has moved into this repository so far; Claude Code, Codex and
> OpenClaw still test against the copy in `integrations/conformance/` of
> [topoteretes/cognee-integrations](https://github.com/topoteretes/cognee-integrations).
> Until they move too, change a table in both places.

# Session id sanitization conformance

Every Cognee integration turns a native session id into a Cognee session id with
the same shape: `{agent}_{native_id}`. Before the native id is used, each
integration runs it through a small sanitizer. The rule is:

- Keep only the ASCII set `[A-Za-z0-9-_.]`.
- Replace every other character with `_`.
- Trim leading and trailing `.` and `_`.
- Cap the length at 120 characters.

`session_id_cases.json` is the single source of truth for that rule. It is a
list of `{ "input", "expected", "note" }` cases. Each test loads this file and
checks its sanitizer against it. The tests live with the suite that already
covers the tree they test, so no integration needs a test project of its own:

- claude-code and codex:
  `integrations/tests/tests/unit/test_session_id_conformance.py` — the shared
  hook-plugin suite, parametrized over both plugin trees.
- hermes-agent: `integrations/hermes-agent/tests/test_session_id_conformance.py`
- openclaw: `integrations/openclaw/__tests__/unit/test_session_id_conformance.ts`

Because every test reads the same file, any implementation that drifts from the
rule fails its test. CI runs all of them on every pull request through the
existing jobs in `.github/workflows/ci.yml`: the shared suite via `test-python`
(`detect-changes` maps a claude-code or codex change onto the `tests` cell) and
again under Python 3.9 via `test-hooks-python39`, hermes-agent via `test-python`
(it has its own `pyproject.toml`), and openclaw via `test-typescript`, which
runs `npm test` as well as the type check.

## Note on empty results

The table only holds inputs whose sanitized output is not empty. hermes-agent
returns `"session"` when the result would be empty, while the others return an
empty string. That one case is left out on purpose so the shared table stays
identical for all four integrations.

# Dataset name sanitization conformance

cognee rejects a dataset name that contains a space or a dot
(`check_dataset_name`, run on every write: remember, improve, memify and the
pipelines). Every integration therefore runs a configured or derived dataset
name through a small sanitizer before using it. The rule is deliberately narrow:

- Trim surrounding whitespace.
- Replace every run of spaces and dots with a single `_`.
- If nothing but underscores is left (and the input was not already that),
  use the integration's default dataset instead.

Nothing else changes. A name the server accepts today (`Foo+Bar`, `héllo`,
`_x_`) comes back unchanged, so no user is silently moved to a new, empty
dataset. A name that contains a space or a dot never took a write, so rewriting
it orphans nothing. The vscode extension's `sanitizeDatasetName` is stricter
(`[^A-Za-z0-9_-]` → `_`) because it only cleans names it derives itself.

`dataset_name_cases.json` holds `{ "input", "fallback", "expected", "note" }`
cases; each test passes `fallback` as the default, so one table serves
integrations with different defaults. The tests live next to the session-id ones:

- claude-code, codex and antigravity:
  `integrations/tests/tests/unit/test_dataset_name_conformance.py`
- hermes-agent: `integrations/hermes-agent/tests/test_dataset_name_conformance.py`
- openclaw: `integrations/openclaw/__tests__/unit/test_dataset_name_conformance.ts`

Where a name is typed explicitly (the dataset-switch commands), an invalid name
is refused with the sanitized form as a suggestion instead of being rewritten,
so a switch never lands in a dataset the user did not name.

# Recall text trimming conformance

On cognee 1.6.x a graph recall with `only_context` returns the full prompt the
server's own LLM would have received: the conversation history, the question
template around the retrieved context, and the `## Active session guidance`
block. The hook plugins and openclaw keep only the retrieved context and shrink
it the same way, without an LLM:

- Find the context by the question marker (`` The question is: `<prompt>` ``)
  and the final backtick, or the last `` `\n\n## Active session guidance ``.
  Text that does not fit is returned whole and flagged `parsed: false`.
- Drop passages whose `Session ID:` (or `# Session learning … (session <id>)`)
  is exactly the current session id.
- Cap each passage (default 2000 chars, `COGNEE_RECALL_PASSAGE_CHARS`) at a
  paragraph break, marked `…[+N chars]`.
- Prune entity bullets that only state a type or a node set, and headings left
  without bullets. Facts are kept as they are.
- Drop passages from the end until the whole fits the budget (default 12000
  chars, `COGNEE_RECALL_CONTEXT_CHARS`). `0` disables either limit.

Lengths are Python `len()` (code points) and whitespace is Python's
`str.isspace()`, so the TypeScript port counts and strips the same way.

`recall_text_cases.json` is generated by running the hook plugins'
`_recall_text.py`, so its `expected` values are ground truth. Each case has a
`kind`: `trim` (the whole pipeline: `input_text`, `prompt`, `session_id`,
`passage_chars`, `context_chars`, where `null` means the reference default
above (2000 / 12000; a suite may ship different defaults, so the tests pin
these through the environment); it expects
`context`, `guidance` and the logged `stats`), `cap_passage`,
`prune_entities` or `split_passages`. The tests:

- claude-code, codex and antigravity:
  `integrations/tests/tests/unit/test_recall_text_conformance.py`
- openclaw: `integrations/openclaw/__tests__/unit/test_recall_text.ts`
  (port in `integrations/openclaw/src/recall-text.ts`)
- hermes-agent: `integrations/hermes-agent/tests/test_recall_text_conformance.py`
  (port in `integrations/hermes-agent/cognee_integration_hermes/recall_text.py`)

When `_recall_text.py` changes on purpose, regenerate the table from it and
make the port pass again.

# Capture policy conformance

Automatic capture decides which tool calls are stored and scrubs secrets from
what is stored. The Python hook plugins implement it in `_capture_policy.py`
(claude-code, codex and antigravity ship identical copies); openclaw ports it to
`integrations/openclaw/src/capture-policy.ts`. The rules:

- `COGNEE_CAPTURE_TOOLS` (allow, default `*`) and `COGNEE_CAPTURE_DENY_TOOLS`
  (deny, applied after the allow list) take tool names, fnmatch globs, or
  `Tool(prefix:*)`. A prefix is checked against the first word(s) of the
  command and must end on a word boundary: `Bash(git:*)` keeps `git status`,
  drops `gitk` and `cd x && git status`.
- An entry that cannot be parsed (`Bash(rg)`, `Bash()`, a broken JSON array)
  fails closed: nothing is captured and the error is reported.
- Any path-like param that matches the built-in secret-file globs (`.env*`,
  `*.pem`, `*.key`, ssh and aws credentials, `secrets.*`, `credentials.*`, …)
  or `COGNEE_CAPTURE_DENY_PATHS` drops the call.
- The same globs apply to the words of a shell command (any tool whose params
  carry a string `command`): the command is split POSIX-style (`shlex`; plain
  whitespace if the quoting does not parse), each word is cut at `; | & ( ) < >`
  and backticks, and a leading `$HOME` / `${HOME}` is read as `~`. So
  `cat ~/.aws/credentials`, `less "$HOME/.ssh/id_rsa"`, `wc -l <.env` and
  `echo $(cat .env)` are not captured. It is a word check, not a shell: a path
  reached through another variable (`$AWS_DIR/credentials`), a glob
  (`~/.aws/cred*`) or a program that reads a file without naming it is not
  caught. Like a structured path, any mention counts, so `git add .env.example`
  is not captured either.
- Redaction (on unless `COGNEE_CAPTURE_REDACT` is false) replaces private keys,
  connection strings, bearer/basic tokens, credential assignments (including
  `aws_secret_access_key = …`, `SECRET_KEY = …`, `AccessKeyId`, `SessionToken`,
  npm `_authToken=` / `_auth=`), vendor keys (including `npm_…` and `pypi-…`
  tokens), AWS access key ids (`AKIA…`, `ASIA…`), bcrypt hashes, secret-named
  object keys and `COGNEE_CAPTURE_REDACT_PATTERNS` matches with
  `[redacted:<kind>]`. Object keys are redacted like any string; a secret-named
  key keeps its name and has its value masked.

`capture_policy_cases.json` holds `tool_cases` (`{ note, env, tool, params,
expected }`, where `expected` is `true`, `false` or `"error"`) and
`redact_cases` (`{ note, env, input, expected }`, or `error: true`). `env` is
the `COGNEE_CAPTURE_*` values for the case. The expected values were produced by
running the Python module; when it changes on purpose, regenerate them from it.
Cases use Claude Code tool names (`Bash`, `Read`); openclaw matches tool names
case-insensitively (its tools are `exec`, `read`, …) and runs every case with
the lowercased name too.

- claude-code, codex and antigravity:
  `integrations/tests/tests/unit/test_capture_policy_conformance.py`
- openclaw: `integrations/openclaw/__tests__/unit/test_capture_policy.ts`
- hermes-agent: `integrations/hermes-agent/tests/test_capture_policy_conformance.py`
  (port in `integrations/hermes-agent/cognee_integration_hermes/capture_policy.py`;
  settings come from the plugin config, falling back to the `COGNEE_CAPTURE_*`
  variables, and tool names match case-insensitively as in openclaw)
