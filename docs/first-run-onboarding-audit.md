# Calm, honest first run audit

This audit covers the first-run paths from issue #3605 and focuses on low-risk changes that
help a new user reach a first result without guessing the next command.

## Scope

Checked the README quickstart, CLI command help, local CLI command behavior, and Docker/UI
setup notes. This pass did not run a live token-consuming LLM recall; the quick wins below
target the no-key and empty-result friction that can be tested without external services.

## Path snapshot

| Path | First result path | Decisions before step 1 | Main first-run risk |
| --- | --- | --- | --- |
| Python SDK | Install, set `LLM_API_KEY`, call `remember`, then `recall`. | Package tool, API key/provider, session vs dataset memory. | Missing or invalid key appears only when the user reaches LLM-backed work. |
| CLI | `remember` is the shortest path; `add` + `cognify` is the lower-level path. | Which command to start with, dataset name, background or foreground processing. | Success and empty-result output did not consistently say what to do next. |
| Local UI | `cognee-cli -ui`. | Docker/Colima availability and local service startup. | Docker is required before the UI path can work. |
| Docker Compose | Copy `.env.template`, set `LLM_API_KEY`, choose profiles, run Compose. | Which profile, which backing services, local image vs prebuilt image. | More choices before the user sees a memory result. |
| Prebuilt image | Pull and run with env configuration. | Env file/API key and exposed services. | Fast to start, but still needs honest key/service prerequisites. |

## Findings

| Rank | Path | Friction | Impact |
| --- | --- | --- | --- |
| 1 | CLI `add` / `cognify` / `remember` | Successful commands ended without a concrete next command. | A new user can ingest data and still not know how to ask the first question. |
| 2 | CLI `search` / `recall` | Empty results only said that nothing was found. | First-run users cannot tell whether they used the wrong query or skipped ingestion. |
| 3 | CLI LLM-backed commands | Provider or API-key failures surfaced as raw provider text. | The fix is usually simple, but users have to infer that `LLM_API_KEY` is required. |
| 4 | CLI `remember --background` | Background processing could imply recall is ready immediately. | Users may query too early and interpret empty results as product failure. |
| 5 | UI / Docker | `cognee-cli -ui` needs Docker or Colima; Compose adds profile choices. | Users can spend time on infrastructure before seeing a memory result. |

## Proposal

The calm CLI path should be:

1. `cognee-cli remember "Cognee turns documents into AI memory."`
2. `cognee-cli recall "What should I remember?"`
3. If the user chooses the lower-level path, `add` should point to `cognify`, and `cognify`
   should point to `search`.

The CLI should also keep failure states actionable:

- Empty `search` and `recall` output should say how to create memory first.
- API-key and authentication failures should mention `LLM_API_KEY`.
- Background runs should say to wait for processing before recall.

## Quick wins implemented

- Added next-step hints after local `add`, `cognify`, `remember`, and `forget`.
- Added matching next-step hints for `--api-url` dispatch paths.
- Added first-run guidance after empty local/API `search` and `recall`.
- Added API-key guidance to local command failures that mention API-key or authentication errors.

## Follow-ups

- Add `cognee doctor` for Python version, Docker, API key, and backend reachability checks.
- Add a no-key sample-data path if the project supports an offline model or mocked local mode.
- Split UI and Docker onboarding into a separate checklist with exact Docker/Colima diagnostics.
