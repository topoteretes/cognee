# Cognee — Project Memory (VS Code)

Give your editor a persistent, **citable memory** of the project. Remember and recall
knowledge without leaving VS Code, and **ask your project memory** — get answers grounded in
what Cognee knows about the current repository, with links back to the source files.

> Powered by [Cognee](https://github.com/topoteretes/cognee). Works against a local Cognee
> server or a Cognee Cloud tenant.

## Features

- **Remember Selection / Remember File** — store code or notes into per-repository memory.
- **Ask My Project Memory** — a panel that answers questions about the repo, with clickable citations.
- **Recall** — a quick query from the command palette.
- **Index Workspace** — opt-in bulk ingest, respecting `.gitignore`/`.cogneeignore`, with a preflight summary.
- **Forget Project Memory** — clear the graph (keep files) or delete the dataset.
- **Per-repository isolation** — each repo maps to a stable `vscode_<hash>` dataset (from the git remote, or the workspace path as a fallback).

## Requirements

A reachable Cognee backend — either of:

- **Cognee Cloud** (no local setup): your tenant URL (e.g. `https://<tenant>.cognee.ai`) and an API key.
- **Local server**: a Cognee server running on `http://localhost:8011`
  (see the [Cognee docs](https://docs.cognee.ai/)); this path needs its own `.env` (e.g. `LLM_API_KEY`).

The extension itself needs **no environment variables** — it is configured entirely through VS Code
settings and secret storage.

## Quick start

1. Run **`Cognee: Set Up`** from the command palette.
   - Enter your endpoint (`http://localhost:8011` or your Cloud tenant URL).
   - Enter your API key if using Cloud (stored securely in the OS keychain, not in settings).
   - The command runs a health check so you know the connection works.
2. Open a file, select some code, and run **`Cognee: Remember Selection`**.
3. Run **`Cognee: Ask My Project Memory`** and ask a question — the answer appears with its sources.

## Commands

| Command | What it does |
| --- | --- |
| `Cognee: Ask My Project Memory` | Open the panel and query the repo's memory with citations. |
| `Cognee: Recall` | One-off query from the palette. |
| `Cognee: Remember Selection` | Store the current selection (or the whole file). |
| `Cognee: Remember File` | Store a file (also on the explorer context menu). |
| `Cognee: Index Workspace` | Bulk-ingest eligible files after a preflight confirmation. |
| `Cognee: Forget Project Memory` | Clear memory (keep files) or delete the dataset. |
| `Cognee: Set Up` | Configure endpoint + key and run a health check. |

## Settings

| Setting | Default | Description |
| --- | --- | --- |
| `cognee.endpoint` | `http://localhost:8011` | Cognee backend base URL. |
| `cognee.apiKey` | `""` | API key (`X-Api-Key`). Prefer `Cognee: Set Up`, which uses secret storage. |
| `cognee.datasetOverride` | `""` | Fixed dataset name; empty derives `vscode_<hash>` per repo. |
| `cognee.searchType` | `auto` | Recall strategy; `auto` lets Cognee route the query. |
| `cognee.topK` | `15` | Max recall results. |
| `cognee.includeReferences` | `true` | Attach source citations to answers. |
| `cognee.ingestion.respectGitignore` | `true` | Skip ignored files when indexing. |
| `cognee.ingestion.maxFileSizeKb` | `512` | Skip files larger than this when indexing. |
| `cognee.requestTimeoutMs` | `300000` | HTTP request timeout. |

## How it works

The extension talks to Cognee over the HTTP API:

- **Ask / Recall** → `POST /api/v1/recall` with `include_references: true`, scoped to the workspace dataset.
  Citations are parsed from the answer's `Evidence` block (chunk/document level today). When several
  files share the cited name, the one whose content matches the snippet is opened; otherwise you're
  asked to pick.
- **Remember / Index** → `POST /api/v1/remember` (ingest + graph build in one call).
- **Forget** → `POST /api/v1/forget`.

All editor-agnostic logic lives in [`src/core`](src/core) (no `vscode` imports), so it is unit-tested
against a mocked backend and can back other editors (e.g. a JetBrains sidecar) unchanged.

## Development

```bash
npm install
npm run build      # bundle to dist/extension.js (esbuild)
npm run typecheck  # tsc --noEmit
npm test           # vitest — runs against a mocked backend, no live keys
```

Press **F5** in this folder to launch an Extension Development Host with the extension loaded.

## License

Apache-2.0. See [LICENSE](LICENSE).
