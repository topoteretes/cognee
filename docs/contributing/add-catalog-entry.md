# Adding a catalog entry

The Cognee Integrations Hub and Use-Case Gallery are generated from the YAML entries under `catalog/entries/`. Adding a new integration, use-case, or community package is a single-PR change: catalog entry, plus a runnable example the entry points at.

This guide covers the mechanics. The contract itself is `catalog/schema.json`.

## What the catalog is for

Users looking to adopt cognee ask two questions:

- "Does cognee work with my stack?" (answered by the Integrations Hub)
- "How do I do the thing I need done?" (answered by the Use-Case Gallery)

Every catalog entry is one card in one of those views. It carries just enough context for a user to decide it's the right starting point, plus a runnable example they can copy-paste.

## Anatomy of an entry

Every entry is a YAML file under `catalog/entries/{integrations,use-cases,packages}/`. The filename stem must match the entry's `id`. Example:

```yaml
# catalog/entries/use-cases/document-qa.yaml
id: document-qa
title: Document Q&A backed by a knowledge graph
kind: use-case
stack: use-case
tags:
  - document-qa
  - graph-rag
  - recall
  - reading-comprehension
summary: Answer questions grounded in a set of documents using cognee's graph-completion retrieval instead of plain RAG.
what_youll_build: A recall pipeline that ingests documents, extracts entities and relationships into a graph, and answers questions with citations back to source chunks.
quickstart: |
  uv pip install cognee
  export LLM_API_KEY=your_openai_key
  python examples/guides/recall_core.py
expected_output: |
  A ranked list of retrieved passages plus a synthesized natural-language
  answer citing the original documents. Compare against a plain RAG baseline
  to see graph-completion's contribution.
difficulty: easy
example_path: examples/guides/recall_core.py
```

## Fields

The full schema lives in `catalog/schema.json`. Highlights:

- **`id`**: Lowercase, dashes, no spaces. Must match the filename stem. This is the stable machine identifier every downstream tool uses.
- **`title`**: The human-readable name as it appears on the Hub card.
- **`kind`**: `integration`, `use-case`, or `package`. Drives which subdirectory the file lives under and which view it appears in.
- **`stack`**: The primary technology bucket (`llm-provider`, `vector-store`, `graph-store`, `relational-store`, `framework`, `agent-runtime`, `workflow-tool`, `observability`, `loader`, or `use-case`). Drives the Hub filter chips.
- **`tags`**: Free-form filter labels. Redundant with `stack` on purpose so users can search by outcome or provider without hitting a taxonomy wall.
- **`summary`**: One sentence, shown on the Hub card. Answers "does this work with my stack".
- **`what_youll_build`**: One sentence promising a concrete outcome. Answers "what do I get if I follow the quickstart".
- **`quickstart`**: A copy-paste block that gets a newcomer running. Include install, env, and a single run command. Multi-line YAML block scalar (`quickstart: |`).
- **`expected_output`**: A concrete description of what the user should see after running the quickstart. Doesn't need to be exact values, but should be specific enough that a user knows if their run worked.
- **`difficulty`**: `easy`, `medium`, or `advanced`. Measures the effort for a new user, not code complexity.
- **`repo`, `path`**: Required for integrations and packages. Point at the source in one of `topoteretes/cognee`, `topoteretes/cognee-community`, or `topoteretes/cognee-integrations`.
- **`example_path`**: Required for use-cases. Path within `topoteretes/cognee` to a runnable script.
- **`inventory_slug`**: Optional. If your integration already has a slug in `cognee-integrations/integrations/inventory.yml`, set it here so the drift check keeps them aligned.
- **`docs_url`**: Optional link to a longer doc page (e.g. `docs.cognee.ai/integrations/openai`).

## Adding an entry: the workflow

1. Pick the right subdirectory (`integrations/`, `use-cases/`, or `packages/`).
2. Copy an existing entry as a starting point:
   ```bash
   cp catalog/entries/use-cases/document-qa.yaml catalog/entries/use-cases/your-new-entry.yaml
   ```
3. Edit every field. The schema catches typos in field names, so a stray letter turns into a CI error, not a silent bug.
4. Make sure the `example_path` (for use-cases) or the `path` (for integrations/packages pointing at `topoteretes/cognee`) resolves against the current checkout.
5. If you're referencing an example that doesn't have a mocked test yet, coordinate with #3601 so the example runs in CI.
6. Run the validation locally (a standard `uv sync` already provides the `pyyaml` and `jsonschema` the tooling needs):
   ```bash
   uv sync
   uv run python -m catalog.loader
   uv run python -m catalog.inventory_sync
   ```
   The loader must exit 0 for the entry to be considered valid. The drift check tolerates coverage gaps (integrations in the inventory but not yet in the catalog) but fails on stale references.
7. Open a PR. The `Catalog` workflow will re-run both checks.

## Coverage gaps

If you add an entry with an `inventory_slug`, the drift check confirms the slug exists upstream. If you don't add one, you're implicitly saying "this is a brand-new integration not in inventory.yml yet." That's fine.

If you want to help close a coverage gap, run the drift check locally to see the outstanding list, then add entries one at a time. Batch PRs of 5-10 entries are welcome; larger batches are harder to review.

## What if I'm adding an integration that lives in a private fork?

Point `repo` at the private fork, or leave `repo` and `path` unset and treat it as a use-case (`kind: use-case`) with a public `example_path`. The catalog is designed for public discovery, so anything a user can't reach shouldn't be an entry.
