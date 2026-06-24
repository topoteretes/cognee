# Cognee Migration Tutorials

Step-by-step tutorials for migrating memories from other AI memory systems into Cognee.

Each tutorial uses an existing Cognee importer (no new code required) and a small inline sample
dump so the script runs end-to-end with a single command.

## Tutorials

| Script | What it migrates | Importer |
|---|---|---|
| [`migrate_from_graphiti_tutorial.py`](migrate_from_graphiti_tutorial.py) | Graphiti OSS knowledge graph (episodes, entities, bi-temporal facts) | `GraphitiSource` (`cognee/modules/migration/sources/zep.py`) |

## Running a tutorial

```bash
# One-time setup
uv sync --dev --all-extras --reinstall
cp .env.template .env          # add LLM_API_KEY at minimum

# Run
uv run python examples/tutorials/migrate_from_graphiti_tutorial.py
```

## Import modes

All importers accept a `mode` argument that controls how source records are stored:

| Mode | Behaviour | LLM tokens |
|---|---|---|
| `preserve` | Map source entities/facts directly into the graph — no extraction pipeline | None |
| `re-derive` | Ingest raw episode text and re-run Cognee's extraction (`cognify`) | Yes |
| `hybrid` | Both: preserve the existing graph and cognify raw episode text | Yes |

## COGX memory standard

Source records are normalised to COGX before storage:

| Source concept | COGX record |
|---|---|
| Episode / episodic node | `COGXEpisode` |
| Entity / named node | `COGXEntity` |
| Fact / edge (with bi-temporal timestamps) | `COGXFact` |

Reference: https://docs.cognee.ai/cogx
