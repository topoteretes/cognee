# Cognee Migration Tutorials

Step-by-step tutorials for migrating memories from other AI memory systems into Cognee.

Each tutorial uses an existing Cognee importer (no new code required) and a small
sample dump so the script runs end-to-end with a single command.

## Tutorials

| Script | What it migrates | Importer |
|---|---|---|
| [`migrate_from_letta_and_zep_tutorial.py`](migrate_from_letta_and_zep_tutorial.py) | Letta/MemGPT agent files + Zep graph exports | `LettaSource`, `ZepSource` |

## Running a tutorial

```bash
# One-time setup
uv sync --dev --all-extras --reinstall
cp .env.template .env          # add LLM_API_KEY at minimum

# Run
uv run python examples/tutorials/migrate_from_letta_and_zep_tutorial.py
```

## Import modes

All importers accept a `mode` argument that controls how source records are stored:

| Mode | Behaviour | LLM tokens |
|---|---|---|
| `preserve` | Map source records directly into the graph with no extraction | None |
| `re-derive` | Ingest raw content and re-run Cognee's extraction (`cognify`) | Yes |
| `hybrid` | Both: preserve the source graph and cognify raw content | Yes |

## COGX record mapping

Source concepts are normalized to COGX records before storage:

### Letta/MemGPT

| Letta concept | COGX record |
|---|---|
| Core memory block (`human`, `persona`, ...) | `COGXMemoryBlock` |
| Message history | `COGXEpisode` (with `COGXTurn` per message) |
| Archival passage | `COGXDocument` |

### Zep

| Zep concept | COGX record |
|---|---|
| Episode / episodic node | `COGXEpisode` |
| Entity / named node | `COGXEntity` |
| Fact / edge (with bi-temporal timestamps) | `COGXFact` |

Reference: `cognee/modules/migration/COGX.md`
