# Code graph ingestion

## Reindexing symbols created before mapping version 3

Symbols now use `(repository, kind, name, file)` as their graph identity. Two
same-named declarations in different files stay distinct; moving line numbers
keeps the identity. Other node kinds retain their existing identities. Explicit
Enola `target_id` references resolve only to that fact. Missing explicit targets
and ambiguous name-only references are skipped rather than assigned to another
symbol.

Rebuild the code graph in the existing dataset from its repository manifest:

```python
await cognee.cognify(
    datasets=[dataset_id],
    data_ids=[repository_data_id],
    incremental_loading=False,
    data_cache=False,
)
```

Ordinary cognify checks stored mapping versions before completed items are skipped
and logs one reindex instruction per affected dataset invocation. This check is
best-effort diagnostics and never changes the outcome of ingestion.

Both flags are necessary to revisit a completed data item. Mapping version 4
forces extraction mapping even when the repository snapshot has not changed.
The loader writes current nodes and snapshot-derived edges, sweeps obsolete
code identities, then stamps the new mapping version. Repeating this operation
keeps the new identities stable. The source repository must still be available;
for a remote repository, re-add it first if its cached checkout is gone.

If the dataset also contains docstrings produced by the optional code-docs
pipeline, refresh them after reindexing to update their `symbol_id` metadata
and `describes` edges:

```python
from cognee.memify_pipelines.cognify_code_docs import cognify_code_docs_pipeline

await cognify_code_docs_pipeline(user=user, dataset=dataset_id)
```

This uses the existing docs pipeline and its configured embedding provider.
Unchanged document chunks retain their IDs. Before writing, the loader checks
references to old symbol IDs. A unique successor preserves the relationship,
evidence, and original dataset/data/run ownership. A legacy merged symbol with
multiple successors raises `SymbolMigrationError` (wrapped by cognify) before
node writes. Its `issues` list identifies the relationship and candidate IDs.
The old symbol's `file_path` is not evidence of which declaration an annotation
meant. Resolve those annotations explicitly before retrying; the old graph stays
intact on this preflight error.

Snapshot-derived edges now carry `producer=cognee_code_graph`. Only those edges
are eligible for stale-edge cleanup between surviving code nodes. Unmarked old
edges are preserved because their ownership cannot reliably be inferred from a
relationship name. Removed source-code nodes are still swept with their incident
edges; reference preservation here concerns identity migration, not keeping
references to deleted declarations.


## Coverage, imports and external references

Use `code_query={"operation": "coverage"}` for repository coverage. Other
operations retain their response shape; `ask.coverage` describes query context.
Capabilities version 2 advertises the new operations. The statuses are `partial` (observed skipped
work), `unknown` (missing extraction/mapping counters), and `observed_no_gaps`.
None means exhaustive static analysis. The report includes the extractor's
quality counters, git provenance and SDK mapping diagnostics, without claiming
that an extractor's sampled skip report is a complete file inventory.

Internal imports resolve against files actually
present in the snapshot and the modules containing them. Relative paths, Python
dotted imports and index/package files are supported. Go modulePath prefixes,
Java dotted paths and Rust crate paths resolve against the same known inventory;
this is not a full compiler package or alias resolver. Missing/ambiguous targets,
unknown aliases and external dependencies do not bind to a similarly named local
module. Enola's explicit target IDs remain authoritative.

`CodeReference` and `link_code_references` from `cognee.tasks.code_graph` link an
already-ingested README, document or Linear issue chunk to an exact
`repo + file + symbol`, with source URL, evidence and optional external issue ID.
They are opt-in pipeline tasks, not connector changes or a title-matching join.
The task verifies source data ownership in the active dataset and preserves
chunk provenance. Unresolved targets appear in the returned report; invalid
source ownership fails before any write. A repeated reference writes the same
edge. Use distinct chunks for distinct evidence about the same symbol.

Read those edges (and optional docstring `describes` edges) with
`code_query={"operation": "references", "id": symbol_id}`. This operation reads
external nodes on demand; it does not include document text in the default code
snapshot cache. See `examples/guides/code_reference_task.py` for a custom pipeline
task and search example. Linking adds no extraction, embedding or LLM cost.

Integration tests ingest real React, FastAPI and SQLAlchemy sources through
`add -> cognify`. They verify component -> hook -> client API and server endpoint
-> handler -> helper paths, and verify absent cross-repo deployment wiring and
ORM-query-to-table edges remain absent. Recognizing a table is not proof that a
particular query uses it; the pinned extractor does not emit that latter edge
for the test fixture.


## Mapper comparison (2026-10-27)

A fresh Enola snapshot of the tracked cognee source selection contained 32,739
facts. Mapping those exact facts with the branch HEAD mapper and the revised
mapper produced the following results (no LLM or network calls):

| Metric | Branch HEAD mapper | Revised mapper |
| --- | ---: | ---: |
| Nodes | 31,382 | 31,419 |
| Edges | 52,072 | 57,624 |
| Skipped relations | 30,907 | 24,041 |
| Mapping time, one local run | 1.597 s | 1.571 s |

At repository/kind/name level, 6,101 edge signatures were added and 586 were
removed: 513 imports, 47 calls and 26 implements edges. This normalization
intentionally ignores the new file-scoped identities, so it is a regression
inventory, not a ground-truth precision/recall score. Some previous imports
bound external package names to local modules or storage with the same name;
those are no longer accepted. Ambiguous symbols and missing explicit targets
remain unresolved rather than being rebound by name. The revised report still
contains 114 unresolved internal imports and 47 ambiguous relations; these are
observable limitations, not evidence of a complete graph.

A unique same-file symbol now breaks a name-only tie. Explicit target IDs stay
authoritative even if absent from the snapshot: name resolution in that case
could select a different declaration. External imports resolve only to a unique
package explicitly declared by a manifest, never to a same-named local module.
