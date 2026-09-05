# Docstring pilot — the ten selected files

This file is the **allowlist**, not a suggestion. `tools/check_docstring_pilot_path.py`
parses the table below and rejects any `path` that is not in it, so
`.github/workflows/docstring_pilot.yml` cannot be pointed at a file the pilot did not
choose. The `docstring-author` skill beside it reads the same table, so the allowlist and
the author's instructions cannot drift apart. Adding a row is a human decision made in a
reviewed PR — the automation never extends this list.

It lives under `docstring-author` rather than somewhere neutral because the author is the
only agent that reads it: the reasons below say what each file was picked to demonstrate,
which is the expected answer, so `docstring-critic` is deliberately kept away from them.

Selection principle: files a developer or a coding agent is likely to open while
*evaluating or integrating* cognee — package entrypoints, public SDK functions, core
abstractions, extension points, and the enum whose members users pass as arguments.
Five files have weak or missing docstrings and test whether the author can add real
context; five are already well documented and test whether the critic has the restraint to
leave them alone.

Docstring counts below were measured on `dev` at `66cf082` and will drift as the
repo moves. They record why each file was picked, not a contract.

## Weak or incomplete surfaces

| File | Why it is a high-value docstring surface |
|---|---|
| `cognee/api/v1/search/search.py` | Public SDK entrypoint re-exported as `cognee.search`, whose one function takes over thirty keyword arguments and carries no docstring at all, so `help(cognee.search)` tells an integrator nothing about which argument applies to which search type. |
| `cognee/modules/search/types/SearchType.py` | The enum whose twenty members are the `query_type` value users pass to `search()` and `recall()`, with no module, class, or member documentation — so choosing between `HYBRID_COMPLETION` and `GRAPH_COMPLETION` today means reading the retriever implementations. |
| `cognee/shared/data_models.py` | Holds `KnowledgeGraph`, `Node`, and `Edge`, the shapes every LLM extraction call fills in and the baseline a custom `graph_model` is written against, yet 28 of its 41 public models are undocumented and several that are documented are tautological (`"""Knowledge graph."""`). |
| `cognee/infrastructure/llm/LLMGateway.py` | The single seam every LLM call in cognee passes through and the place a new provider is wired in, with all three public gateway methods (`acreate_structured_output`, `create_transcript`, `transcribe_image`) undocumented. |
| `cognee/modules/pipelines/operations/pipeline.py` | The orchestrator that `add`, `cognify`, and `memify` all funnel into; it has no module docstring and its public `run_pipeline_per_dataset` is undocumented, leaving the per-dataset locking and concurrency contract invisible to anyone writing a custom pipeline. |

## Already-good surfaces (restraint tests)

| File | Why it is a high-value docstring surface |
|---|---|
| `cognee/api/v1/add/add.py` | Public SDK entrypoint whose single ~6.6k-character docstring already covers loaders, storage, and permissions — the clearest test of whether the critic pass leaves a strong surface alone instead of padding it. |
| `cognee/api/v1/remember/remember.py` | The primary v1 memory entrypoint, all six public functions documented; its visibility means an unsupported claim added here would do the most damage, which makes it the strictest restraint test in the set. |
| `cognee/infrastructure/databases/graph/graph_db_interface.py` | The extension point every new graph backend implements, already documented across all 50 public members — tests whether the author pass can add missing system-level context (which backends isolate per dataset, what a new adapter must guarantee) without inflating 50 method docstrings. |
| `cognee/infrastructure/engine/models/DataPoint.py` | The base class every graph node inherits, fully documented today; its metadata and versioning contract is exactly the place a plausible-sounding but unverifiable claim would slip in. |
| `cognee/modules/retrieval/base_retriever.py` | The extension point a contributor implements to add a new `SearchType`, with all nine public members documented and the three-step retrieval workflow already spelled out in the class docstring — a restraint test on a file people read instead of the docs. |
