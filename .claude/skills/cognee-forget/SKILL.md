---
name: cognee-forget
description: Use when removing data from cognee memory with forget() in the SDK, HTTP API, or CLI — finding which dataset and document hold the content to delete (listing datasets and data items, reading raw content), choosing between deleting one document, a whole dataset, or only the graph/vector memory, and doing it safely.
---

# Remove data with forget()

`forget()` is cognee's one deletion API. It removes one document, a whole
dataset, or only the derived memory (graph + vectors) while keeping the raw
files. Deletion cannot be undone, so the workflow is always **find, read,
confirm, then delete**.

> **Hard limits for agents**
> - Delete only what the user asked to forget. Identify it by reading the
>   content first; never guess from a file name alone.
> - Confirm the exact items with the user before deleting, unless they
>   already named exact ids.
> - The widest deletion you may run is `forget(everything=True)`, and only
>   when the user explicitly asks to wipe all of their memory. Never use any
>   other reset or wipe mechanism to delete data.

## Use it

### 1. Find the dataset

```python
import cognee

datasets = await cognee.datasets.list_datasets()   # datasets the user can read
for ds in datasets:
    print(ds.id, ds.name)
```

HTTP: `GET /api/v1/datasets`. CLI: `cognee-cli datasets list`.

### 2. List its documents

```python
items = await cognee.datasets.list_data(dataset_id)   # all Data rows, oldest first
for item in items:
    print(item.id, item.name, item.extension, item.created_at)
```

HTTP: `GET /api/v1/datasets/{dataset_id}/data?limit=100&offset=0` (limit up
to 1000; `GET .../data/count` for the total; example:
`examples/python/dataset_data_pagination.py`). CLI:
`cognee-cli datasets data <dataset_uuid>`.

Each item has `id`, `name`, `created_at`, `extension`, `mime_type`,
`raw_data_location`, `dataset_id`, `label`, `external_metadata` (including
any `node_set`), and `data_size`.

### 3. Read the content before deciding

Names are often `text_<hash>.txt`, so read the content to find what the user
means.

- HTTP: `GET /api/v1/datasets/{dataset_id}/data/{data_id}/raw` returns the
  stored file (404 if it is gone).
- SDK: there is no "get raw" helper; open the stored location:

```python
from cognee.infrastructure.files.utils.open_data_file import open_data_file

async with open_data_file(item.raw_data_location, mode="rb") as f:
    preview = f.read(2000).decode("utf-8", errors="replace")
```

Judge matches by meaning, not only by keywords, and show the user the
candidates (name + a short preview) before deleting.

### 4. Delete

| Goal | Call | What remains |
|---|---|---|
| One document | `forget(data_id=..., dataset_id=...)` (or `dataset="name"`) | Nothing of that document; shared entities stay while another document still references them |
| A whole dataset | `forget(dataset="name")` or `forget(dataset_id=...)` | The dataset is emptied: data rows, graph, vectors |
| Rebuild a dataset's graph later | `forget(dataset="name", memory_only=True)` | Raw files and data rows; graph, vectors, sessions and pipeline status are reset, so the data can be re-processed |
| One document's memory only | `forget(dataset="name", data_id=..., memory_only=True)` | That document's raw file and row |
| Everything the user owns | `forget(everything=True)` | Nothing. Only on explicit request (see the hard limits) |

Return values: `{"data_id", "dataset_id", "status"}` for a document,
`{"dataset_id", "status"}` for a dataset (plus `data_records_reset` with
`memory_only`), `{"datasets_removed", "status"}` for everything.

**HTTP:** `POST /api/v1/forget` with a JSON body; camelCase and snake_case
keys both work: `{"datasetId": "...", "dataId": "..."}`,
`{"dataset": "name", "memoryOnly": true}`, `{"everything": true}`. Invalid
combinations return 422.

**CLI:** `cognee-cli forget --dataset NAME | --dataset-id UUID
[--data-id UUID] [--memory-only]`, or `--everything` / `--all`. The CLI
**does not ask for confirmation**; confirm with the user first.

## Pitfalls

- **A `data_id` that is not in the dataset returns success and deletes
  nothing.** The delete path treats an unknown id as a custom-graph-model
  delete. Always take the id from `list_data` for that same dataset, and
  check it is still listed afterwards if it matters.
- **Pass either `dataset` or `dataset_id`, not both** (`ValueError`).
  `data_id` and `memory_only` both need a dataset.
- **`memory_only` is ignored when `everything=True`** in the SDK (the CLI
  rejects the combination). `everything=True` always deletes everything.
- **Not found and not allowed look the same.** An unknown dataset name and
  one the user cannot delete both raise `DatasetNotFoundError`. Deleting
  needs the `delete` permission on the dataset (see the `cognee-permissions`
  skill). `everything=True` covers only datasets the user owns.
- **Sessions that cited deleted data are invalidated** so recall stops
  returning answers built on it. Agent-trace entries are not invalidated.
- **Changing a document is not a delete.** To replace a document's content,
  use `cognee.update(data_id=..., data=..., dataset_id=...)`, which keeps its
  id and re-extracts only the changed parts.
- `cognee.delete()` is deprecated; use `forget()`.

## How it works

`forget()` resolves the dataset with the `delete` permission, then:

- **One document** → `datasets.delete_data()`: takes the dataset lock,
  deletes the graph nodes and edges the document owns, the matching vectors
  and edge evidence, invalidates sessions that cited them, then deletes the
  `Data` row. Ownership is tracked per document (source-refs on graph
  elements), so an entity shared by two documents survives until both are
  deleted. Raw files are reference-counted by storage location.
- **A dataset** → `datasets.empty_dataset()`.
- **`memory_only`** → drops the dataset's graph/vector memory and resets its
  pipeline status, leaving raw data for a rebuild.
- **`everything`** → `datasets.delete_all()` for the user's datasets plus
  their session cache.

Key files:

- `cognee/api/v1/forget/forget.py` (SDK), `routers/get_forget_router.py`
  (HTTP), `cognee/cli/commands/forget_command.py` (CLI)
- `cognee/api/v1/datasets/datasets.py` (`list_datasets`, `list_data`,
  `delete_data`, `empty_dataset`, `delete_all`)
- `cognee/api/v1/datasets/routers/get_datasets_router.py` (list, count, raw)
- `cognee/infrastructure/databases/provenance/source_refs.py` (per-document
  ownership of graph elements)
- `cognee/infrastructure/files/utils/open_data_file.py`

## Extending it

- Anything new that writes graph nodes or edges for a document must record
  its source-refs, or `forget(data_id=...)` cannot find and remove it.
- Anything new that stores per-document data outside the graph (like edge
  evidence) needs a cleanup step in the delete path and in `memory_only`.
- Unit tests for `forget()` are in `cognee/tests/unit/api/v1/forget/`
  (argument validation, the HTTP endpoint, `memory_only`); cover both the
  document and the `memory_only` paths for new deletion behaviour.
