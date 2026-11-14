# Markdown task-list SDK connector

A Markdown task-list connector for [cognee](https://github.com/topoteretes/cognee): sync a
folder of `.md` files that hold `- [ ]` / `- [x]` checklists into memory as documents,
incrementally, with forget-on-delete. Read-only: the connector never writes the source
files.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Each file becomes one document — its tasks as a checklist, open and done —
that goes through normal chunking and LLM graph extraction, so "what is still open in the
home renovation list?" is one search.

## Requirements

Bundled with the Cognee SDK, no extra install. There is no account, no token and no network
access: the connector reads files from the machine that runs it.

**SDK only.** The connector has no entry in the integration provider table
(`cognee/modules/integrations/ingestion.py`): nothing hosted can reach a folder on your disk.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import markdown_tasks_source

source = markdown_tasks_source(source_path="~/notes/tasks")
await cognee.remember(
    source,
    dataset_name="project_tasks",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
    self_improvement=False,
)

# Counts only: files published and files forgotten by this run.
assert source.cognee_sync_stats["deleted"] == 0
```

Run the same call again whenever you want to sync. A file whose tasks did not change makes
the same row id and is not re-cognified. Mark a task done in the file, run again, and the
document is updated.

| Argument | Meaning | Default |
|---|---|---|
| `source_path` | The folder walked recursively. `ValueError` unless it is an existing directory | required |
| `resource_name` | Names this folder: it keys the staging table and the sync state | `markdown_tasks_<12 hex of the sha256 of the resolved folder>` |
| `check_active` | Host checkpoint, called before each file is read and before each row is published | none |

All arguments are keyword-only. After a run, `source.cognee_sync_stats` holds counts only:
`synced` (files published) and `deleted` (files forgotten).

## What is read

Every `*.md` file under the folder (case-insensitive on the suffix). Not read: hidden files
and folders, symbolic links, and any other suffix.

A **task** is a standard checklist line — `- [ ]`, `- [x]` (and `*` / `+` bullets, `[X]`
counts as done) — with its nesting indentation preserved, so a sub-task renders nested
under its parent. Everything else in the file (prose, headings, frontmatter) is not a task
and is not carried over. **Fenced code blocks are skipped**, so a README that documents the
syntax with an example checklist does not become tasks.

A file with no checklist items publishes no row: it is not a task file. If it held tasks
before and holds none now, they are forgotten the same way a deleted file's are.

One file becomes one document:

| Field | Value |
|---|---|
| id | `markdown-tasks:<12 hex of the sha256 of path + rendered body>` (the dlt primary key) |
| title | the file name without `.md` |
| content | `# <title>` followed by the file's tasks as a checklist, nesting preserved |
| url | empty |
| cognee_node_set | `markdown-tasks:open` while the file has open tasks, `markdown-tasks:done` once it also has done ones — only when there is any |
| cognee_structure | the file's folders, see below. Only for a file inside a folder |

The content hash is part of the id, so a checked-off task gives the file a new id and the
old version is forgotten on the same run that adds the new one.

**Structure** is `{"ancestors": [...]}`, nearest folder first, each as
`{"kind": "folder", "id": "<folder path>", "name": "<folder name>"}` — where a file sits,
like the Obsidian source. A file at the top of the folder has no ancestors.

## Incremental model

There is no cursor. Every run walks the whole folder, reads every `.md` file and publishes
a row for each one that holds tasks. A file whose text did not change makes the same row
(the id carries the hash), cognee recognises it, and nothing is extracted again, so a
quiet sync costs the reads and no LLM calls. File times are never used.

The sync state is kept in `dlt.current.resource_state()` and holds the ids the last
complete run published (`members`, sorted) and a format version (`v`). It is written after
the last row, so a run that raises keeps the old state.

## Deletion and its gaps

A file is forgotten when a complete run no longer publishes it: deleted, renamed, outside
the folder, or stripped of its checklists. Every file is read before the first row is
published, so a failure forgets nothing:

- If `source_path` is not a directory when the run starts, the run raises
  `MarkdownTasksUnavailableError` before anything is published or forgotten.
- If the folder is empty but earlier runs synced files from it, the run raises the same
  way — a drive that did not mount looks exactly like a folder whose files were all
  deleted. To forget everything on purpose, leave one file in the folder.
- A file that cannot be read (permissions, not UTF-8) raises `MarkdownTasksReadError`
  before any row is published. Fix the file and run again.

Gaps: a very large folder is read in full on every run; `cognee.forget(data_id=...)` of a
file still in the folder is undone by the next sync; task *content* is whatever the line
says — no dates, assignees or reminders (the connector is read-only and does not schedule).

## Privacy

Nothing is read until you construct a source and call `remember`. Task text becomes cognee
documents and, during cognify, is sent to your configured LLM and embedding providers like
any document — check that you may send it there, or use local models. Logs, errors and
`cognee_sync_stats` carry counts, classes and folder-relative paths, never an absolute
path. dlt's staging database for the dataset (`dlt_database_<dataset>`) holds the parsed
tasks, like any dlt source.

## Graph shape

A task file is one document whose checklist is text. A task named in two files merges into
one node during extraction; folders are `StructureContainer` nodes linked by `child_of`;
the open/done node sets scope a recall to what is still open.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_markdown_tasks_source.py
```

No account and no network: the tests build folders of fixture files in a temporary folder.
