# Obsidian SDK connector

An Obsidian data-source connector for [cognee](https://github.com/topoteretes/cognee): sync a
local vault (a folder of Markdown notes) into memory, incrementally, with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Every note becomes one text document that goes through normal chunking and LLM
graph extraction.

It fixes what `cognee.remember(<folder>)` does with a vault today: an edited note leaves its old
version in memory, and a deleted note is never forgotten. Here a note's id is its path in the
vault, so an edit replaces the old document and a note that left the vault is forgotten on the
next sync.

## Requirements

Bundled with the Cognee SDK, no extra install. There is no account, no token and no network
access: the connector reads files from a folder on the machine that runs it. It parses
frontmatter with PyYAML, which dlt already brings in.

**SDK only.** The connector has no entry in the integration provider table
(`cognee/modules/integrations/ingestion.py`). That table serves hosts that hold credentials for
a provider's API, and nothing hosted can reach a vault on your disk.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import obsidian_source

source = obsidian_source(vault_path="~/Documents/Vault")
await cognee.remember(
    source,
    dataset_name="my_vault",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
    self_improvement=False,
)

# A normal return can still have skipped notes: read the counts.
assert source.cognee_sync_stats["failed"] == 0
```

Run the same call again whenever you want to sync.

| Argument | Meaning | Default |
|---|---|---|
| `vault_path` | The vault folder, `~` expanded and resolved. A `ValueError` unless it is an existing directory | required |
| `resource_name` | Names this vault: it keys the staging table, the sync state and the cleanup of deleted notes | `obsidian_<12 hex of the sha256 of the resolved vault path>` |
| `folders` | Vault-relative folders to read, with `/`, subfolders included. `None` is the whole vault | `None` |
| `check_active` | Host checkpoint, called before each note is read and before each row is published | none |

All arguments are keyword-only.

`folders` is a list or tuple of folder paths like `["Projects/Work", "Journal"]`. It is checked
when the source is built: a bare string, an empty list, an absolute path, a `..`, a folder
whose name starts with a dot (never read), a backslash and a path with leading or trailing
spaces are a `ValueError`, because each would select nothing. It is sorted and deduplicated, so
the order does not matter. At run time a selected folder that exists on disk but that the walk
did not find under exactly that name (another letter case on a case-insensitive file system,
another Unicode form, a symbolic link) raises `ObsidianListingError` before anything is
published or forgotten. A selected folder that is a file raises the same error, saying so. A
selected folder that does not exist on disk at all is a removed folder: its notes are
forgotten. Once it has no notes left (a typo, or a folder deleted on purpose and already
forgotten) it selects nothing, and every run logs one warning naming the folder, until you remove
it from `folders`.

**The default `resource_name` follows the vault path.** Moving or renaming the vault folder
changes it, and the next sync then ingests every note again under the new name, while the
documents synced under the old name stay until you forget them. Pass a `resource_name` to pin it.
Two vaults in one dataset need different names (the default already differs).

**One vault, one source per dataset.** The default name comes from the vault path alone, not
from `folders`, so two sources over the same vault with different `folders` share one table and
one state and undo each other: each run forgets the notes the other selected. Put two selections
of one vault in two datasets, or give each source its own pinned `resource_name`. With pinned
names, narrowing one selection no longer forgets notes across them.

After a run, `source.cognee_sync_stats` holds counts only: `synced` (notes read and published),
`deleted` (tombstones), `failed` and `failed_read` (notes that could not be read; the two are
equal today). An empty dictionary means dlt resumed a pending load without extracting this
resource: build a fresh source and run again.

## What is read

Every `*.md` file under the vault, or under the selected folders. A note becomes one row:

| Field | Value |
|---|---|
| id | `obsidian:<path of the note in the vault, with />` (the dlt primary key) |
| title | the frontmatter `title` when it is a non-empty string, else the file name without `.md` |
| content | the file text exactly as read, frontmatter included |
| url | empty (a note has no link that works outside your machine) |
| cognee_node_set | the frontmatter tags, see below. Only when there are any |
| cognee_structure | the note's folders, see below. Only for a note inside a folder |

Only regular `*.md` files are read. Not read: anything whose name starts with a dot, so the
directories `.obsidian`, `.trash` and `.git` and hidden files such as `._note.md` (Obsidian does
not show them either), every file that is not `*.md`, symbolic links (a linked note is skipped,
a linked folder is not entered), and FIFOs and device files. **Attachments are not read in this
version**: images, PDFs and other files are ignored, and a note's embed or link to one stays as
the text you wrote. `[[wikilinks]]` are text too, they do not become edges.

The file is stored as read. cognee turns it into `# <title>` followed by the content, and strips
the ends of that text, so a note's trailing newline is not kept.

**Frontmatter** is the YAML between a `---` line at the very top of the file and the next `---`
line. It is read only for `title` and tags. If it cannot be parsed, is not a YAML mapping, or is
not closed, the note still syncs: no title (the file name is used), no tags, text unchanged.

**Tags** come from `tags`, or the older `tag` key, as a list or as a single string. Each becomes
the node set `obsidian:tag:<tag>`, with a leading `#` dropped, sorted and without repeats. A
single string is one tag, it is not split on commas: write `tags: [work, idea]`. Entries that
are not strings (a number) are skipped. **Inline `#tags` in the body are not read** in this
version, and neither are `aliases` or other properties. A node set is part of the row's
identity, so adding, removing or renaming a tag re-ingests that note (its text changed as well).
`recall(..., node_name=["obsidian:tag:work"])` scopes a search to the tagged notes.

**Structure** is `{"ancestors": [...]}`, nearest folder first, each as
`{"kind": "folder", "id": "<folder path>", "name": "<folder name>"}`. A note at the top of the
vault has no ancestors and no column. Folders are containers without a row of their own. Where a
note sits is not part of its text, and cognee does not hash it. After cognify each note has a
`child_of` edge to its folder and each folder to its parent.

## Incremental model

There is no cursor. Every run walks the whole selection in sorted order, reads every note and
publishes a row for each one. A note whose file text did not change makes the same row, cognee
recognises it by its content hash, and nothing is extracted again, so a quiet sync costs the
reads and no LLM calls. File times are never used: touching a note changes nothing.

Rows contain no timestamps and every list is sorted, so the same vault gives the same rows
byte for byte.

The sync state is kept in `dlt.current.resource_state()` and holds the ids of the notes the last
complete run published (`members`, sorted) and a format version (`v`). It is written after the
last row, so a run that raises keeps the old state. It holds paths and no note text.

## Deletion and its gaps

A note is forgotten when a complete listing no longer contains a note that an earlier run
published. That covers three cases, all handled the same way:

- the note was deleted in the vault,
- the note was **renamed or moved**: its id is its path, so it is a new document and the old one
  is forgotten (renaming a folder does this to every note in it),
- the note fell outside `folders` after you narrowed the selection. Widening the selection
  ingests the notes that were left out.

The listing is complete before the first row is published. If any part of it fails, nothing is
forgotten:

- If `vault_path` is not a directory when the run starts (a folder that was moved or removed, a
  drive that is gone), the run raises `ObsidianVaultUnavailableError` before it reads anything
  or touches the state.
- If the vault root has no entries at all (not even a hidden `.obsidian` folder) and an earlier
  run published notes, the run raises `ObsidianVaultUnavailableError` the same way. A drive or
  bind mount that did not come up is an existing empty directory on Linux, and looks exactly
  like a vault whose notes were all deleted. To forget everything on purpose, leave one file or
  folder in the vault. A root with no entries and nothing published before syncs normally.
- If a folder inside the vault cannot be listed (permissions), the run raises
  `ObsidianListingError` naming the folder relative to the vault.
- A note that cannot be read or is not valid UTF-8 is not published and not forgotten. If an
  earlier run published it, it stays a member and keeps its document and graph content. It is
  counted in `failed_read` and `failed`, and logged with its relative path. Fix the file and the
  next run picks it up.

Gaps:

- **A vault with entries but no notes forgets everything.** A vault that still has at least one
  entry (any file or folder, hidden ones included) but no notes, or whose selected folders have
  none, forgets every note it published before. A root with no entries at all is refused when
  there is something to forget, because a drive that did not mount looks exactly like this. A
  selected folder that does not exist on disk forgets its notes the same way, with no error,
  because a folder that was deleted on purpose must forget them. If a typo, or a folder that no
  longer exists, is the only selected folder (or every selected folder is missing), every note
  synced earlier is outside the selection and is forgotten, with one warning per missing folder.
  A folder name that does exist under another spelling on disk raises instead (see Usage).
- The Unicode form of a file name (NFC or NFD) is part of the note id. A vault moved between
  file systems by a sync tool that changes the form ingests every note again and forgets the
  old ones.
- A note that is edited while a run reads it is stored as read, and the next run replaces it.
- Only one active sync per vault and dataset is supported.
- A very large vault is read in full on every run. Memory use is one note at a time.
- `cognee.forget(data_id=...)` of a note that is still in the vault is undone by the next sync
  (see Privacy).

## Privacy

This connector reads personal notes. Nothing is read until you construct a source and call
`remember`.

**Where it goes**

- The note text becomes cognee documents, frontmatter included. During cognify it is sent to
  your configured LLM and embedding providers, like any document you add to cognee. Check that
  you may send your notes there, or use local models. Anyone with read access to the target
  dataset can read them, so keep a vault in its own dataset, and use `folders` to leave out what
  should not leave the folder.
- `.obsidian`, `.trash`, `.git` and every other hidden file or folder are never read. A symbolic
  link inside the vault is not followed.
- dlt state stores the note ids, which are paths. dlt's staging database for the dataset
  (`dlt_database_<dataset>`) holds the text of every note.
- Logs, errors and `cognee_sync_stats` carry counts, classes and paths relative to the vault,
  never an absolute path or note text.

**What forget removes and what it does not**

- `cognee.forget(data_id=...)` or `forget(dataset=...)` removes the documents, graph and
  vectors, but **not** the staged copy of the text or the dlt state, so the next sync under the
  same dataset and `resource_name` builds everything again. Permanent erasure today means
  deleting the note in the vault or taking its folder out of `folders`, then syncing.
- dlt also keeps completed load packages, with the text of every row, under `~/.dlt` of the
  process, outside cognee's data directories. A note that was deleted in the vault leaves the
  graph but not that copy. This is a core gap, not specific to Obsidian.

## Graph shape

Notes are plain text documents with a position in the vault. Entities with the same name merge
across notes, so a person named in two notes is one node. People and places named in a note are
text, not typed entities. Tags are node sets (`obsidian:tag:<tag>`), and folders are
`StructureContainer` nodes linked by `child_of`.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_obsidian_source.py \
  cognee/tests/integration/tasks/connectors/test_obsidian_ingestion.py \
  cognee/tests/integration/tasks/connectors/test_obsidian_forget.py
```

No Obsidian install is needed: the tests build a vault of plain files in a temporary folder.
The unit tests drive the walk with a dict as state and the resource through a real dlt pipeline
on SQLite. The integration tests run the real cognee pipeline, and `test_obsidian_forget.py` goes
through `remember()` with the LLM and embeddings mocked, checking the graph after an edit, a
deletion, a rename and a narrowed selection.
