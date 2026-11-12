# Kindle SDK connector

A Kindle highlights connector for [cognee](https://github.com/topoteretes/cognee): sync the
clippings file of a Kindle (`My Clippings.txt`) into memory as one markdown document per
book, incrementally, with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Each book becomes one document — every highlight and note in location
order, with page, location and date — that goes through normal chunking and LLM graph
extraction, so "what have I highlighted about habits across all my books?" is one search.

There is no account, no token and no network access: the connector reads the file the
Kindle itself writes. (The Readwise route needs a paid account; Amazon's web notebook
needs a login and breaks often.)

## Getting the file

On the Kindle it lives at `documents/My Clippings.txt`. Copy it off the device over USB, or
sync it off the mounted drive — the connector accepts the file directly, or a folder that
contains it (the mounted Kindle's root or its `documents` folder).

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import kindle_source

source = kindle_source(clippings_path="~/kindle/documents/My Clippings.txt")
await cognee.remember(
    source,
    dataset_name="kindle_highlights",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
    self_improvement=False,
)

# Counts only: books published and books forgotten by this run.
assert source.cognee_sync_stats["deleted"] == 0
```

Run the same call again whenever you want to sync. A book whose highlights did not change
makes the same row id and is not re-cognified.

| Argument | Meaning | Default |
|---|---|---|
| `clippings_path` | The `My Clippings.txt` file, or a folder containing it. `ValueError` when no clippings file is found | required |
| `resource_name` | Names this clippings file: it keys the staging table and the sync state | `kindle_<12 hex of the sha256 of the resolved path>` |
| `check_active` | Host checkpoint, called before the file is read and before each row is published | none |

All arguments are keyword-only. After a run, `source.cognee_sync_stats` holds counts only:
`synced` (books published) and `deleted` (books forgotten).

## Entry format

Entries are separated by a line of ten or more `=` signs. Each entry is a title/author
line (`Atomic Habits (James Clear)` — the author is optional), a metadata line
(`- Your Highlight on page 55 | location 558-560 | Added on Tuesday, October 12, 2021
3:45:00 PM`), and the clipped text. The page and the location are optional in Kindle's
variants; the type (`Highlight`, `Note`, `Bookmark`) is not.

## What is read

- **One document per book**, title as the document title, the author as an `Author:` line.
- **Highlights** render as `> text (location 558-560, page 55, 2021-10-12)` — whichever of
  the three the entry carries — in location order.
- **Notes** attach to the highlight whose location range contains them, rendered as
  `>> Note: text` under it; a note with no containing highlight stands on its own line.
- **Bookmarks** have no text and are skipped.
- An unchanged book keeps a stable content-hash id, so a quiet sync costs one file read and
  no LLM calls.

**Extended highlights.** When you extend a highlight on the Kindle, it appends a new entry
instead of updating the old one. Where location ranges of the same type overlap in one
book, only the newest entry (by its `Added on` timestamp, file order breaking a tie) is
kept, so the document holds the current text and not a growing pile of overlapping versions.

## Incremental model

There is no cursor. Every run parses the whole file and publishes one row per book. The id
is the hash of the book's title, author and rendered body, so an edited highlight gives the
book a new id and the old version is forgotten on the same run that adds the new one.

The sync state is kept in `dlt.current.resource_state()` and holds the ids the last
complete run published (`members`, sorted) and a format version (`v`). It is written after
the last row, so a run that raises keeps the old state.

## Deletion and its gaps

- A book whose entries all left the file is tombstoned on the next run, the last book
  included: a clippings file emptied on purpose forgets everything.
- The whole file is parsed and every row built **before the first row is published**, so a
  malformed entry (no title line, no metadata line, an unparseable `Added on` timestamp)
  raises `KindleClippingsParseError` with nothing published and nothing forgotten. Fix the
  file and run again.
- If the file is not there when the run starts (the Kindle did not mount), the run raises
  `KindleClippingsUnavailableError` the same way.
- Gaps: non-English export formats (the metadata line is matched in English, as Kindle
  writes it; a localized export fails with a clear error rather than parsing wrong), and
  DRM-protected book content is not part of the clippings file anyway. Notes are attached
  to highlights by location containment; a note taken across a range boundary stands alone.

## Privacy

Nothing is read until you construct a source and call `remember`. Highlight text becomes
cognee documents and, during cognify, is sent to your configured LLM and embedding
providers like any document — check that you may send it there, or use local models. Logs,
errors and `cognee_sync_stats` carry counts and classes, never clipped text. dlt's staging
database for the dataset (`dlt_database_<dataset>`) holds the parsed paragraphs, like any
dlt source.

## Graph shape

A book is one document; authors are text on the document (an author of several books
merges into one node during extraction). Locations and dates are text on the highlight
lines, so a time-scoped question ("what did I highlight last October?") can be answered
from the document text.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_kindle_source.py
```

No Kindle needed: the tests build clippings files in a temporary folder.
