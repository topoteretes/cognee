# Subtitles SDK connector

A subtitle data-source connector for [cognee](https://github.com/topoteretes/cognee): sync a
folder of `.srt` / `.vtt` transcripts (or one file) into memory as readable documents,
incrementally, with forget-on-delete.

It exposes a `dlt` source you hand to `cognee.remember(...)`, reusing cognee's existing DLT
ingestion path. Each file becomes one markdown document — cue numbers, timing lines and
markup stripped — that goes through normal chunking and LLM graph extraction, so a search
result can point back to the moment in the recording.

This fixes what `cognee.add()` on the raw file does today: timing noise and markup reach the
graph, and a two-second cue splits a sentence in the middle. Here cues are merged into
paragraphs that keep who said what and when.

## Requirements

Bundled with the Cognee SDK, no extra install. There is no account, no token and no network
access: the connector reads files from the machine that runs it, exported by whatever made
the recording.

**SDK only.** The connector has no entry in the integration provider table
(`cognee/modules/integrations/ingestion.py`): nothing hosted can reach a folder on your disk.

## Getting the files

- **Zoom**: record to the cloud, open the recording, download "Audio Transcript" (`.vtt`).
- **Microsoft Teams**: recording transcript downloads as `.vtt` from the meeting's Files tab.
- **Google Meet**: transcript attached to the meeting event as `.doc`/`.vtt` depending on
  workspace settings; `.vtt` works as-is.
- **OBS / local audio**: run Whisper (`whisper talk.wav --output_format srt` or `vtt`),
  Whisper writes the file next to the audio.
- Any other tool that writes SRT or WebVTT (Premiere, DaVinci Resolve, YouTube) works.

## Usage

```python
import cognee
from cognee.tasks.ingestion.connectors import subtitles_source

source = subtitles_source(source_path="~/Recordings/transcripts")
await cognee.remember(
    source,
    dataset_name="meeting_transcripts",
    primary_key="id",
    write_disposition="merge",  # REQUIRED, the add pipeline defaults to "replace"
    max_rows_per_table=0,
    self_improvement=False,
)

# Counts only: files published and files forgotten by this run.
assert source.cognee_sync_stats["deleted"] == 0
```

Run the same call again whenever you want to sync. A file whose text did not change makes
the same row id and is not re-cognified.

| Argument | Meaning | Default |
|---|---|---|
| `source_path` | A folder walked recursively, or one `.srt` / `.vtt` file. `ValueError` unless it exists | required |
| `resource_name` | Names this folder: it keys the staging table and the sync state | `subtitles_<12 hex of the sha256 of the resolved folder>` |
| `min_gap_seconds` | Silence between cues under which they merge into one paragraph; a speaker change always starts one. Must be >= 0 | `3.0` |
| `check_active` | Host checkpoint, called before each file is read and before each row is published | none |

All arguments are keyword-only. One folder, one source per dataset: the default
`resource_name` follows the folder path, and two sources over one folder in one dataset
share one table and one state and undo each other. Pass pinned names to keep them apart.

After a run, `source.cognee_sync_stats` holds counts only: `synced` (files published) and
`deleted` (files forgotten). An empty dictionary means dlt resumed a pending load without
extracting this resource: build a fresh source and run again.

## What is read

Every `*.srt` and `*.vtt` file under the folder (or the one file given), case-insensitive on
the suffix. Not read: hidden files and folders, symbolic links, and any other suffix.

**Parsing.** SRT (`HH:MM:SS,mmm`) and WebVTT (`HH:MM:SS.mmm` or `MM:SS.mmm`) timing lines,
with or without cue identifiers / sequence numbers, cue settings after the arrow
(`line:0 position:50%`), and WebVTT `NOTE` / `STYLE` / `REGION` blocks skipped. Inline
markup is stripped: voice tags, `<i>`, `<b>`, `<u>`, `<c>`, `<ruby>`/`<rt>` and inline
timestamps. UTF-8 with or without BOM, CRLF or LF. A UTF-16 export is refused with an error
saying so, not read as mojibake.

**Merging.** Consecutive cues merge into one paragraph while the speaker did not change
and no more than `min_gap_seconds` of silence precede the next cue. The speaker comes from
a WebVTT voice tag (`<v Name>`); an annotation like `<v Name (upset)>` names the speaker
`Name`. Without a voice tag every cue shares the unnamed speaker. Each paragraph renders as
`[HH:MM:SS] Speaker: text`, or `[HH:MM:SS] text` when nothing named a speaker, and leading
dialogue dashes (`- text`) are dropped.

The file becomes one document:

| Field | Value |
|---|---|
| id | `subtitles:<path>:<12 hex of the sha256 of the file text>` (the dlt primary key) |
| title | the file name without its suffix |
| content | the merged paragraphs, blank-line separated |
| url | empty |
| cognee_structure | the file's folders, see below. Only for a file inside a folder |

The content hash is part of the id, so an edited file is a new document and the old one is
forgotten on the same run that adds the new one.

**Structure** is `{"ancestors": [...]}`, nearest folder first, each as
`{"kind": "folder", "id": "<folder path>", "name": "<folder name>"}` — where a file sits,
like the Obsidian source. A file at the top of the folder has no ancestors. After cognify
each document has a `child_of` edge to its folder and each folder to its parent.

## Incremental model

There is no cursor. Every run walks the whole folder, parses every file and publishes a row
for each one. A file whose text did not change makes the same row (the id carries the hash),
cognee recognises it, and nothing is extracted again, so a quiet sync costs the reads and no
LLM calls. File times are never used.

The sync state is kept in `dlt.current.resource_state()` and holds the ids the last complete
run published (`members`, sorted) and a format version (`v`). It is written after the last
row, so a run that raises keeps the old state. It holds paths and hashes, no transcript text.

## Deletion and its gaps

A file is forgotten when a complete listing no longer contains a file that an earlier run
published: deleted, renamed, or outside the folder. The listing and the parse of every file
complete before the first row is published, so a failure forgets nothing:

- If `source_path` is not there when the run starts, the run raises
  `SubtitlesSourceUnavailableError` before anything is published or forgotten.
- If the folder is empty but earlier runs synced files from it, the run raises the same way —
  a drive that did not mount looks exactly like a folder whose files were all deleted. To
  forget everything on purpose, leave one file in the folder.
- A folder that cannot be listed raises `SubtitlesSourceUnavailableError`.
- A file that cannot be parsed (no timing line in a block, an unparseable timing, a cue that
  ends before it starts, no cues at all, or text that is not UTF-8) raises
  `SubtitlesParseError` **before any row is published**, so the run stops with the previous
  sync intact. Fix the file and run again.

Gaps:

- A very large folder is read and parsed in full on every run; memory holds one run's rows.
- `cognee.forget(data_id=...)` of a file still in the folder is undone by the next sync.
- Overlapping speakers (one timing range inside another) are merged by start time, not
  interleaved: cue order in the file decides.

## Privacy

Nothing is read until you construct a source and call `remember`. The transcript text
becomes cognee documents and, during cognify, is sent to your configured LLM and embedding
providers like any document — check that you may send it there, or use local models.
Logs, errors and `cognee_sync_stats` carry counts, classes and folder-relative paths, never
an absolute path or transcript text. dlt's staging database for the dataset
(`dlt_database_<dataset>`) holds the parsed paragraphs, like any dlt source.

## Graph shape

A transcript is one document with `[HH:MM:SS] Speaker: text` paragraphs. Speakers named by
voice tags appear as text in the document (a person named in several files merges into one
node during extraction); folders are `StructureContainer` nodes linked by `child_of`.

## Testing

```bash
uv run pytest cognee/tests/unit/tasks/connectors/test_subtitles_source.py
```

No account and no network: the tests build folders of fixture files in a temporary folder.
