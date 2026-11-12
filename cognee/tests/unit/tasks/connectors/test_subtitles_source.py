"""The subtitles source parses SRT and WebVTT into readable documents, and forgets safely.

No account and no network: fixture files are written into a temporary folder. The walker is
driven with a plain dict for state, like the Obsidian source's tests.
"""

from pathlib import Path

import pytest

from cognee.tasks.ingestion.connectors import subtitles as subtitles_module
from cognee.tasks.ingestion.connectors.subtitles import (
    SubtitlesParseError,
    SubtitlesSourceUnavailableError,
    _paragraphs,
    _parse_transcript,
    _strip_tags,
)
from cognee.tasks.ingestion.dlt_utils import STRUCTURE_COLUMN

EMPTY_STATS = {"synced": 0, "deleted": 0}

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello, and welcome to the demo.

2
00:00:04,000 --> 00:00:06,000
Today we talk about memory.

3
00:00:40,000 --> 00:00:42,000
Second topic starts here.
"""

VTT = """WEBVTT - Meeting recording

NOTE
This note is not speech.

STYLE
::cue { color: red }

introduction
00:00:01.000 --> 00:00:03.500 line:0 position:50%
<v Ada>Hello, and welcome to the demo.

00:00:02.000 --> 00:00:04.000
<v Ada>Second sentence of the same turn.

00:00:30.000 --> 00:00:32.000
<v Bob (upset)>I disagree about the <i>checkout</i> flow.
"""


def _write(folder: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return folder


def _rows(folder: Path, state=None, stats=None, min_gap=3.0):
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(subtitles_module._iter_rows(folder, {} if state is None else state, stats, min_gap))


# --- parsing ------------------------------------------------------------------


def test_srt_parses_cues_without_the_noise():
    cues = _parse_transcript(SRT, ".srt")
    assert [(c.start, c.end, c.speaker, c.text) for c in cues] == [
        (1.0, 3.5, None, "Hello, and welcome to the demo."),
        (4.0, 6.0, None, "Today we talk about memory."),
        (40.0, 42.0, None, "Second topic starts here."),
    ]


def test_vtt_skips_note_and_style_blocks_and_cue_settings():
    cues = _parse_transcript(VTT, ".vtt")
    assert len(cues) == 3
    assert cues[0].speaker == "Ada" and cues[0].text == "Hello, and welcome to the demo."
    # The annotation in <v Bob (upset)> is not part of who spoke; the <i> tag is gone.
    assert cues[2].speaker == "Bob"
    assert cues[2].text == "I disagree about the checkout flow."


def test_vtt_accepts_minutes_only_and_srt_comma_timings():
    cues = _parse_transcript("WEBVTT\n\n00:05.500 --> 00:07.000\nShort talk.\n", ".vtt")
    assert cues[0].start == 5.5 and cues[0].end == 7.0
    cues = _parse_transcript("1\n01:00:02,250 --> 01:00:04,000\nLong one.\n", ".srt")
    assert cues[0].start == 3602.25


def test_a_bom_and_crlf_file_parses():
    cues = _parse_transcript("﻿1\n00:00:01,000 --> 00:00:02,000\r\nOne line.\r\n", ".srt")
    assert cues[0].text == "One line."


def test_inline_timestamp_and_formatting_tags_are_stripped():
    speaker, text = _strip_tags("<v Ada><00:00:01.000> Hi <c.lead>there</c> now</v>")
    assert speaker == "Ada" and text == "Hi there now"


@pytest.mark.parametrize(
    "text",
    [
        "1\n00:00:01,000 --> banana\nText.\n",  # unparseable timing
        "1\nno arrow here\nText.\n",  # a block with no timing line
        "1\n00:00:03,000 --> 00:00:01,000\nBackwards.\n",  # ends before it starts
        "WEBVTT\n\njust a header\n",  # no cues at all
    ],
)
def test_malformed_files_fail_loudly(text):
    with pytest.raises(SubtitlesParseError):
        _parse_transcript(text, ".srt")


# --- merging --------------------------------------------------------------------


def test_consecutive_cues_merge_within_the_gap_and_split_on_speaker():
    cues = _parse_transcript(VTT, ".vtt")
    paragraphs = _paragraphs(cues, min_gap=3.0)
    # Ada's two cues are 1.5s apart and share a speaker; Bob starts a new paragraph.
    assert paragraphs == [
        "[00:00:01] Ada: Hello, and welcome to the demo. Second sentence of the same turn.",
        "[00:00:30] Bob: I disagree about the checkout flow.",
    ]


def test_silence_beyond_the_gap_starts_a_paragraph():
    cues = _parse_transcript(SRT, ".srt")
    paragraphs = _paragraphs(cues, min_gap=3.0)
    assert paragraphs == [
        "[00:00:01] Hello, and welcome to the demo. Today we talk about memory.",
        "[00:00:40] Second topic starts here.",
    ]
    # A wider gap threshold joins them all.
    assert len(_paragraphs(cues, min_gap=60.0)) == 1


# --- the source -----------------------------------------------------------------


def test_the_folder_yields_one_document_per_file_with_folder_structure(tmp_path):
    folder = _write(
        tmp_path / "talks", {"a.srt": SRT, "day2/b.vtt": VTT, "notes.md": "not a transcript"}
    )
    state: dict = {}

    rows = _rows(folder, state)

    assert [row["title"] for row in rows] == ["a", "b"]
    assert rows[0]["content"].startswith("[00:00:01]")
    assert rows[0]["id"].startswith("subtitles:a.srt:")
    assert rows[0]["url"] == "" and rows[0]["_deleted"] is False
    # A file at the top of the folder has no position to record; the .md file is not a
    # transcript and is not read.
    assert STRUCTURE_COLUMN not in rows[0]
    assert rows[1][STRUCTURE_COLUMN] == {
        "ancestors": [{"kind": "folder", "id": "day2", "name": "day2"}]
    }
    assert state == {
        "v": 1,
        "members": [row["id"] for row in rows],
    }


def test_a_file_at_the_top_of_the_folder_has_no_ancestors(tmp_path):
    folder = _write(tmp_path / "talks", {"a.srt": SRT})
    rows = _rows(folder)
    assert STRUCTURE_COLUMN not in rows[0]


def test_a_removed_file_is_tombstoned_and_an_unchanged_one_keeps_its_id(tmp_path):
    folder = _write(tmp_path / "talks", {"a.srt": SRT, "b.vtt": VTT})
    state: dict = {}
    first = _rows(folder, state)
    (folder / "b.vtt").unlink()

    second = _rows(folder, state)

    assert second[0]["id"] == first[0]["id"], "an unchanged file keeps its id"
    assert second[1] == {"id": first[1]["id"], "_deleted": True}


def test_an_edited_file_gets_a_new_id_and_forges_the_old_one(tmp_path):
    folder = _write(tmp_path / "talks", {"a.srt": SRT})
    state: dict = {}
    first = _rows(folder, state)
    _write(folder, {"a.srt": SRT + "\n4\n00:01:00,000 --> 00:01:02,000\nOne more thing.\n"})

    second = _rows(folder, state)

    assert second[0]["id"] != first[0]["id"]
    assert {row["id"] for row in second[1:]} == {first[0]["id"]}


def test_a_malformed_file_stops_the_run_before_anything_is_forgotten(tmp_path):
    folder = _write(tmp_path / "talks", {"a.srt": SRT, "b.vtt": VTT})
    state: dict = {}
    first = _rows(folder, state)
    # The malformed file is sorted before b.vtt, and nothing may be published at all.
    _write(folder, {"0-broken.vtt": "WEBVTT\n\nnot a cue\n"})

    with pytest.raises(SubtitlesParseError):
        _rows(folder, state)

    assert state["members"] == [row["id"] for row in first]


def test_a_utf16_export_is_refused_as_text_not_read_as_mojibake(tmp_path):
    folder = tmp_path / "talks"
    folder.mkdir()
    (folder / "utf16.vtt").write_bytes(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi.\n".encode("utf-16")
    )

    with pytest.raises(SubtitlesParseError, match="UTF-16"):
        _rows(folder)


def test_a_missing_folder_never_forgets(tmp_path):
    state: dict = {"v": 1, "members": ["subtitles:a.srt:abc"]}

    with pytest.raises(SubtitlesSourceUnavailableError, match="not available"):
        _rows(tmp_path / "gone", state)


def test_an_empty_folder_that_had_files_never_forgets(tmp_path):
    folder = tmp_path / "talks"
    folder.mkdir()
    state: dict = {"v": 1, "members": ["subtitles:a.srt:abc"]}

    with pytest.raises(SubtitlesSourceUnavailableError, match="empty"):
        _rows(folder, state)


def test_an_empty_folder_that_never_had_files_syncs_nothing(tmp_path):
    folder = tmp_path / "talks"
    folder.mkdir()
    stats = dict(EMPTY_STATS)

    assert _rows(folder, stats=stats) == []
    assert stats == {"synced": 0, "deleted": 0}


# --- through a real dlt pipeline ----------------------------------------------------------


def test_sync_edit_and_delete_through_a_real_dlt_pipeline(tmp_path):
    import dlt

    from cognee.tasks.ingestion.connectors import subtitles_source
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_COLUMN_HINTS

    folder = _write(tmp_path / "talks", {"a.srt": SRT, "b.vtt": VTT})
    table = "subtitles_test"

    def pipeline():
        return dlt.pipeline(
            pipeline_name="subtitles_test",
            destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'a.db'}"),
            dataset_name="subtitles",
            pipelines_dir=str(tmp_path / "pipelines"),
        )

    def run(source) -> tuple[dict[str, bool], list[str]]:
        pipeline().run(source, columns=DOCUMENT_COLUMN_HINTS)
        with (
            pipeline().sql_client() as sql,
            sql.execute_query(f'SELECT id, "_deleted" FROM "{table}" ORDER BY id') as cursor,
        ):
            rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}, [row[0] for row in rows]

    source = subtitles_source(source_path=folder, resource_name=table)
    stored, ids = run(source)
    assert source.cognee_sync_stats == {"synced": 2, "deleted": 0}
    assert stored == {row["id"]: False for row in _rows(folder)}

    # A quiet run publishes the same rows again and forgets nothing.
    source = subtitles_source(source_path=folder, resource_name=table)
    stored, _ = run(source)
    assert source.cognee_sync_stats == {"synced": 2, "deleted": 0}
    assert stored == {row["id"]: False for row in _rows(folder)}

    # Deleting a file tombstones it; the hard-delete column removes the row.
    (folder / "b.vtt").unlink()
    source = subtitles_source(source_path=folder, resource_name=table)
    stored, _ = run(source)
    assert source.cognee_sync_stats == {"synced": 1, "deleted": 1}
    assert list(stored) == ids[:1]


# --- the source constructor -------------------------------------------------------


def test_source_path_must_be_an_srt_or_vtt_file_or_a_folder(tmp_path):
    from cognee.tasks.ingestion.connectors import subtitles_source

    with pytest.raises(ValueError, match="existing"):
        subtitles_source(source_path=tmp_path / "missing")
    single = _write(tmp_path / "talks", {"a.srt": SRT}) / "a.srt"
    source = subtitles_source(source_path=single)
    assert source.cognee_sync_stats == {}
    with pytest.raises(ValueError, match="\\.srt or \\.vtt"):
        subtitles_source(source_path=_write(tmp_path / "other", {"x.md": "text"}) / "x.md")
    with pytest.raises(ValueError, match="min_gap_seconds"):
        subtitles_source(source_path=tmp_path / "talks", min_gap_seconds=-1)


def test_the_resource_is_named_after_the_folder_and_carries_the_source_tag(tmp_path):
    from cognee.tasks.ingestion.connectors import subtitles_source
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    folder = _write(tmp_path / "talks", {"a.srt": SRT})
    source = subtitles_source(source_path=folder, resource_name="pinned")
    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "subtitles"
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "pinned"
    other = subtitles_source(source_path=folder)
    assert getattr(other, PIPELINE_SCOPE_ATTR) != getattr(source, PIPELINE_SCOPE_ATTR)
