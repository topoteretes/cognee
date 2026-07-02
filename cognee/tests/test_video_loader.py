import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.loaders.core.video_loader import (
    VideoLoader,
    _build_timestamped_text,
    _format_timestamp,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "test_data", "sample_video.mp4")

# A verbose-transcription payload as litellm returns it: segments carrying
# start times and text. Object- and dict-shaped segments are both exercised.
SEGMENTS = [
    {"start": 0.0, "end": 3.0, "text": " Welcome to the demo."},
    {"start": 3.0, "end": 7.5, "text": " This part covers setup."},
    {"start": 83.0, "end": 88.0, "text": " And this wraps up."},
]
EXPECTED_TRANSCRIPT = (
    "[00:00:00] Welcome to the demo.\n"
    "[00:00:03] This part covers setup.\n"
    "[00:01:23] And this wraps up."
)


class _FakePayload:
    def __init__(self, segments):
        self.segments = segments


class _FakeTranscript:
    def __init__(self, text, segments=None):
        self.text = text
        self.payload = _FakePayload(segments) if segments is not None else object()


@pytest.fixture
def loader():
    return VideoLoader()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extension, mime_type, expected",
    [
        ("mp4", "video/mp4", True),
        ("m4v", "video/x-m4v", True),
        ("mov", "video/quicktime", True),
        ("webm", "video/webm", True),
        ("mkv", "video/x-matroska", True),
        ("avi", "video/x-msvideo", True),
        # Audio must not be captured by the video loader.
        ("mp3", "audio/mpeg", False),
        # Extension right, MIME wrong (and vice versa) must not match.
        ("mp4", "text/plain", False),
        ("txt", "video/mp4", False),
    ],
)
def test_can_handle(loader, extension, mime_type, expected):
    assert loader.can_handle(extension, mime_type) == expected


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_format_timestamp():
    assert _format_timestamp(0) == "00:00:00"
    assert _format_timestamp(83) == "00:01:23"
    assert _format_timestamp(3661.9) == "01:01:01"


def test_build_timestamped_text_dicts():
    assert _build_timestamped_text(SEGMENTS) == EXPECTED_TRANSCRIPT


def test_build_timestamped_text_objects():
    class Seg:
        def __init__(self, start, text):
            self.start = start
            self.text = text

    segments = [Seg(0.0, " Hello"), Seg(5.0, " World")]
    assert _build_timestamped_text(segments) == "[00:00:00] Hello\n[00:00:05] World"


def test_build_timestamped_text_skips_blank_segments():
    assert _build_timestamped_text([{"start": 1.0, "text": "   "}]) == ""


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("cognee.infrastructure.loaders.core.video_loader.get_file_storage")
@patch("cognee.infrastructure.loaders.core.video_loader.get_storage_config")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.LLMGateway.create_transcript",
    new_callable=AsyncMock,
)
@patch.object(VideoLoader, "_extract_audio", new_callable=AsyncMock)
@patch("cognee.infrastructure.loaders.core.video_loader._resolve_ffmpeg")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.get_file_metadata",
    new_callable=AsyncMock,
)
async def test_load_persists_timestamped_transcript(
    mock_metadata,
    mock_ffmpeg,
    mock_extract,
    mock_transcript,
    mock_storage_config,
    mock_get_storage,
    loader,
):
    """ffmpeg present: extract audio, transcribe with segments, store transcript."""
    mock_metadata.return_value = {"content_hash": "hash123", "extension": "mp4"}
    mock_ffmpeg.return_value = "/usr/bin/ffmpeg"
    mock_extract.return_value = "/tmp/extracted.wav"
    mock_transcript.return_value = _FakeTranscript("plain", segments=SEGMENTS)

    mock_storage_config.return_value = {"data_root_directory": "/fake/root"}
    storage_instance = MagicMock()
    storage_instance.store = AsyncMock(return_value="/fake/root/text_hash123.txt")
    mock_get_storage.return_value = storage_instance

    with patch("cognee.infrastructure.loaders.core.video_loader.os.remove"):
        result = await loader.load(FIXTURE_PATH)

    assert result == "/fake/root/text_hash123.txt"
    mock_extract.assert_awaited_once()
    storage_instance.store.assert_awaited_once_with("text_hash123.txt", EXPECTED_TRANSCRIPT)


@pytest.mark.asyncio
@patch(
    "cognee.infrastructure.loaders.core.video_loader.LLMGateway.create_transcript",
    new_callable=AsyncMock,
)
@patch.object(VideoLoader, "_extract_audio", new_callable=AsyncMock)
@patch("cognee.infrastructure.loaders.core.video_loader._resolve_ffmpeg")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.get_file_metadata",
    new_callable=AsyncMock,
)
async def test_load_persist_false_returns_text(
    mock_metadata, mock_ffmpeg, mock_extract, mock_transcript, loader
):
    """persist=False returns the raw timestamped transcript instead of a path."""
    mock_metadata.return_value = {"content_hash": "hash123", "extension": "mp4"}
    mock_ffmpeg.return_value = "/usr/bin/ffmpeg"
    mock_extract.return_value = "/tmp/extracted.wav"
    mock_transcript.return_value = _FakeTranscript("plain", segments=SEGMENTS)

    with patch("cognee.infrastructure.loaders.core.video_loader.os.remove"):
        result = await loader.load(FIXTURE_PATH, persist=False)

    assert result == EXPECTED_TRANSCRIPT


@pytest.mark.asyncio
@patch(
    "cognee.infrastructure.loaders.core.video_loader.LLMGateway.create_transcript",
    new_callable=AsyncMock,
)
@patch.object(VideoLoader, "_extract_audio", new_callable=AsyncMock)
@patch("cognee.infrastructure.loaders.core.video_loader._resolve_ffmpeg")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.get_file_metadata",
    new_callable=AsyncMock,
)
async def test_load_without_ffmpeg_transcribes_mp4_directly(
    mock_metadata, mock_ffmpeg, mock_extract, mock_transcript, loader
):
    """No ffmpeg: mp4 goes straight to transcription, no extraction attempted."""
    mock_metadata.return_value = {"content_hash": "hash123", "extension": "mp4"}
    mock_ffmpeg.return_value = None
    mock_transcript.return_value = _FakeTranscript("plain", segments=SEGMENTS)

    result = await loader.load(FIXTURE_PATH, persist=False)

    assert result == EXPECTED_TRANSCRIPT
    mock_extract.assert_not_awaited()
    # The original container path is what gets transcribed.
    assert mock_transcript.call_args.args[0] == FIXTURE_PATH


@pytest.mark.asyncio
@patch("cognee.infrastructure.loaders.core.video_loader._resolve_ffmpeg")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.get_file_metadata",
    new_callable=AsyncMock,
)
async def test_load_without_ffmpeg_rejects_mov(mock_metadata, mock_ffmpeg, loader):
    """No ffmpeg: a container that needs extraction raises an actionable error."""
    mock_metadata.return_value = {"content_hash": "hash123", "extension": "mov"}
    mock_ffmpeg.return_value = None

    with pytest.raises(RuntimeError, match="ffmpeg"):
        await loader.load(FIXTURE_PATH, persist=False)


@pytest.mark.asyncio
@patch(
    "cognee.infrastructure.loaders.core.video_loader.LLMGateway.create_transcript",
    new_callable=AsyncMock,
)
@patch.object(VideoLoader, "_extract_audio", new_callable=AsyncMock)
@patch("cognee.infrastructure.loaders.core.video_loader._resolve_ffmpeg")
@patch(
    "cognee.infrastructure.loaders.core.video_loader.get_file_metadata",
    new_callable=AsyncMock,
)
async def test_load_falls_back_to_plain_transcript(
    mock_metadata, mock_ffmpeg, mock_extract, mock_transcript, loader
):
    """When segmented transcription is unsupported, fall back to plain text."""
    mock_metadata.return_value = {"content_hash": "hash123", "extension": "mp4"}
    mock_ffmpeg.return_value = "/usr/bin/ffmpeg"
    mock_extract.return_value = "/tmp/extracted.wav"
    # First (verbose) call raises; second (plain) call returns text without segments.
    mock_transcript.side_effect = [
        RuntimeError("verbose_json not supported"),
        _FakeTranscript("plain transcript text"),
    ]

    with patch("cognee.infrastructure.loaders.core.video_loader.os.remove"):
        result = await loader.load(FIXTURE_PATH, persist=False)

    assert result == "plain transcript text"
    assert mock_transcript.await_count == 2


@pytest.mark.asyncio
async def test_load_missing_file_raises(loader):
    with pytest.raises(FileNotFoundError):
        await loader.load("/nonexistent/video.mp4")
