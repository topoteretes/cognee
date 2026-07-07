import asyncio
import os
import shutil
import subprocess
import tempfile
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Video containers whose audio the OpenAI transcription endpoint accepts
# directly, so no local audio extraction is required when ffmpeg is absent.
DIRECT_TRANSCRIBE_EXTENSIONS = {"mp4", "webm"}


def _resolve_ffmpeg() -> str | None:
    """Return a usable ffmpeg executable path, or None if unavailable.

    Prefers a system ffmpeg on PATH, then falls back to the static binary
    bundled with ``imageio-ffmpeg`` when that optional package is installed.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _format_timestamp(seconds: float) -> str:
    """Format a number of seconds as ``HH:MM:SS``."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_segments(payload: Any) -> list:
    """Pull the timed segments out of a transcription payload, if present."""
    if payload is None:
        return []
    segments = getattr(payload, "segments", None)
    if segments is None and isinstance(payload, dict):
        segments = payload.get("segments")
    return segments or []


def _segment_field(segment: Any, field: str) -> Any:
    """Read a field from a segment that may be a dict or an object."""
    if isinstance(segment, dict):
        return segment.get(field)
    return getattr(segment, field, None)


def _build_timestamped_text(segments: list) -> str:
    """Render segments as ``[HH:MM:SS] text`` lines, one per segment."""
    lines = []
    for segment in segments:
        text = (_segment_field(segment, "text") or "").strip()
        if not text:
            continue
        start = _segment_field(segment, "start") or 0.0
        lines.append(f"[{_format_timestamp(start)}] {text}")
    return "\n".join(lines)


class VideoLoader(LoaderInterface):
    """
    Core video file loader.

    Turns a video file into text so it can flow through the normal ingestion
    pipeline. The audio track is transcribed with
    ``LLMGateway.create_transcript`` and the transcript is written out with
    per-segment ``[HH:MM:SS]`` timestamps inlined, so the timing survives
    chunking and stays searchable (Solution 1). Keyframe captioning via
    ``LLMGateway.transcribe_image`` layers on in a follow-up.

    When ffmpeg is available the audio track is extracted locally first,
    which works for every container and keeps the upload small. When ffmpeg
    is absent the loader still handles ``mp4`` and ``webm`` by sending the
    container straight to the transcription endpoint; other containers raise
    an actionable error explaining how to enable them.
    """

    loader_name = "video_loader"

    @property
    def supported_extensions(self) -> list[str]:
        """Supported video file extensions."""
        return [
            "mp4",
            "m4v",
            "mov",
            "webm",
            "mkv",
            "avi",
        ]

    @property
    def supported_mime_types(self) -> list[str]:
        """Supported MIME types for video content."""
        return [
            "video/mp4",
            "video/x-m4v",
            "video/quicktime",
            "video/webm",
            "video/x-matroska",
            "video/x-msvideo",
        ]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: Optional MIME type

        Returns:
            True if file can be handled, False otherwise
        """
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True
        return False

    async def load(self, file_path: str, **kwargs: Any) -> str:
        """
        Load and process the video file.

        Args:
            file_path: Path to the file to load
            **kwargs: Additional configuration (``persist`` controls whether the
                transcript is stored to disk or returned directly)

        Returns:
            Path to the stored transcript text file, or the raw transcript text
            when ``persist=False`` is passed.

        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If the container needs ffmpeg to extract audio and
                ffmpeg is not available
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
        # Name ingested file of current loader based on original file content hash
        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"
        extension = (file_metadata.get("extension") or "").lower()

        audio_path, temp_audio = await self._prepare_audio(file_path, extension)
        try:
            transcript = await self._transcribe(audio_path)
        finally:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)

        if not kwargs.get("persist", True):
            return transcript

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, transcript)

        return full_file_path

    async def _prepare_audio(self, file_path: str, extension: str) -> tuple[str, str | None]:
        """Return an audio path to transcribe and a temp file to clean up.

        With ffmpeg available, extracts the audio track for every container.
        Without ffmpeg, only ``mp4``/``webm`` are transcribable directly; any
        other container raises an actionable error.
        """
        ffmpeg = _resolve_ffmpeg()
        if ffmpeg is not None:
            temp_audio = await self._extract_audio(ffmpeg, file_path)
            return temp_audio, temp_audio

        if extension in DIRECT_TRANSCRIBE_EXTENSIONS:
            return file_path, None

        raise RuntimeError(
            f"Cannot process a '.{extension}' video without ffmpeg, which is needed to "
            "extract the audio track for transcription. Install ffmpeg and make sure it "
            "is on your PATH (or `pip install imageio-ffmpeg`), or provide the video as "
            ".mp4 or .webm, which can be transcribed directly."
        )

    async def _extract_audio(self, ffmpeg: str, file_path: str) -> str:
        """Extract a mono 16 kHz WAV audio track from the video with ffmpeg."""
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        command = [
            ffmpeg,
            "-y",
            "-i",
            file_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ]

        def run_ffmpeg() -> subprocess.CompletedProcess:
            return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        result = await asyncio.to_thread(run_ffmpeg)
        if result.returncode != 0:
            if os.path.exists(out_path):
                os.remove(out_path)
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to extract audio from '{file_path}': {stderr}")
        return out_path

    async def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio, preferring segmented output for timestamps.

        Requests verbose/segmented transcription so each segment can be
        prefixed with an ``[HH:MM:SS]`` marker. Providers or models that do not
        support segmented output fall back to a plain transcript.
        """
        result = None
        try:
            result = await LLMGateway.create_transcript(
                audio_path,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        except Exception as error:
            logger.warning(
                "Segmented transcription unavailable (%s); falling back to plain transcript.",
                error,
            )

        if result is None:
            plain = await LLMGateway.create_transcript(audio_path)
            return plain.text if plain is not None else ""

        segments = _extract_segments(result.payload)
        if segments:
            return _build_timestamped_text(segments)
        return result.text or ""
