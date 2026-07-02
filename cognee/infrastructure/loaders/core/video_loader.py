from typing import Any

from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface


class VideoLoader(LoaderInterface):
    """
    Core video file loader.

    Turns a video file into text so it can flow through the normal
    ingestion pipeline. The audio track is transcribed with
    ``LLMGateway.create_transcript`` (Solution 1); keyframe captioning via
    ``LLMGateway.transcribe_image`` is layered on in a follow-up.

    Registered as a core loader so video is always recognized. Extraction
    that needs ffmpeg (containers OpenAI's transcription endpoint does not
    accept directly) checks for ffmpeg at load time and fails with an
    actionable message when it is missing.
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
            **kwargs: Additional configuration

        Returns:
            Path to the stored transcript text file, or the raw text when
            ``persist=False`` is passed.

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        raise NotImplementedError(
            "VideoLoader.load is implemented in the following commit "
            "(audio-track transcription via create_transcript)."
        )
