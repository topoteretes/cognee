import os
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata


class AudioLoader(LoaderInterface):
    """
    Core text file loader that handles basic text file formats.

    This loader is always available and serves as the fallback for
    text-based files when no specialized loader is available.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Supported text file extensions."""
        return [
            "aac",  # Audio documents
            "mid",
            "mp3",
            "m4a",
            "ogg",
            "flac",
            "wav",
            "amr",
            "aiff",
        ]

    @property
    def supported_mime_types(self) -> List[str]:
        """Supported MIME types for text content."""
        return [
            "audio/aac",
            "audio/midi",
            "audio/mpeg",
            "audio/mp4",
            "audio/ogg",
            "audio/flac",
            "audio/wav",
            "audio/amr",
            "audio/aiff",
            "audio/x-wav",
        ]

    @property
    def loader_name(self) -> str:
        """Unique identifier for this loader."""
        return "audio_loader"

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

    async def load(self, file_path: str, **kwargs):
        """
        Load and process the audio file.

        Args:
            file_path: Path to the file to load
            **kwargs: Additional configuration (unused)

        Returns:
            LoaderResult containing the file content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
        # Name ingested file of current loader based on original file content hash
        storage_file_name = "text_" + file_metadata["content_hash"] + ".txt"

        result = await LLMGateway.create_transcript(file_path)

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        full_file_path = await storage.store(storage_file_name, result.text)

        return full_file_path
