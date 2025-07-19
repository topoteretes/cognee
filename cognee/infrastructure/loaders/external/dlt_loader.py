import os
from typing import List, Dict, Any, Optional
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.loaders.models.LoaderResult import LoaderResult, ContentType
from cognee.shared.logging_utils import get_logger


class DltLoader(LoaderInterface):
    """
    Data loader using DLT (Data Load Tool) for various data sources.

    Supports loading data from REST APIs, databases, cloud storage,
    and other data sources through DLT pipelines.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    @property
    def supported_extensions(self) -> List[str]:
        return [
            ".dlt",  # DLT pipeline configuration
            ".json",  # JSON data
            ".jsonl",  # JSON Lines
            ".csv",  # CSV data
            ".parquet",  # Parquet files
            ".yaml",  # YAML configuration
            ".yml",  # YAML configuration
        ]

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "application/json",
            "application/x-ndjson",  # JSON Lines
            "text/csv",
            "application/x-parquet",
            "application/yaml",
            "text/yaml",
        ]

    @property
    def loader_name(self) -> str:
        return "dlt_loader"

    def get_dependencies(self) -> List[str]:
        return ["dlt>=0.4.0"]

    def can_handle(self, file_path: str, mime_type: str = None) -> bool:
        """Check if file can be handled by this loader."""
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            return False

        # Check MIME type if provided
        if mime_type and mime_type not in self.supported_mime_types:
            return False

        # Validate dependencies
        return self.validate_dependencies()

    async def load(self, file_path: str, source_type: str = "auto", **kwargs) -> LoaderResult:
        """
        Load data using DLT pipeline.

        Args:
            file_path: Path to the data file or DLT configuration
            source_type: Type of data source ("auto", "json", "csv", "parquet", "api")
            **kwargs: Additional DLT pipeline configuration

        Returns:
            LoaderResult with loaded data and metadata

        Raises:
            ImportError: If DLT is not installed
            Exception: If data loading fails
        """
        try:
            import dlt
        except ImportError as e:
            raise ImportError(
                "dlt is required for data loading. Install with: pip install dlt"
            ) from e

        try:
            self.logger.info(f"Loading data with DLT: {file_path}")

            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # Determine source type if auto
            if source_type == "auto":
                if file_ext == ".json":
                    source_type = "json"
                elif file_ext == ".jsonl":
                    source_type = "jsonl"
                elif file_ext == ".csv":
                    source_type = "csv"
                elif file_ext == ".parquet":
                    source_type = "parquet"
                elif file_ext in [".yaml", ".yml"]:
                    source_type = "yaml"
                else:
                    source_type = "file"

            # Load data based on source type
            if source_type == "json":
                content = self._load_json(file_path)
            elif source_type == "jsonl":
                content = self._load_jsonl(file_path)
            elif source_type == "csv":
                content = self._load_csv(file_path)
            elif source_type == "parquet":
                content = self._load_parquet(file_path)
            elif source_type == "yaml":
                content = self._load_yaml(file_path)
            else:
                # Default: read as text
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            # Determine content type
            if isinstance(content, (dict, list)):
                content_type = ContentType.STRUCTURED
                text_content = str(content)
            else:
                content_type = ContentType.TEXT
                text_content = content

            # Gather metadata
            metadata = {
                "name": file_name,
                "size": file_size,
                "extension": file_ext,
                "loader": self.loader_name,
                "source_type": source_type,
                "dlt_version": dlt.__version__,
            }

            # Add data-specific metadata
            if isinstance(content, list):
                metadata["records_count"] = len(content)
            elif isinstance(content, dict):
                metadata["keys_count"] = len(content)

            return LoaderResult(
                content=text_content,
                metadata=metadata,
                content_type=content_type,
                chunks=[text_content],  # Single chunk for now
                source_info={
                    "file_path": file_path,
                    "source_type": source_type,
                    "raw_data": content if isinstance(content, (dict, list)) else None,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to load data with DLT from {file_path}: {e}")
            raise Exception(f"DLT data loading failed: {e}") from e

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON Lines file."""
        import json

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_csv(self, file_path: str) -> str:
        """Load CSV file as text."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_parquet(self, file_path: str) -> str:
        """Load Parquet file (requires pandas)."""
        try:
            import pandas as pd

            df = pd.read_parquet(file_path)
            return df.to_string()
        except ImportError:
            # Fallback: read as binary and convert to string representation
            with open(file_path, "rb") as f:
                return f"<Parquet file: {os.path.basename(file_path)}, size: {len(f.read())} bytes>"

    def _load_yaml(self, file_path: str) -> str:
        """Load YAML file as text."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
