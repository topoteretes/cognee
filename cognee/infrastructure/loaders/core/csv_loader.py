import csv
import io
from typing import List, Dict, Any
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class CsvLoader(LoaderInterface):
    """
    CSV loader that preserves row-column relationships for structured data ingestion.

    This loader reads CSV files and converts them into a structured format where
    each row's values are explicitly associated with their column headers.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Supported CSV file extensions."""
        return ["csv"]

    @property
    def supported_mime_types(self) -> List[str]:
        """Supported MIME types for CSV content."""
        return ["text/csv", "application/csv"]

    @property
    def loader_name(self) -> str:
        """Unique identifier for this loader."""
        return "csv_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """
        Check if this loader can handle the given file.

        Args:
            extension: File extension
            mime_type: MIME type of the file

        Returns:
            True if file can be handled, False otherwise
        """
        return extension in self.supported_extensions and mime_type in self.supported_mime_types

    async def load(
        self,
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        quotechar: str = '"',
        **kwargs,
    ) -> str:
        """
        Load and process the CSV file, preserving row-column relationships.

        Args:
            file_path: Path to the CSV file to load
            encoding: Text encoding to use (default: utf-8)
            delimiter: CSV field delimiter (default: comma)
            quotechar: CSV quote character (default: double quote)
            **kwargs: Additional configuration

        Returns:
            Path to the processed text file containing structured CSV data

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file cannot be decoded with specified encoding
            csv.Error: If CSV parsing fails
            OSError: If file cannot be read
        """
        logger.info(f"Loading CSV file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                file_metadata = await get_file_metadata(f)

            # Name ingested file based on original file content hash
            storage_file_name = "csv_" + file_metadata["content_hash"] + ".txt"

            # Read and process CSV content
            with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                # Detect dialect if not specified
                sample = csvfile.read(1024)
                csvfile.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                    if delimiter == ",":  # Only use detected delimiter if not explicitly set
                        delimiter = dialect.delimiter
                        quotechar = dialect.quotechar
                except csv.Error:
                    # Use defaults if detection fails
                    pass

                reader = csv.DictReader(csvfile, delimiter=delimiter, quotechar=quotechar)

                structured_content = self._process_csv_rows(reader)

            # Store the processed content
            storage_config = get_storage_config()
            data_root_directory = storage_config["data_root_directory"]
            storage = get_file_storage(data_root_directory)

            full_file_path = await storage.store(storage_file_name, structured_content)

            logger.info(
                f"Successfully processed CSV file with {len(reader.fieldnames or [])} columns"
            )
            return full_file_path

        except Exception as e:
            logger.error(f"Failed to process CSV {file_path}: {e}")
            raise Exception(f"CSV processing failed: {e}") from e

    def _process_csv_rows(self, reader: csv.DictReader) -> str:
        """
        Process CSV rows into structured text format preserving column context.

        Args:
            reader: CSV DictReader instance

        Returns:
            Formatted string with structured row data
        """
        content_parts = []
        fieldnames = reader.fieldnames or []

        # Add header information
        content_parts.append(f"CSV Data with columns: {', '.join(fieldnames)}\n")

        row_count = 0
        for row_num, row in enumerate(reader, 1):
            try:
                # Create structured representation of the row
                row_content = self._format_row(row, fieldnames, row_num)
                content_parts.append(row_content)
                row_count += 1

            except Exception as e:
                logger.warning(f"Failed to process row {row_num}: {e}")
                continue

        content_parts.append(f"\nTotal rows processed: {row_count}")
        return "\n".join(content_parts)

    def _format_row(self, row: Dict[str, Any], fieldnames: List[str], row_num: int) -> str:
        """
        Format a single CSV row preserving column-value relationships.

        Args:
            row: Dictionary representing the CSV row
            fieldnames: List of column names
            row_num: Row number for reference

        Returns:
            Formatted string representation of the row
        """
        row_parts = [f"Row {row_num}:"]

        for field in fieldnames:
            value = row.get(field, "")
            # Clean and format the value
            if value is not None:
                value_str = str(value).strip()
                if value_str:
                    row_parts.append(f"  {field}: {value_str}")
                else:
                    row_parts.append(f"  {field}: [empty]")
            else:
                row_parts.append(f"  {field}: [null]")

        return "\n".join(row_parts) + "\n"
