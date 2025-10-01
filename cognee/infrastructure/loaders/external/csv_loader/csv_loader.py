import csv
import io
from typing import List, Dict, Any, Optional
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage
from cognee.infrastructure.files.storage.get_storage_config import get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata, FileMetadata
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class CsvLoadError(Exception):
    """Exception raised when CSV loading fails."""
    pass


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
        return [
            "text/csv", 
            "application/csv", 
            "application/vnd.ms-excel", 
            "text/plain", 
            "text/x-csv"
        ]

    @property
    def loader_name(self) -> str:
        """Unique identifier for this loader."""
        return "csv_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """
        Check if this loader can handle the given file.
        Uses extension-first matching strategy with constrained MIME type fallback.

        Args:
            extension: File extension (may be None)
            mime_type: MIME type of the file (may be None)

        Returns:
            True if file can be handled, False otherwise
        """
        # Guard against None values and normalize inputs
        extension = (extension or "").strip().lower()
        mime_type = (mime_type or "").strip().lower()
        
        # Normalize extension (remove dot prefix for consistency)
        if extension.startswith('.'):
            extension = extension[1:]
            
        # Use sets for efficient membership testing
        supported_extensions_set = {ext.lower() for ext in self.supported_extensions}
        supported_mime_types_set = {mt.lower() for mt in self.supported_mime_types}
        
        # Extension-first matching strategy
        if extension:
            if extension in supported_extensions_set:
                return True
        
        # Constrained MIME type fallback - avoid risky MIME types without matching extension
        if mime_type:
            # Risky MIME types that could incorrectly route non-CSV files
            risky_mime_types = {"text/plain", "application/vnd.ms-excel"}
            
            if mime_type in risky_mime_types:
                # Only accept risky MIME types if extension also matches
                return extension in supported_extensions_set
            elif mime_type in supported_mime_types_set:
                # Safe MIME types can be accepted without extension match
                return True
        
        # Neither extension nor MIME type matched
        return False

    async def load(
        self,
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        quotechar: str = '"',
        file_stream: Optional[io.IOBase] = None,
        **kwargs
    ) -> str:
        """
        Load and process the CSV file, preserving row-column relationships.

        Args:
            file_path: Path to the CSV file to load
            encoding: Text encoding to use (default: utf-8)
            delimiter: CSV field delimiter (default: comma)
            quotechar: CSV quote character (default: double quote)
            file_stream: Optional file stream to use instead of opening file_path
            **kwargs: Additional keyword arguments (accepted for compatibility)

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
            # Get file metadata - ensure we always pass a binary stream to get_file_metadata
            binary_stream = None
            if file_stream is not None:
                # Determine if the provided stream is text or binary
                if isinstance(file_stream, io.TextIOBase):
                    # Text stream - need to get underlying binary stream for metadata
                    if hasattr(file_stream, 'buffer'):
                        # TextIOWrapper has a buffer attribute pointing to binary stream
                        file_metadata = await get_file_metadata(file_stream.buffer)
                    else:
                        # Cannot extract metadata from text-only stream without binary access
                        raise CsvLoadError("Cannot extract metadata from text stream without binary buffer access")
                    should_close_binary_stream = False
                else:
                    # Binary stream (including BytesIO) - can use directly for metadata
                    file_metadata = await get_file_metadata(file_stream)
                    should_close_binary_stream = False
            else:
                binary_stream = open(file_path, "rb")
                file_metadata = await get_file_metadata(binary_stream)
                should_close_binary_stream = True

            # Name ingested file based on original file content hash
            storage_file_name = "csv_" + file_metadata["content_hash"] + ".txt"

            # Read and process CSV content
            if file_stream is not None:
                # Determine how to handle the provided stream
                if isinstance(file_stream, io.TextIOBase):
                    # Already a text stream - use directly
                    csvfile = file_stream
                    wrapper_created = False
                else:
                    # Binary stream - wrap in TextIOWrapper
                    csvfile = io.TextIOWrapper(file_stream, encoding=encoding, newline="")
                    wrapper_created = True
            else:
                csvfile = io.TextIOWrapper(binary_stream, encoding=encoding, newline="")
                wrapper_created = True

            try:
                # Detect dialect if not specified
                sample = csvfile.read(1024)
                csvfile.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                    # Only use detected delimiter if not explicitly set and valid
                    if delimiter == "," and isinstance(dialect.delimiter, str) and len(dialect.delimiter) == 1:
                        delimiter = dialect.delimiter
                    # Only use detected quotechar if not explicitly set and valid
                    if quotechar == '"' and isinstance(dialect.quotechar, str) and len(dialect.quotechar) == 1:
                        quotechar = dialect.quotechar
                except csv.Error:
                    # Use defaults if detection fails
                    pass

                reader = csv.DictReader(csvfile, delimiter=delimiter, quotechar=quotechar)

                structured_content = self._process_csv_rows(reader)
                column_count = len(reader.fieldnames or [])

            finally:
                # Clean up wrapper if we created it, but preserve original stream
                if wrapper_created and file_stream is not None:
                    csvfile.detach()  # Detach to avoid closing the underlying stream
                elif wrapper_created and should_close_binary_stream:
                    csvfile.close()

            # If we opened a binary stream ourselves, close it
            if should_close_binary_stream and binary_stream is not None:
                binary_stream.close()

            # Store the processed content
            storage_config = get_storage_config()
            
            # Handle both object and dict return types for data_root_directory
            data_root_directory = getattr(storage_config, "data_root_directory", None)
            if data_root_directory is None:
                # Fallback to dict access
                try:
                    data_root_directory = storage_config.get("data_root_directory")
                except AttributeError:
                    # Neither object nor dict-like
                    raise CsvLoadError(
                        "storage_config does not contain 'data_root_directory' attribute or key"
                    )
            
            if data_root_directory is None:
                raise CsvLoadError(
                    "data_root_directory is not configured in storage settings"
                )
                
            storage = get_file_storage(data_root_directory)

            full_file_path = await storage.store(storage_file_name, structured_content)

            logger.info(
                f"Successfully processed CSV file with {column_count} columns"
            )
            return full_file_path

        except Exception as e:
            logger.exception(f"Failed to process CSV {file_path}")
            raise CsvLoadError(f"CSV processing failed: {e}") from e

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

            except (ValueError, KeyError, AttributeError) as e:
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
            
            # Handle None values properly
            if value is None:
                row_parts.append(f"  {field}: [null]")
            elif isinstance(value, str):
                # Preserve exact string content, only escape control characters
                if value == "":
                    row_parts.append(f"  {field}: [empty]")
                else:
                    # Escape only control characters to maintain structure
                    escaped_value = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    row_parts.append(f"  {field}: {escaped_value}")
            else:
                # For non-string values, convert safely without stripping
                value_str = str(value)
                if value_str == "":
                    row_parts.append(f"  {field}: [empty]")
                else:
                    # Escape control characters in converted string
                    escaped_value = value_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    row_parts.append(f"  {field}: {escaped_value}")

        return "\n".join(row_parts) + "\n"
