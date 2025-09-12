import csv
import re
from typing import AsyncGenerator, List, Dict, Any
from uuid import NAMESPACE_OID, uuid5

from cognee.shared.logging_utils import get_logger
from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

logger = get_logger(__name__)


class CSVChunker(Chunker):
    """
    Custom chunker for CSV data that preserves row-column relationships.

    This chunker operates at row granularity, ensuring each chunk contains
    complete row information with column context preserved.
    """

    def __init__(
        self,
        document,
        get_text: callable,
        max_chunk_tokens: int,
        chunk_size: int = 1024,
        rows_per_chunk: int = 1,
    ):
        """
        Initialize CSV chunker.

        Args:
            document: Document being chunked
            get_text: Callable to get text content
            max_chunk_tokens: Maximum tokens per chunk
            chunk_size: Target chunk size in characters
            rows_per_chunk: Number of CSV rows per chunk (default: 1)
        """
        super().__init__(document, get_text, max_chunk_tokens)
        self.chunk_size = chunk_size
        if not isinstance(rows_per_chunk, int):
            raise TypeError("rows_per_chunk must be an integer")
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be >= 1")
        self.rows_per_chunk = rows_per_chunk
        self.header_info = ""

    async def read(self) -> AsyncGenerator[DocumentChunk, None]:
        """
        Read and chunk CSV content preserving row-column relationships.

        Yields:
            DocumentChunk instances containing structured CSV row data
        """
        async for content_text in self.get_text():
            try:
                # Parse the structured CSV content
                csv_data = self._parse_csv_content(content_text)

                if not csv_data:
                    logger.warning("No valid CSV data found in content")
                    continue

                # Extract header information
                self.header_info = csv_data.get("header", "")
                rows = csv_data.get("rows", [])

                if not rows:
                    logger.warning("No data rows found in CSV content")
                    continue

                # Chunk rows while preserving column context
                async for chunk in self._chunk_csv_rows(rows):
                    yield chunk

            except Exception:
                logger.exception("Error processing CSV content")
                raise

    def _parse_csv_content(self, content: str) -> Dict[str, Any]:
        """
        Parse the structured CSV content created by CsvLoader.

        Args:
            content: Structured CSV content from loader

        Returns:
            Dictionary with header and rows information
        """
        lines = content.strip().splitlines()

        if not lines:
            return {}

        # Extract header information (first line should contain column info)
        header_line = lines[0]

        # Parse individual rows
        rows = []
        current_row = {}
        row_num = None

        for line in lines[1:]:  # Skip header line
            original_line = line  # Keep original for pattern matching
            line = line.strip()  # Stripped for empty line checking

            # Skip empty lines and summary lines
            if not line or line.startswith("Total rows processed:"):
                # End of current row, save it
                if current_row and row_num is not None:
                    rows.append(
                        {
                            "row_number": row_num,
                            "data": current_row.copy(),
                            "raw_content": self._format_row_for_chunk(current_row, row_num),
                        }
                    )
                    current_row = {}
                    row_num = None
                continue

            # Check if this is a row header (use stripped line)
            row_match = re.match(r"^Row (\d+):$", line)
            if row_match:
                # Save previous row if exists
                if current_row and row_num is not None:
                    rows.append(
                        {
                            "row_number": row_num,
                            "data": current_row.copy(),
                            "raw_content": self._format_row_for_chunk(current_row, row_num),
                        }
                    )
                    current_row = {}

                row_num = int(row_match.group(1))
                continue

            # Check if this is a field line (use original line with indentation)
            field_match = re.match(r"^  ([^:]+): (.*)$", original_line)
            if field_match and row_num is not None:
                field_name = field_match.group(1).strip()
                field_value = field_match.group(2).strip()

                # Clean up special values
                if field_value in ["[empty]", "[null]"]:
                    field_value = ""

                current_row[field_name] = field_value
                continue

        # Don't forget the last row (if not already processed)
        if current_row and row_num is not None:
            rows.append(
                {
                    "row_number": row_num,
                    "data": current_row.copy(),
                    "raw_content": self._format_row_for_chunk(current_row, row_num),
                }
            )

        return {"header": header_line, "rows": rows}

    def _format_row_for_chunk(self, row_data: Dict[str, str], row_num: int) -> str:
        """
        Format a row's data for inclusion in a chunk.

        Args:
            row_data: Dictionary of column-value pairs
            row_num: Row number

        Returns:
            Formatted string representation
        """
        parts = [f"Row {row_num}:"]

        for field, value in row_data.items():
            if value:
                parts.append(f"  {field}: {value}")
            else:
                parts.append(f"  {field}: [empty]")

        return "\n".join(parts)

    async def _chunk_csv_rows(
        self, rows: List[Dict[str, Any]]
    ) -> AsyncGenerator[DocumentChunk, None]:
        """
        Chunk CSV rows while preserving structure.

        Args:
            rows: List of parsed row data

        Yields:
            DocumentChunk instances
        """
        # Initialize tokenizer once
        tokenizer = None
        try:
            from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
            embedding_engine = get_vector_engine().embedding_engine
            tokenizer = getattr(embedding_engine, "tokenizer", None)
        except (ImportError, ModuleNotFoundError, AttributeError):
            tokenizer = None

        # Group rows into chunks
        for i in range(0, len(rows), self.rows_per_chunk):
            chunk_rows = rows[i : i + self.rows_per_chunk]

            # Create chunk text with header context
            chunk_parts = []

            # Add header context for each chunk
            if self.header_info:
                chunk_parts.append(self.header_info)
                chunk_parts.append("")  # Empty line for separation

            # Add row data
            for row in chunk_rows:
                chunk_parts.append(row["raw_content"])
                chunk_parts.append("")  # Empty line between rows

            chunk_text = "\n".join(chunk_parts).strip()

            # Extract columns in original CSV order from header_info
            columns: List[str] = []
            if self.header_info:
                m = re.search(r"CSV Data with columns:\s*(.+)$", self.header_info)
                if m:
                    try:
                        columns_row = next(csv.reader([m.group(1)], skipinitialspace=True))
                        columns = [c for c in (col.strip() for col in columns_row) if c]
                    except (StopIteration, csv.Error, ValueError):
                        pass
            if not columns:
                # Fallback: union of keys (order not guaranteed)
                columns = sorted({col for row in chunk_rows for col in row["data"].keys()})

            # Create metadata for the chunk
            chunk_metadata = {
                "index_fields": ["text"],
                "csv_metadata": {
                    "row_numbers": [row["row_number"] for row in chunk_rows],
                    "row_count": len(chunk_rows),
                    "columns": columns,
                    "chunk_type": "csv_rows",
                },
            }

            # Calculate token count if possible
            if tokenizer is not None:
                token_count = tokenizer.count_tokens(chunk_text)
            else:
                # Fallback to word count if tokenizer not available
                token_count = len(chunk_text.split())

            # Ensure we don't exceed max tokens
            if token_count > self.max_chunk_size:
                logger.warning(
                    f"CSV chunk with {token_count} tokens exceeds max size of {self.max_chunk_size}. "
                    f"Consider reducing rows_per_chunk."
                )

            # Create and yield the chunk
            chunk = DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{self.document.id!s}-csv-{self.chunk_index}"),
                text=chunk_text,
                chunk_size=token_count,
                is_part_of=self.document,
                chunk_index=self.chunk_index,
                cut_type="csv_row_boundary",
                contains=[],
                metadata=chunk_metadata,
            )

            yield chunk
            self.chunk_index += 1
