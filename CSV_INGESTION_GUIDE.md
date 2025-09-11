# CSV Ingestion Pipeline

This document describes the dedicated CSV ingestion pipeline for Cognee that preserves row-column relationships in the produced chunks.

## Overview

The CSV ingestion pipeline consists of two main components:

1. **CsvLoader**: A custom loader that reads CSV files and converts them into structured text format
2. **CSVChunker**: A custom chunker that operates at row granularity while preserving column context

## Features

- **Row-Column Preservation**: Each chunk maintains explicit associations between column names and row values
- **Configurable Chunking**: Chunk size can be configured at both the row level and token level
- **Structured Output**: Produces human-readable and machine-parseable structured text
- **Error Handling**: Robust parsing with error handling for malformed CSV data
- **Metadata Rich**: Each chunk includes detailed metadata about row numbers, column information, and chunk type

## Usage

### Basic Usage

```python
import cognee
from cognee.modules.chunking import CSVChunker

# Ingest CSV file
await cognee.add("path/to/your/file.csv", preferred_loaders=["csv_loader"])

# Process with CSV-specific chunking
chunks = await cognee.cognify(chunker=CSVChunker)
```

### Advanced Configuration

```python
from cognee.infrastructure.loaders.core.csv_loader import CsvLoader
from cognee.modules.chunking.CSVChunker import CSVChunker

# Configure CSV loader
loader = CsvLoader()

# Load with custom settings
processed_path = await loader.load(
    file_path="data.csv",
    encoding="utf-8",           # Text encoding
    delimiter=",",              # CSV delimiter
    quotechar='"'               # Quote character
)

# Configure CSV chunker
chunker = CSVChunker(
    document=document,
    get_text=get_text_function,
    max_chunk_tokens=1000,      # Maximum tokens per chunk
    rows_per_chunk=5            # Number of CSV rows per chunk
)

# Process chunks
async for chunk in chunker.read():
    print(f"Chunk {chunk.chunk_index} contains rows: {chunk.metadata['csv_metadata']['row_numbers']}")
```

## Output Format

### CSV Loader Output

The CSV loader converts raw CSV data into a structured text format:

```text
```text
CSV Data with columns: Name, Age, Department, Salary, Location, Start_Date

Row 1:
  Name: John Doe
  Age: 30
  Department: Engineering
  Salary: 75000
  Location: New York
  Start_Date: 2020-01-15

Row 2:
  Name: Jane Smith
  Age: 28
  Department: Marketing
  Salary: 65000
  Location: San Francisco
  Start_Date: 2019-03-22

Total rows processed: 2
```
```

### Chunker Output

Each chunk produced by CSVChunker includes:

- **Header Context**: Column information is included in every chunk
- **Row Data**: Complete row information with field-value mappings
- **Rich Metadata**: Detailed information about the chunk contents

Example chunk content:
```text
CSV Data with columns: Name, Age, Department, Salary, Location, Start_Date

Row 1:
  Name: John Doe
  Age: 30
  Department: Engineering
  Salary: 75000
  Location: New York
  Start_Date: 2020-01-15

Row 2:
  Name: Jane Smith
  Age: 28
  Department: Marketing
  Salary: 65000
  Location: San Francisco
  Start_Date: 2019-03-22
```

Example chunk metadata:
```python
{
    "index_fields": ["text"],
    "csv_metadata": {
        "row_numbers": [1, 2],
        "row_count": 2,
        "columns": ["Name", "Age", "Department", "Salary", "Location", "Start_Date"],
        "chunk_type": "csv_rows"
    }
}
```

## Technical Details

### Supported File Types

- **Extensions**: `.csv`
- **MIME Types**: `text/csv`, `application/csv`

### CSV Features

- Automatic dialect detection (delimiter, quote character)
- Support for quoted fields with commas
- Handles empty fields and null values
- Unicode support with configurable encoding

### Performance Considerations

- **Memory Efficient**: Streams data rather than loading entire files into memory
- **Configurable Chunking**: Balance between context preservation and chunk size
- **Token Counting**: Uses vector engine tokenizer when available, falls back to word count

## Best Practices

1. **Choose Appropriate Chunk Size**: 
   - For small tables: Use `rows_per_chunk=1` for maximum granularity
   - For large tables: Use `rows_per_chunk=5-10` for better performance

2. **Consider Data Types**:
   - Text-heavy fields may require larger token limits
   - Numeric data is typically more compact

3. **Monitor Chunk Quality**:
   - Check that chunks don't exceed `max_chunk_tokens`
   - Verify that column context is preserved in all chunks

## Integration with Cognee

The CSV pipeline integrates seamlessly with Cognee's existing infrastructure:

- **Loader Registration**: CsvLoader is automatically registered in the loader engine
- **Priority Handling**: Use `preferred_loaders=["csv_loader"]` to ensure CSV files use the dedicated loader
- **Chunker Selection**: Specify `CSVChunker` when processing documents for optimal CSV handling

## Error Handling

The pipeline includes comprehensive error handling:

- **File Access Errors**: Proper error messages for missing or inaccessible files
- **Encoding Issues**: Graceful handling of encoding problems with fallback options
- **Malformed CSV**: Robust parsing that continues processing when possible
- **Token Limits**: Warnings when chunks exceed specified token limits

## Testing

Run the included test suite to verify the pipeline:

```bash
python test_csv_pipeline.py
```

This test validates:
- CSV loader functionality
- Chunker parsing and output
- Row-column relationship preservation
- Metadata accuracy

## Troubleshooting

### Common Issues

1. **"No suitable loader found"**: Ensure CsvLoader is registered and use `preferred_loaders=["csv_loader"]`
2. **Empty chunks**: Check CSV file format and ensure it has proper headers
3. **Large chunks**: Reduce `rows_per_chunk` or increase `max_chunk_tokens`
4. **Encoding errors**: Specify the correct encoding parameter in the loader

### Debug Mode

For debugging, you can examine the intermediate output:

```python
# Check loader output
loader = CsvLoader()
processed_path = await loader.load("data.csv")

# Read the structured content
with open(processed_path, 'r') as f:
    structured_content = f.read()
    print(structured_content)
```
