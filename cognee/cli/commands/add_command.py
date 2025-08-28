import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException


class AddCommand(SupportsCliCommand):
    command_string = "add"
    help_string = "Add data to Cognee for knowledge graph processing"
    docs_url = DEFAULT_DOCS_URL
    description = """
Add data to Cognee for knowledge graph processing.

This is the first step in the Cognee workflow - it ingests raw data and prepares it
for processing. The function accepts various data formats including text, files, and
binary streams, then stores them in a specified dataset for further processing.

Supported Input Types:
- **Text strings**: Direct text content
- **File paths**: Local file paths (absolute paths starting with "/")  
- **File URLs**: "file:///absolute/path" or "file://relative/path"
- **S3 paths**: "s3://bucket-name/path/to/file"
- **Lists**: Multiple files or text strings in a single call

Supported File Formats:
- Text files (.txt, .md, .csv)
- PDFs (.pdf)
- Images (.png, .jpg, .jpeg) - extracted via OCR/vision models
- Audio files (.mp3, .wav) - transcribed to text
- Code files (.py, .js, .ts, etc.) - parsed for structure and content
- Office documents (.docx, .pptx)

After adding data, use `cognee cognify` to process it into knowledge graphs.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "data",
            nargs="+",
            help="Data to add: text content, file paths (/path/to/file), file URLs (file://path), S3 paths (s3://bucket/file), or mix of these",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset name to organize your data (default: main_dataset)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Import cognee here to avoid circular imports
            import cognee

            fmt.echo(f"Adding {len(args.data)} item(s) to dataset '{args.dataset_name}'...")

            # Run the async add function
            async def run_add():
                try:
                    # Pass all data items as a list to cognee.add if multiple items
                    if len(args.data) == 1:
                        data_to_add = args.data[0]
                    else:
                        data_to_add = args.data

                    fmt.echo("Processing data...")
                    await cognee.add(data=data_to_add, dataset_name=args.dataset_name)
                    fmt.success(f"Successfully added data to dataset '{args.dataset_name}'")
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to add data: {str(e)}")

            asyncio.run(run_add())

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error adding data: {str(e)}", error_code=1)
