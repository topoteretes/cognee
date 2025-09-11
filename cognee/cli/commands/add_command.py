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
- **Folder URIs**: "folder:///abs/path" groups files by subfolders into datasets (prefix with parent)
- **Lists**: Multiple items in a single call

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
            help="Data to add: text, files (/path), file://, s3://, folder:///abs/path (datasets by subfolders)",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset name (ignored for folder://; subfolders become datasets)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            # Import cognee here to avoid circular imports
            import cognee

            def contains_folder_uri(items):
                return any(isinstance(i, str) and i.startswith("folder://") for i in items)

            inputs = args.data
            dataset_label = args.dataset_name

            if contains_folder_uri(inputs):
                fmt.echo("Detected folder:// input. Subfolders will be added as separate datasets.")
            else:
                fmt.echo(f"Adding {len(inputs)} item(s) to dataset '{dataset_label}'...")

            # Run the async add function
            async def run_add():
                try:
                    data_to_add = inputs if len(inputs) > 1 else inputs[0]
                    fmt.echo("Processing data...")
                    await cognee.add(data=data_to_add, dataset_name=dataset_label)
                    if contains_folder_uri(inputs):
                        fmt.success("Successfully added folder datasets")
                    else:
                        fmt.success(f"Successfully added data to dataset '{dataset_label}'")
                except Exception as e:
                    raise CliCommandInnerException(f"Failed to add data: {str(e)}")

            asyncio.run(run_add())

        except Exception as e:
            if isinstance(e, CliCommandInnerException):
                raise CliCommandException(str(e), error_code=1)
            raise CliCommandException(f"Error adding data: {str(e)}", error_code=1)
