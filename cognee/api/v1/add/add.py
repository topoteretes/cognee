import inspect
from tempfile import SpooledTemporaryFile
from types import SimpleNamespace
from uuid import UUID
from typing import Union, BinaryIO, List, Optional, Any

from cognee.modules.users.models import User
from cognee.modules.pipelines import Task, run_pipeline
from cognee.modules.pipelines.layers.resolve_authorized_user_dataset import (
    resolve_authorized_user_dataset,
)
from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
    reset_dataset_pipeline_run_status,
)
from cognee.modules.pipelines.layers.pipeline_execution_mode import get_pipeline_executor
from cognee.modules.engine.operations.setup import setup
from cognee.tasks.ingestion import ingest_data, resolve_data_directories
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.resolve_dlt_sources import resolve_dlt_sources
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def _normalize_filename(filename: Optional[str], index: int) -> str:
    if not filename:
        return f"upload_{index}.bin"
    normalized = str(filename).replace("\\", "/").split("/")[-1]
    return normalized or f"upload_{index}.bin"


async def _read_stream_bytes(stream: Any) -> bytes:
    if not hasattr(stream, "read"):
        raise TypeError(f"Expected stream-like object, got: {type(stream)}")

    # Best effort to read from the start of the stream.
    if hasattr(stream, "seek"):
        try:
            stream.seek(0)
        except Exception:
            pass

    data = stream.read()
    if inspect.isawaitable(data):
        data = await data

    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Unsupported stream payload type: {type(data)}")

    return bytes(data)


async def _materialize_stream_for_background(data_item: Any, index: int = 0) -> Any:
    if isinstance(data_item, DataItem):
        return DataItem(
            data=await _materialize_stream_for_background(data_item.data, index=index),
            label=data_item.label,
            external_metadata=data_item.external_metadata,
            data_id=data_item.data_id,
        )

    if isinstance(data_item, list):
        return [
            await _materialize_stream_for_background(item, index=i)
            for i, item in enumerate(data_item)
        ]

    # Keep stable primitives untouched.
    if isinstance(data_item, str):
        return data_item

    stream = getattr(data_item, "file", data_item if hasattr(data_item, "read") else None)
    if stream is None:
        return data_item

    payload = await _read_stream_bytes(stream)
    buffer = SpooledTemporaryFile(mode="w+b")
    buffer.write(payload)
    buffer.seek(0)

    filename = _normalize_filename(
        getattr(data_item, "filename", None) or getattr(stream, "name", None),
        index=index,
    )

    # Ingestion path supports objects exposing `.file` and `.filename`.
    return SimpleNamespace(file=buffer, filename=filename)


async def add(
    data: Union[
        BinaryIO,
        list[BinaryIO],
        str,
        list[str],
        DataItem,
        list[DataItem],
        Any,  # DltResource, SourceFactory, or other dlt types
    ],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    dataset_id: Optional[UUID] = None,
    preferred_loaders: Optional[List[Union[str, dict[str, dict[str, Any]]]]] = None,
    incremental_loading: bool = True,
    data_per_batch: Optional[int] = 20,
    importance_weight: Optional[float] = 0.5,
    run_in_background: bool = False,
    **kwargs,
):
    """
    Add data to Cognee for knowledge graph processing.

    This is the first step in the Cognee workflow - it ingests raw data and prepares it
    for processing. The function accepts various data formats including text, files, urls and
    binary streams, then stores them in a specified dataset for further processing.

    Prerequisites:
        - **LLM_API_KEY**: Must be set in environment variables for content processing
        - **Database Setup**: Relational and vector databases must be configured
        - **User Authentication**: Uses default user if none provided (created automatically)

    Supported Input Types:
        - **Text strings**: Direct text content (str) - any string not starting with "/" or "file://"
        - **File paths**: Local file paths as strings in these formats:
            * Absolute paths: "/path/to/document.pdf"
            * File URLs: "file:///path/to/document.pdf" or "file://relative/path.txt"
            * S3 paths: "s3://bucket-name/path/to/file.pdf"
        - **Binary file objects**: File handles/streams (BinaryIO)
        - **Lists**: Multiple files or text strings in a single call

    Supported File Formats:
        - Text files (.txt, .md, .csv)
        - PDFs (.pdf)
        - Images (.png, .jpg, .jpeg) - extracted via OCR/vision models
        - Audio files (.mp3, .wav) - transcribed to text
        - Code files (.py, .js, .ts, etc.) - parsed for structure and content
        - Office documents (.docx, .pptx)

            Workflow:
        1. **Data Resolution**: Resolves file paths and validates accessibility
        2. **Content Extraction**: Extracts text content from various file formats
        3. **Dataset Storage**: Stores processed content in the specified dataset
        4. **Metadata Tracking**: Records file metadata, timestamps, and user permissions
        5. **Permission Assignment**: Grants user read/write/delete/share permissions on dataset

    Args:
        data: The data to ingest. Can be:
            - Single text string: "Your text content here"
            - Absolute file path: "/path/to/document.pdf"
            - File URL: "file:///absolute/path/to/document.pdf" or "file://relative/path.txt"
            - S3 path: "s3://my-bucket/documents/file.pdf"
            - List of mixed types: ["text content", "/path/file.pdf", "file://doc.txt", file_handle]
            - Binary file object: open("file.txt", "rb")
            - url: A web link url (https or http)
        dataset_name: Name of the dataset to store data in. Defaults to "main_dataset".
                    Create separate datasets to organize different knowledge domains.
        user: User object for authentication and permissions. Uses default user if None.
              Default user: "default_user@example.com" (created automatically on first use).
              Users can only access datasets they have permissions for.
        node_set: Optional list of node identifiers for graph organization and access control.
                 Used for grouping related data points in the knowledge graph.
        vector_db_config: Optional configuration for vector database (for custom setups).
        graph_db_config: Optional configuration for graph database (for custom setups).
        dataset_id: Optional specific dataset UUID to use instead of dataset_name.
        run_in_background: If True, starts ingestion asynchronously and returns immediately.
                          If False (default), waits for completion before returning.
        extraction_rules: Optional dictionary of rules (e.g., CSS selectors, XPath) for extracting specific content from web pages using BeautifulSoup
        tavily_config: Optional configuration for Tavily API, including API key and extraction settings
        soup_crawler_config: Optional configuration for BeautifulSoup crawler, specifying concurrency, crawl delay, and extraction rules.

    Returns:
        PipelineRunInfo: Information about the ingestion pipeline execution including:
            - Pipeline run ID for tracking
            - Dataset ID where data was stored
            - Processing status and any errors
            - Execution timestamps and metadata

    Next Steps:
        After successfully adding data, call `cognify()` to process the ingested content:

        ```python
        import cognee

        # Step 1: Add your data (text content or file path)
        await cognee.add("Your document content")  # Raw text
        # OR
        await cognee.add("/path/to/your/file.pdf")  # File path

        # Step 2: Process into knowledge graph
        await cognee.cognify()

        # Step 3: Search and query
        results = await cognee.search("What insights can you find?")
        ```

    Example Usage:
        ```python
        # Add a single text document
        await cognee.add("Natural language processing is a field of AI...")

        # Add multiple files with different path formats
        await cognee.add([
            "/absolute/path/to/research_paper.pdf",        # Absolute path
            "file://relative/path/to/dataset.csv",         # Relative file URL
            "file:///absolute/path/to/report.docx",        # Absolute file URL
            "s3://my-bucket/documents/data.json",           # S3 path
            "Additional context text"                       # Raw text content
        ])

        # Add to a specific dataset
        await cognee.add(
            data="Project documentation content",
            dataset_name="project_docs"
        )

        # Add a single file
        await cognee.add("/home/user/documents/analysis.pdf")

        # Add a single url and bs4 extract ingestion method
        extraction_rules = {
            "title": "h1",
            "description": "p",
            "more_info": "a[href*='more-info']"
        }
        await cognee.add("https://example.com",extraction_rules=extraction_rules)

        # Add a single url and tavily extract ingestion method
        Make sure to set TAVILY_API_KEY = YOUR_TAVILY_API_KEY as a environment variable
        await cognee.add("https://example.com")

        # Add multiple urls
        await cognee.add(["https://example.com","https://books.toscrape.com"])
        ```

    Environment Variables:
        Required:
        - LLM_API_KEY: API key for your LLM provider (OpenAI, Anthropic, etc.)

        Optional:
        - LLM_PROVIDER: "openai" (default), "anthropic", "gemini", "ollama", "mistral", "bedrock"
        - LLM_MODEL: Model name (default: "gpt-5-mini")
        - DEFAULT_USER_EMAIL: Custom default user email
        - DEFAULT_USER_PASSWORD: Custom default user password
        - VECTOR_DB_PROVIDER: "lancedb" (default), "chromadb", "pgvector"
        - GRAPH_DATABASE_PROVIDER: "ladybug" (default), "neo4j"
        - TAVILY_API_KEY: YOUR_TAVILY_API_KEY

    """
    # Route to remote instance if connected via serve()
    from cognee.api.v1.serve.state import get_remote_client

    client = get_remote_client()
    if client is not None:
        result = await client.add(data, dataset_name)
        # Wrap in a simple namespace so callers expecting .model_dump() still work
        from types import SimpleNamespace

        return SimpleNamespace(**result)

    if preferred_loaders is not None:
        transformed = {}
        for item in preferred_loaders:
            if isinstance(item, dict):
                transformed.update(item)
            else:
                transformed[item] = {}
        preferred_loaders = transformed

    tasks = [
        Task(resolve_data_directories, include_subdirectories=True),
        Task(
            ingest_data,
            dataset_name,
            user,
            node_set,
            dataset_id,
            preferred_loaders,
            importance_weight,
        ),
    ]

    await setup()

    user, authorized_dataset = await resolve_authorized_user_dataset(
        dataset_name=dataset_name, dataset_id=dataset_id, user=user
    )

    # Expand DLT resources (and auto-detected CSV/connection strings) into
    # standard DataItems before the pipeline sees them.
    data = await resolve_dlt_sources(
        data,
        dataset_name=dataset_name,
        user=user,
        **kwargs,
    )

    # Background runs must not depend on caller/request-scoped stream lifetimes.
    # Materialize stream-like inputs into owned in-memory buffers up front.
    if run_in_background:
        data = await _materialize_stream_for_background(data)

    await reset_dataset_pipeline_run_status(
        authorized_dataset.id, user, pipeline_names=["add_pipeline", "cognify_pipeline"]
    )

    pipeline_executor_func = get_pipeline_executor(run_in_background=run_in_background)

    result = await pipeline_executor_func(
        pipeline=run_pipeline,
        tasks=tasks,
        datasets=[authorized_dataset.id],
        data=data,
        user=user,
        pipeline_name="add_pipeline",
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
        use_pipeline_cache=True,
        incremental_loading=incremental_loading,
        data_per_batch=data_per_batch,
    )

    # run_pipeline_blocking returns {dataset_id: PipelineRunInfo} but callers
    # expect a single PipelineRunInfo (add always processes one dataset).
    if isinstance(result, dict) and len(result) == 1:
        return next(iter(result.values()))

    return result
