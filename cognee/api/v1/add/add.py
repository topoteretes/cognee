from uuid import UUID
from typing import Union, BinaryIO, List, Optional

from cognee.modules.pipelines import Task
from cognee.modules.users.models import User
from cognee.modules.pipelines import cognee_pipeline
from cognee.tasks.ingestion import ingest_data, resolve_data_directories


async def add(
    data: Union[BinaryIO, list[BinaryIO], str, list[str]],
    dataset_name: str = "main_dataset",
    user: User = None,
    node_set: Optional[List[str]] = None,
    vector_db_config: dict = None,
    graph_db_config: dict = None,
    dataset_id: Optional[UUID] = None,
):
    """
    Add data to Cognee for knowledge graph processing.

    This is the first step in the Cognee workflow - it ingests raw data and prepares it
    for processing. The function accepts various data formats including text, files, and
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
        ```

    Environment Variables:
        Required:
        - LLM_API_KEY: API key for your LLM provider (OpenAI, Anthropic, etc.)

        Optional:
        - LLM_PROVIDER: "openai" (default), "anthropic", "gemini", "ollama"
        - LLM_MODEL: Model name (default: "gpt-4o-mini")
        - DEFAULT_USER_EMAIL: Custom default user email
        - DEFAULT_USER_PASSWORD: Custom default user password
        - VECTOR_DB_PROVIDER: "lancedb" (default), "chromadb", "qdrant", "weaviate"
        - GRAPH_DATABASE_PROVIDER: "kuzu" (default), "neo4j", "networkx"

    Raises:
        FileNotFoundError: If specified file paths don't exist
        PermissionError: If user lacks access to files or dataset
        UnsupportedFileTypeError: If file format cannot be processed
        InvalidValueError: If LLM_API_KEY is not set or invalid
    """
    tasks = [
        Task(resolve_data_directories, include_subdirectories=True),
        Task(ingest_data, dataset_name, user, node_set, dataset_id),
    ]

    pipeline_run_info = None

    async for run_info in cognee_pipeline(
        tasks=tasks,
        datasets=dataset_id if dataset_id else dataset_name,
        data=data,
        user=user,
        pipeline_name="add_pipeline",
        vector_db_config=vector_db_config,
        graph_db_config=graph_db_config,
    ):
        pipeline_run_info = run_info

    return pipeline_run_info
