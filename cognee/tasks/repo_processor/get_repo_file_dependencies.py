import asyncio
import math
import os
from pathlib import Path
from typing import Set
from typing import AsyncGenerator, Optional, List
from uuid import NAMESPACE_OID, uuid5
from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile, Repository

# constant, declared only once
EXCLUDED_DIRS: Set[str] = {
    ".venv",
    "venv",
    "env",
    ".env",
    "site-packages",
    "node_modules",
    "dist",
    "build",
    ".git",
    "tests",
    "test",
}


async def get_source_code_files(
    repo_path,
    language_config: dict[str, list[str]] | None = None,
    excluded_paths: Optional[List[str]] = None,
):
    """
    Retrieve Python source code files from the specified repository path.

    This function scans the given repository path for files that have the .py extension
    and yields those files as data points, along with their content, for further processing.

    Args:
        repo_path: The directory to search for source code files.
        language_config: Dictionary mapping language names to file extensions.
        excluded_paths: List of paths to exclude from scanning.

    Yields:
        File information and content as DataPoints.

    Example:
        repo_path = Path("/path/to/repo")
        async for files_data in get_source_code_files(repo_path):
            print(files_data)
    """
    # default language_config if none is provided
    if language_config is None:
        language_config = {"Python": [".py"]}

    repo_path = Path(repo_path).resolve()

    # get the allowed extensions from the language_config
    allowed_extensions = set()
    for extensions in language_config.values():
        allowed_extensions.update(extensions)

    # build a set of user-specified excluded paths (absolute)
    user_excluded = set()
    if excluded_paths:
        for ep in excluded_paths:
            ep_abs = (repo_path / ep).resolve()
            user_excluded.add(ep_abs)

    # Default language configuration with all supported languages
    language_config = {
        "Python": [".py"],
        "C#": [".cs"],
        "C++": [".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"],
    }

    def should_exclude_path(path: Path) -> bool:
        """
        Return True if the given path (file or dir) should be skipped.
        Checks both standard EXCLUDED_DIRS and user-provided excluded_paths.
        """
        # check against user excluded paths
        path_resolved = path.resolve()
        if path_resolved in user_excluded:
            return True

        # check against standard EXCLUDED_DIRS
        for part in path.parts:
            if part in EXCLUDED_DIRS:
                return True
        return False

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_path(Path(root) / d)]

        for file_name in files:
            file_path = Path(root) / file_name

            # skip if the file itself is in an excluded path
            if should_exclude_path(file_path):
                continue

            if not any(file_name.endswith(ext) for ext in allowed_extensions):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    file_content = file.read()

                yield DataPoint(
                    data={
                        "relative_path": str(file_path.relative_to(repo_path)),
                        "file_name": file_name,
                        "file_content": file_content,
                    },
                )

            except (UnicodeDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"Skipping {file_path}: {e}")


def get_repo_id(repo_name: str) -> str:
    """
    Generate a unique repository ID using UUID5.
    """
    return str(uuid5(NAMESPACE_OID, repo_name))


async def process_files_in_chunks(
    repo_name: str,
    file_generator: AsyncGenerator[DataPoint, None],
    chunk_size: int = 10,
):
    """
    Processes files from the file_generator in chunks of chunk_size.

    Args:
        repo_name: Name of the repository
        file_generator: AsyncGenerator that yields DataPoints with file info
        chunk_size: Number of files to process in each chunk

    Yields:
        Chunks of CodeFile objects (via DataPoint)
    """
    chunk = []
    async for file_data_point in file_generator:
        chunk.append(file_data_point)
        if len(chunk) >= chunk_size:
            yield DataPoint(
                data=await process_file_chunk(repo_name, chunk),
            )
            chunk = []

    # process remaining files in the last incomplete chunk
    if chunk:
        yield DataPoint(
            data=await process_file_chunk(repo_name, chunk),
        )


async def process_file_chunk(repo_name: str, chunk: list[DataPoint]) -> list[CodeFile]:
    """
    Processes a single chunk of files, extracting dependencies.

    Args:
        repo_name: Name of the repository
        chunk: List of DataPoints containing file information

    Returns:
        List of CodeFile objects with extracted dependencies
    """
    # Python dependency extractor
    from .get_local_dependencies import get_local_script_dependencies
    # C# dependency extractor
    from .get_csharp_dependencies import get_csharp_script_dependencies
    # C++ dependency extractor
    from .get_cpp_dependencies import get_cpp_script_dependencies

    tasks = []
    for file_data_point in chunk:
        file_data = file_data_point.data
        file_path = Path(file_data["relative_path"])
        file_name = file_data["file_name"]
        file_content = file_data["file_content"]

        # Determine file language based on extension
        if file_name.endswith(".py"):
            # Python file - use existing extractor
            tasks.append(
                get_local_script_dependencies(
                    file_path=file_path,
                    file_content=file_content,
                )
            )
        elif file_name.endswith(".cs"):
            # C# file - use new C# extractor
            tasks.append(
                asyncio.to_thread(
                    get_csharp_script_dependencies,
                    file_path=file_path,
                    file_content=file_content,
                )
            )
        elif any(file_name.endswith(ext) for ext in [".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"]):
            # C++ file - use new C++ extractor
            tasks.append(
                asyncio.to_thread(
                    get_cpp_script_dependencies,
                    file_path=file_path,
                    file_content=file_content,
                )
            )
        else:
            # Unsupported language - create minimal CodeFile
            tasks.append(
                asyncio.to_thread(
                    lambda fp=file_path, fn=file_name: CodeFile(
                        file_path=str(fp),
                        file_name=fn,
                        code_parts=[],
                        dependencies=[],
                    )
                )
            )

    results = await asyncio.gather(*tasks)
    return results


async def get_repo_file_dependencies(
    dataset: AsyncGenerator[DataPoint, None],
    batch_size: int = 10,
) -> AsyncGenerator[DataPoint, None]:
    """
    Extracts file dependencies from a dataset of repository files.

    Args:
        dataset: AsyncGenerator of DataPoints, each containing repository metadata
        batch_size: Number of files to process in each batch

    Yields:
        DataPoint containing Repository object with processed files
    """
    async for data_point in dataset:
        # unpack data from the incoming data_point
        repo_name = data_point.data["repo_name"]
        repo_path = data_point.data["repo_path"]
        excluded_paths = data_point.data.get("excluded_paths", None)
        language_config = data_point.data.get("language_config", None)

        repo_id = get_repo_id(repo_name)

        file_generator = get_source_code_files(
            repo_path,
            language_config=language_config,
            excluded_paths=excluded_paths,
        )

        # process files in chunks
        all_files = []
        async for chunk_data_point in process_files_in_chunks(
            repo_name, file_generator, batch_size
        ):
            all_files.extend(chunk_data_point.data)

        repository = Repository(
            id=repo_id,
            name=repo_name,
            path=str(repo_path),
            files=all_files,
        )

        yield DataPoint(data=repository)
