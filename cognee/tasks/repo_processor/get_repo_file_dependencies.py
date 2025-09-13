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
    while excluding test files and files within a virtual environment. It returns a list of
    absolute paths to the source code files that are not empty.

    Parameters:
    -----------
    - repo_path: Root path of the repository to search
    - language_config: dict mapping language names to file extensions, e.g.,
            {'python': ['.py'], 'javascript': ['.js', '.jsx'], ...}
    - excluded_paths: Optional list of path fragments or glob patterns to exclude

    Returns:
    --------
        A list of (absolute_path, language) tuples for source code files.
    """

    def _get_language_from_extension(file, language_config):
        for lang, exts in language_config.items():
            for ext in exts:
                if file.endswith(ext):
                    return lang
        return None

    # Default config if not provided
    if language_config is None:
        language_config = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "csharp": [".cs"],
            "go": [".go"],
            "rust": [".rs"],
            "cpp": [".cpp", ".c", ".h", ".hpp"],
        }

    if not os.path.exists(repo_path):
        return []

    source_code_files = set()
    for root, _, files in os.walk(repo_path):
        for file in files:
            lang = _get_language_from_extension(file, language_config)
            if lang is None:
                continue
            # Exclude tests, common build/venv directories and files provided in exclude_paths
            excluded_dirs = EXCLUDED_DIRS
            excluded_paths = {Path(p).resolve() for p in (excluded_paths or [])}  # full paths

            root_path = Path(root).resolve()
            root_parts = set(root_path.parts)  # same as before
            base_name, _ext = os.path.splitext(file)
            if (
                base_name.startswith("test_")
                or base_name.endswith("_test")
                or ".test." in file
                or ".spec." in file
                or (excluded_dirs & root_parts)  # name match
                or any(
                    root_path.is_relative_to(p)  # full-path match
                    for p in excluded_paths
                )
            ):
                continue
            file_path = os.path.abspath(os.path.join(root, file))
            if os.path.getsize(file_path) == 0:
                continue
            source_code_files.add((file_path, lang))

    return sorted(list(source_code_files))


def run_coroutine(coroutine_func, *args, **kwargs):
    """
    Run a coroutine function until it completes.

    This function creates a new asyncio event loop, sets it as the current loop, and
    executes the given coroutine function with the provided arguments. Once the coroutine
    completes, the loop is closed. Intended for use in environments where an existing event
    loop is not available or desirable.

    Parameters:
    -----------

        - coroutine_func: The coroutine function to be run.
        - *args: Positional arguments to pass to the coroutine function.
        - **kwargs: Keyword arguments to pass to the coroutine function.

    Returns:
    --------

        The result returned by the coroutine after completion.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine_func(*args, **kwargs))
    loop.close()
    return result


async def get_repo_file_dependencies(
    repo_path: str,
    detailed_extraction: bool = False,
    supported_languages: list = None,
    excluded_paths: Optional[List[str]] = None,
) -> AsyncGenerator[DataPoint, None]:
    """
    Generate a dependency graph for source files (multi-language) in the given repository path.

    Check the validity of the repository path and yield a repository object followed by the
    dependencies of source files within that repository. Raise a FileNotFoundError if the
    provided path does not exist. The extraction of detailed dependencies can be controlled
    via the `detailed_extraction` argument. Languages considered can be restricted via
    the `supported_languages` argument.

    Parameters:
    -----------

        - repo_path (str): The file path to the repository to process.
        - detailed_extraction (bool): Whether to perform a detailed extraction of code parts.
        - supported_languages (list | None): Subset of languages to include; if None, use defaults.
    """

    if isinstance(repo_path, list) and len(repo_path) == 1:
        repo_path = repo_path[0]

    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path {repo_path} does not exist.")

    # Build language config from supported_languages
    default_language_config = {
        "python": [".py"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "java": [".java"],
        "csharp": [".cs"],
        "go": [".go"],
        "rust": [".rs"],
        "cpp": [".cpp", ".c", ".h", ".hpp"],
        "c": [".c", ".h"],
    }
    if supported_languages is not None:
        language_config = {
            k: v for k, v in default_language_config.items() if k in supported_languages
        }
    else:
        language_config = default_language_config

    source_code_files = await get_source_code_files(
        repo_path, language_config=language_config, excluded_paths=excluded_paths
    )

    repo = Repository(
        id=uuid5(NAMESPACE_OID, repo_path),
        path=repo_path,
    )

    yield repo

    chunk_size = 100
    number_of_chunks = math.ceil(len(source_code_files) / chunk_size)
    chunk_ranges = [
        (
            chunk_number * chunk_size,
            min((chunk_number + 1) * chunk_size, len(source_code_files)) - 1,
        )
        for chunk_number in range(number_of_chunks)
    ]

    # Import dependency extractors for each language (Python for now, extend later)
    from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies
    import aiofiles
    # TODO: Add other language extractors here

    for start_range, end_range in chunk_ranges:
        tasks = []
        for file_path, lang in source_code_files[start_range : end_range + 1]:
            # For now, only Python is supported; extend with other languages
            if lang == "python":
                tasks.append(
                    get_local_script_dependencies(repo_path, file_path, detailed_extraction)
                )
            else:
                # Placeholder: create a minimal CodeFile for other languages
                async def make_codefile_stub(file_path=file_path, lang=lang):
                    async with aiofiles.open(
                        file_path, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        source = await f.read()
                    return CodeFile(
                        id=uuid5(NAMESPACE_OID, file_path),
                        name=os.path.relpath(file_path, repo_path),
                        file_path=file_path,
                        language=lang,
                        source_code=source,
                    )

                tasks.append(make_codefile_stub())

        results: list[CodeFile] = await asyncio.gather(*tasks)

        for source_code_file in results:
            source_code_file.part_of = repo
            if getattr(
                source_code_file, "language", None
            ) is None and source_code_file.file_path.endswith(".py"):
                source_code_file.language = "python"
            yield source_code_file
