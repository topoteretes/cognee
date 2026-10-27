"""Select exact paths from a commit tree without downloading its blobs."""

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

from fastapi import status

SNAPSHOT_FILE = "cognee-snapshot.json"

# These are artifacts, not inputs to enola. Registered document loaders still
# take precedence when add() requests documents (images, PDFs, archives, etc.).
_BINARY_EXTENSIONS = frozenset(
    [
        "bin",
        "exe",
        "dll",
        "so",
        "dylib",
        "a",
        "o",
        "obj",
        "pyc",
        "pyo",
        "class",
        "jar",
        "war",
        "ear",
        "zip",
        "tar",
        "gz",
        "bz2",
        "xz",
        "7z",
        "rar",
        "whl",
        "wasm",
        "woff",
        "woff2",
        "ttf",
        "eot",
        "otf",
        "ico",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "bmp",
        "tiff",
        "tif",
        "avif",
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "parquet",
        "arrow",
        "db",
        "sqlite",
        "sqlite3",
        "pkl",
        "pickle",
        "npy",
        "npz",
        "pt",
        "pth",
        "onnx",
        "safetensors",
        "h5",
        "hdf5",
        "mp3",
        "mp4",
        "mov",
        "avi",
        "wav",
        "flac",
        "ogg",
        "webm",
    ]
)
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def select_tree(tree: str, include_documents: bool) -> tuple[list[str], dict[str, int]]:
    """Read the complete NUL-delimited ls-tree output; malformed records fail closed.

    Unknown extensions are retained: custom enola extractors and manifests may
    need them. There is no file-size cutoff. Symlinks and submodules are explicitly
    excluded, and code-only still retains text (including enola's markdown inputs).
    """
    from cognee.tasks.code_graph.code_repo import (
        REPO_SKIPPED_MEDIA_EXTENSIONS,
        SKIP_DIRS,
        _document_extensions,
    )
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    documents = _document_extensions() if include_documents else frozenset()
    ignored_dirs = {name.lower() for name in SKIP_DIRS | {".git", ".enola"}}
    selected = []
    skipped = {"ignored_path": 0, "binary": 0, "symlink": 0, "submodule": 0}
    if tree and not tree.endswith("\0"):
        raise CodeRepositoryError(message="The repository tree listing is incomplete.")
    seen = set()
    for record in tree.split("\0")[:-1]:
        header, separator, path = record.partition("\t")
        fields = header.split()
        if (
            not separator
            or len(fields) != 3
            or not path
            or path in seen
            or path.startswith("/")
            or any(part in ("", ".", "..") for part in path.split("/"))
            or not _OBJECT_ID.fullmatch(fields[-1])
        ):
            raise CodeRepositoryError(message="The repository tree listing is invalid.")
        seen.add(path)
        mode, kind, _oid = fields
        if mode == "160000" and kind == "commit":
            skipped["submodule"] += 1
        elif mode == "120000" and kind == "blob":
            skipped["symlink"] += 1
        elif mode not in ("100644", "100755") or kind != "blob":
            raise CodeRepositoryError(message="The repository tree contains an unsupported entry.")
        # Keep tracked dotfiles: enola needs ignore rules and hidden CI/config
        # inputs too. The document ingestion walker applies its own hidden-path
        # exclusions later. Git metadata and generated snapshots are never inputs.
        # Directories match at any depth and in any letter case (the checkout can be on a
        # case-insensitive filesystem); a file that merely has such a name is a file.
        elif any(part.lower() in ignored_dirs for part in path.split("/")[:-1]) or (
            path.rsplit("/", 1)[-1].lower() in (".git", ".enola")
        ):
            skipped["ignored_path"] += 1
        else:
            extension = PurePosixPath(path).suffix.lstrip(".").lower()
            if extension in REPO_SKIPPED_MEDIA_EXTENSIONS or (
                extension in _BINARY_EXTENSIONS and extension not in documents
            ):
                skipped["binary"] += 1
            else:
                selected.append(path)
    return selected, skipped


def write_sparse_patterns(repo: Path, paths: list[str]) -> None:
    """Write anchored literal paths; glob characters must not select extra blobs."""
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    patterns = []
    for path in paths:
        if "\n" in path or "\r" in path:
            raise CodeRepositoryError(
                message="A selected repository path contains a newline; sparse checkout is unsafe.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        escaped = re.sub(r"([\\*?\[\] ])", r"\\\1", path)
        patterns.append("/" + escaped)
    info = repo / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text(
        "\n".join(patterns or ["!/*"]) + "\n", encoding="utf-8", errors="surrogateescape"
    )


def remove_unselected_files(repo: Path, paths: list[str]) -> int:
    """Delete every file of the work tree that is not in the selection and return how many.

    The sparse patterns select exactly these paths, but git leaves a file in place when its
    path stops being a file (a file replaced by a submodule keeps its old content).
    """
    selected = set(paths)
    removed = 0
    directories_seen = []
    for directory, directories, files in os.walk(repo):
        if Path(directory) == repo:
            directories[:] = [name for name in directories if name != ".git"]
        directories_seen.append(Path(directory))
        for name in files:
            full = Path(directory) / name
            if full.relative_to(repo).as_posix() not in selected:
                full.unlink()
                removed += 1
    for directory in reversed(directories_seen):
        if directory != repo and not any(directory.iterdir()):
            directory.rmdir()
    return removed


def remove_lfs_pointers(repo: Path, paths: list[str]) -> int:
    """Delete the selected files that are Git LFS pointers and return how many.

    LFS is never expanded implicitly and a pointer is not the file's content, so it must not
    be indexed as if it were: a pointer is removed from the checkout and counted as skipped.
    """
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    prefix = b"version https://git-lfs.github.com/spec/v1"
    removed = 0
    for path in paths:
        try:
            with (repo / path).open("rb") as file:
                head = file.read(len(prefix))
        except FileNotFoundError as error:
            raise CodeRepositoryError(
                message="A selected repository path was not written by git (a name this "
                "platform cannot represent, or a path that is too long)."
            ) from error
        if head == prefix:
            (repo / path).unlink()
            removed += 1
    return removed


def _snapshot_path(repo: Path, *, pending: bool = False) -> Path:
    """Keep snapshot metadata inside the checkout, including through symlinks."""
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    root = os.path.realpath(repo)
    filename = SNAPSHOT_FILE + (".tmp" if pending else "")
    resolved = os.path.realpath(os.path.join(root, ".git", filename))
    if not resolved.startswith(root.rstrip(os.sep) + os.sep):
        raise CodeRepositoryError(message="The repository snapshot is outside its checkout.")
    return Path(resolved)


def read_repository_snapshot(repo: Path) -> dict | None:
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    path = _snapshot_path(repo)
    if not path.is_file():
        return None
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("Expected a snapshot object.")
        return snapshot
    except (ValueError, UnicodeError) as error:
        raise CodeRepositoryError(
            message="The clone's snapshot record is unreadable. Remove the clone from the "
            "clones directory to clone the repository again."
        ) from error


def write_repository_snapshot(repo: Path, snapshot: dict) -> None:
    """Replace the record whole, so an interrupted write never leaves a truncated one."""
    path = _snapshot_path(repo)
    pending = _snapshot_path(repo, pending=True)
    pending.write_text(json.dumps(snapshot), encoding="utf-8")
    os.replace(pending, path)


def selection_fingerprint(paths: list[str], include_documents: bool) -> str:
    """Identify the selection policy and exact paths independently of the commit."""
    value = {"version": 1, "include_documents": include_documents, "paths": sorted(paths)}
    return hashlib.sha256(json.dumps(value, ensure_ascii=True).encode()).hexdigest()
