"""Select exact paths from a commit tree without downloading its blobs."""

import json
import os
import re
from pathlib import Path, PurePosixPath

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
    ignored_dirs = SKIP_DIRS | {".git", ".enola"}
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
        elif any(part in ignored_dirs for part in path.split("/")):
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
                message="A selected repository path contains a newline; sparse checkout is unsafe."
            )
        escaped = re.sub(r"([\\*?\[\] ])", r"\\\1", path)
        patterns.append("/" + escaped)
    info = repo / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text(
        "\n".join(patterns or ["!/*"]) + "\n", encoding="utf-8", errors="surrogateescape"
    )


def check_selected_content(repo: Path, paths: list[str]) -> None:
    """LFS is never expanded implicitly or indexed as if its pointer were the content."""
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    prefix = b"version https://git-lfs.github.com/spec/v1"
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
            raise CodeRepositoryError(
                message="A selected file is stored in Git LFS. Materialize it in a local "
                "checkout before ingestion; automatic LFS downloads are disabled."
            )


def read_repository_snapshot(repo: Path) -> dict | None:
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    path = repo / ".git" / SNAPSHOT_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CodeRepositoryError(
            message="The clone's snapshot record is unreadable. Remove the clone from the "
            "clones directory to clone the repository again."
        ) from error


def write_repository_snapshot(repo: Path, snapshot: dict) -> None:
    """Replace the record whole, so an interrupted write never leaves a truncated one."""
    path = repo / ".git" / SNAPSHOT_FILE
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(snapshot), encoding="utf-8")
    os.replace(pending, path)
