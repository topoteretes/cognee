"""Select exact paths from a commit tree without downloading its blobs."""

import fnmatch
import functools
import hashlib
import json
import os
import re
import unicodedata
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


def validate_path_patterns(
    patterns: list[str] | None, name: str, *, optional: bool = False
) -> None:
    """Validate POSIX globs: ** spans components; *, ? and [] match within one.

    ``optional`` accepts None, which means no filter (include_paths) rather than an empty list.
    """
    from cognee.exceptions import CogneeValidationError

    if patterns is None and optional:
        return
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise CogneeValidationError(f"{name} must be a list of path patterns.")
    for pattern in patterns:
        parts = pattern.split("/")
        if (
            not pattern
            or "\\" in pattern
            or any(ord(char) < 32 or ord(char) == 127 for char in pattern)
            or any(part in ("", ".", "..") for part in parts)
            or re.match(r"^[A-Za-z]:", pattern)
            or any("**" in part and part != "**" for part in parts)
        ):
            raise CogneeValidationError(
                f"Invalid {name} pattern {pattern!r}: use relative POSIX globs."
            )


def path_matches(path: str, pattern: str) -> bool:
    """Match a full relative path, case-sensitively, with ** matching zero or more folders."""
    parts, patterns = path.split("/"), pattern.split("/")
    if "**" not in patterns:
        return len(parts) == len(patterns) and all(
            fnmatch.fnmatchcase(part, rule) for part, rule in zip(parts, patterns)
        )

    @functools.cache
    def match(position: int, rule: int) -> bool:
        if rule == len(patterns):
            return position == len(parts)
        if patterns[rule] == "**":
            return match(position, rule + 1) or (
                position < len(parts) and match(position + 1, rule)
            )
        return (
            position < len(parts)
            and fnmatch.fnmatchcase(parts[position], patterns[rule])
            and match(position + 1, rule + 1)
        )

    return match(0, 0)


def select_tree(
    tree: str,
    include_documents: bool,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Select paths using the same complete-tree rules as preview_tree, counting what is skipped."""
    selected = []
    counts = dict.fromkeys(_skip_reasons(include_paths, exclude_paths), 0)
    for path, reason in _classify_tree(tree, include_documents, include_paths, exclude_paths):
        if reason is None:
            selected.append(path)
        else:
            counts[reason] += 1
    return selected, counts


def preview_tree(
    tree: str,
    include_documents: bool = True,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> dict:
    """Preview a complete NUL-delimited tree listing without fetching blobs.

    Returns selected paths and skipped paths grouped by reason. Exclusions win.
    Unknown extensions remain eligible for custom extractors. Symlinks and
    submodules never qualify. LFS pointers cannot be detected without blobs.
    """
    selected = []
    skipped = {reason: [] for reason in _skip_reasons(include_paths, exclude_paths)}
    for path, reason in _classify_tree(tree, include_documents, include_paths, exclude_paths):
        (selected if reason is None else skipped[reason]).append(path)
    return {"selected": selected, "skipped": skipped}


def _skip_reasons(include_paths: list[str] | None, exclude_paths: list[str] | None) -> list[str]:
    reasons = ["ignored_path", "binary", "symlink", "submodule"]
    if include_paths is not None or exclude_paths:
        reasons += ["not_included", "excluded"]
    return reasons


def _classify_tree(
    tree: str,
    include_documents: bool,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
):
    """Yield ``(path, reason)`` for every tree entry; the reason is None when it is selected.

    Invalid patterns and an incomplete listing raise here, at the call, not at the first item.
    """
    validate_path_patterns(include_paths, "include_paths", optional=True)
    exclude_paths = [] if exclude_paths is None else exclude_paths
    validate_path_patterns(exclude_paths, "exclude_paths")
    if tree and not tree.endswith("\0"):
        from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

        raise CodeRepositoryError(message="The repository tree listing is incomplete.")
    return _walk_tree(tree, include_documents, include_paths, exclude_paths)


def _walk_tree(
    tree: str,
    include_documents: bool,
    include_paths: list[str] | None,
    exclude_paths: list[str],
):
    from cognee.tasks.code_graph.code_repo import (
        REPO_SKIPPED_MEDIA_EXTENSIONS,
        SKIP_DIRS,
        _document_extensions,
    )
    from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

    documents = _document_extensions() if include_documents else frozenset()
    ignored_dirs = {name.lower() for name in SKIP_DIRS | {".git", ".enola"}}
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
            yield path, "submodule"
        elif mode == "120000" and kind == "blob":
            yield path, "symlink"
        elif mode not in ("100644", "100755") or kind != "blob":
            raise CodeRepositoryError(message="The repository tree contains an unsupported entry.")
        # Keep tracked dotfiles: enola needs ignore rules and hidden CI/config
        # inputs too. The document ingestion walker applies its own hidden-path
        # exclusions later. Version-control metadata and generated snapshots are never inputs.
        # Directories match at any depth and in any letter case (the checkout can be on a
        # case-insensitive filesystem); a file that merely has such a name is a file.
        elif any(part.lower() in ignored_dirs for part in path.split("/")[:-1]) or (
            path.rsplit("/", 1)[-1].lower() in (".git", ".enola")
        ):
            yield path, "ignored_path"
        else:
            extension = PurePosixPath(path).suffix.lstrip(".").lower()
            if extension in REPO_SKIPPED_MEDIA_EXTENSIONS or (
                extension in _BINARY_EXTENSIONS and extension not in documents
            ):
                yield path, "binary"
            elif any(path_matches(path, pattern) for pattern in exclude_paths):
                yield path, "excluded"
            elif include_paths is not None and not any(
                path_matches(path, pattern) for pattern in include_paths
            ):
                yield path, "not_included"
            else:
                yield path, None


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


def _folded(path: str) -> str:
    """A path as a case-insensitive, normalisation-insensitive filesystem would compare it."""
    return unicodedata.normalize("NFC", path).casefold()


def remove_unselected_files(repo: Path, paths: list[str]) -> int:
    """Delete every file of the work tree that is not in the selection and return how many.

    The sparse patterns select exactly these paths, but git leaves a file in place when its
    path stops being a file (a file replaced by a submodule keeps its old content). Paths are
    compared folded: on a case-insensitive filesystem git writes ``Src/a.py`` and ``src/b.py``
    into one directory, and a selected file must not be deleted for the case of its name.
    The root ``.git`` and Enola's ``.enola`` output are not the selection's to remove.
    """
    selected = {_folded(path) for path in paths}
    removed = 0
    directories_seen = []
    for directory, directories, files in os.walk(repo):
        if Path(directory) == repo:
            directories[:] = [name for name in directories if name not in (".git", ".enola")]
        directories_seen.append(Path(directory))
        for name in files:
            full = Path(directory) / name
            if _folded(full.relative_to(repo).as_posix()) not in selected:
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
