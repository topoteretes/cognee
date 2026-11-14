"""Markdown task-list connector for cognee: ``- [ ]`` / ``- [x]`` checklists as documents.

People manage personal and project tasks in Markdown files, outside any task application.
This source walks a folder of ``.md`` files, extracts the checklist items of every file and
publishes one document per file — the tasks, open and done, with the file's folders as
structure — with a content-hash id, so an unchanged file is not re-cognified and a file
whose tasks were all removed (or that was deleted) is forgotten on the next sync. The
connector is read-only: it never writes the source files. See ``markdown_tasks.md``.
"""

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

logger = get_logger("markdown_tasks_connector")

# A checklist line: leading indentation (nesting), the box, the task text. The box
# marker itself is excluded, so the match covers ``- [ ]``, ``* [x]`` and ``+ [X]``.
_TASK_LINE = re.compile(r"^(?P<indent>\s*)[-*+]\s+\[(?P<box>[ xX])\]\s*(?P<text>.*)$")


class MarkdownTasksError(RuntimeError):
    """Base error. Messages carry folder-relative paths only, never absolute paths."""


class MarkdownTasksUnavailableError(MarkdownTasksError):
    """The folder is gone at run time, or it is empty though files were synced."""


class MarkdownTasksReadError(MarkdownTasksError):
    """A task file could not be read; the run stops before anything is forgotten."""


def _tasks_of(text: str) -> list[tuple[str, str, bool]]:
    """The checklist items of one file, in file order: (indent, task text, done).

    The indentation is kept apart from the text, so the rendering can nest a sub-task
    under its parent. Every non-checklist line (prose, headings, frontmatter) is not a
    task and is not carried over. A fenced code block is skipped: a README that
    *documents* the checklist syntax with an example must not become tasks.
    """
    tasks: list[tuple[str, str, bool]] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TASK_LINE.match(line)
        if match is None:
            continue
        tasks.append(
            (match.group("indent"), match.group("text").strip(), match.group("box") in "xX")
        )
    return tasks


def _document_text(rel_path: str, tasks: list[tuple[str, str, bool]]) -> str:
    """One file's markdown body: its tasks as a checklist, nesting preserved."""
    title = PurePosixPath(rel_path).stem
    lines = [f"# {title}"] + [
        f"{indent}- [{'x' if done else ' '}] {text}" for indent, text, done in tasks
    ]
    return "\n".join(lines)


def _file_id(rel_path: str, body: str) -> str:
    return "markdown-tasks:" + hashlib.sha256(f"{rel_path}\n{body}".encode()).hexdigest()[:12]


def _ancestors(rel_path: str) -> list[dict]:
    folder = PurePosixPath(rel_path).parent
    return [
        {"kind": "folder", "id": parent.as_posix(), "name": parent.name}
        for parent in [folder, *folder.parents]
        if parent.parts
    ]


def _list_files(root: Path) -> list[str]:
    """Root-relative posix paths of the ``*.md`` files to ingest, sorted.

    The same listing rules as the Obsidian source: regular files only, nothing hidden, no
    symlinks followed. A folder that cannot be listed raises, so an unreadable root never
    looks like a deleted set of files.
    """

    def refuse(error: OSError) -> None:
        rel = os.path.relpath(error.filename, root) if error.filename else error.strerror
        raise MarkdownTasksUnavailableError(
            f"Could not list the folder {rel!r} ({type(error).__name__}); "
            "nothing was synced and nothing was forgotten."
        ) from None

    files = []
    for directory, subdirs, names in os.walk(root, onerror=refuse):
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        for name in names:
            path = os.path.join(directory, name)
            if (
                name.lower().endswith(".md")
                and not name.startswith(".")
                and not os.path.islink(path)
                and os.path.isfile(path)
            ):
                files.append(Path(path).relative_to(root).as_posix())
    return sorted(files)


def _iter_rows(
    root: Path,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a document row for every file that holds tasks, then a tombstone for every gone one.

    Every file is read before the first row is yielded, so an unreadable file stops the run
    with nothing published and nothing forgotten. A file with no checklist items publishes
    no row: it is not a task file, and if it held tasks before, they are forgotten the same
    way a deleted file's are. ``state`` is written after the last row.
    """
    if not root.is_dir():
        raise MarkdownTasksUnavailableError(
            "The task folder is not available; nothing was synced and nothing was forgotten. "
            "Check that the drive is mounted or point source_path at the new location."
        )
    listed = _list_files(root)
    previous = set(state.get("members", []))
    if previous and not os.listdir(root):
        raise MarkdownTasksUnavailableError(
            "The task folder is empty, but earlier runs synced files from it; nothing was "
            "synced and nothing was forgotten. A drive that did not mount looks exactly like "
            "this. Check the mount, or leave one file in the folder to forget everything on "
            "purpose."
        )

    rows: list[dict[str, Any]] = []
    present: set[str] = set()
    for rel_path in listed:
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a failed file
        try:
            text = (root / rel_path).read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as error:
            raise MarkdownTasksReadError(
                f"Could not read the task file {rel_path!r} ({type(error).__name__}); nothing "
                "was synced and nothing was forgotten."
            ) from None
        tasks = _tasks_of(text)
        if not tasks:
            continue
        body = _document_text(rel_path, tasks)
        file_id = _file_id(rel_path, body)
        present.add(file_id)
        row: dict[str, Any] = {
            "id": file_id,
            "title": PurePosixPath(rel_path).stem,
            "content": body,
            "url": "",
            "_deleted": False,
        }
        ancestors = _ancestors(rel_path)
        if ancestors:
            row[STRUCTURE_COLUMN] = {"ancestors": ancestors}
        sets = set()
        if any(not done for _, _, done in tasks):
            sets.add("markdown-tasks:open")
        if any(done for _, _, done in tasks):
            sets.add("markdown-tasks:done")
        if sets:
            # recall(node_name=["markdown-tasks:open"]) scopes a search to the files
            # that still hold open tasks.
            row[NODE_SET_COLUMN] = sorted(sets)
        rows.append(row)

    for row in rows:
        yield row
        # After the yield: a row that check_active stopped before it was published is not counted.
        stats["synced"] += 1
    for file_id in sorted(previous - present):
        yield {"id": file_id, "_deleted": True}
        stats["deleted"] += 1
    state.update(v=1, members=sorted(present))
    logger.info(
        "Markdown tasks: %d file(s) synced, %d forgotten.",
        stats["synced"],
        stats["deleted"],
    )


def markdown_tasks_source(
    *,
    source_path: str | os.PathLike[str],
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per Markdown file that holds tasks.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. A run walks the whole folder; a file whose tasks did not change
    makes the same row id, so nothing is re-cognified. A file removed from the folder, or
    one whose checklists were all removed, is forgotten on the next run.

    Args:
        source_path: The folder walked recursively. A ``ValueError`` if it does not exist.
            A run raises if it has disappeared by then.
        resource_name: Names the staging table and the sync state. The default is derived
            from the resolved path, so moving the folder changes it and ingests everything
            again; pass a name to pin it. Two folders in one dataset need different names.
        check_active: Host authorization checkpoint, called before each file is read and
            before each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Markdown tasks sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    root = Path(source_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("source_path must be an existing directory.")
    if resource_name is None:
        resource_name = f"markdown_tasks_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def task_files():
        stats.update(synced=0, deleted=0)
        rows = _iter_rows(root, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = task_files()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "markdown_tasks")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
