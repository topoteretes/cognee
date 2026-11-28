"""Markdown Task-List connector for Cognee: extract and index markdown checklist tasks into memory.

Reads local Markdown (.md) files, extracts checklist items written with standard Markdown
syntax (``- [ ]`` and ``- [x]`` or ``* [ ]`` / ``+ [ ]``), and yields them as structured
document rows for :func:`cognee.remember`::

    import cognee
    from cognee.tasks.ingestion.connectors import markdown_task_list_source

    await cognee.remember(
        markdown_task_list_source("/path/to/project_notes"),
        dataset_name="project_tasks",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Rows** are flat ``{id, title, content, _deleted, cognee_node_set}``.
  Task IDs are formatted as ``mdtask:<relative_path>:<line_number>``.
* **Context Preservation**: Preserves section headings, completion status (Pending vs Completed),
  source relative file paths, and line numbers.
* **Tombstones**: Tasks removed from markdown files emit ``_deleted: True`` under
  ``write_disposition="merge"`` so Cognee's orphan cleanup updates graph memory.
"""

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = get_logger("markdown_task_list_connector")

# Matches checklist items: - [ ], - [x], * [ ], * [X], + [ ], + [x]
_CHECKLIST_RE = re.compile(r"^[ \t]*[-*+][ \t]+\[([ xX])\][ \t]+(.*)$")
# Matches Markdown headings: # Section, ## Sub-section
_HEADING_RE = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.*)$")

SKIP_DIRS = frozenset(
    {".git", ".svn", ".hg", ".obsidian", ".trash", "node_modules", ".venv", "__pycache__"}
)


class MarkdownTaskSourceError(RuntimeError):
    """Base error for markdown task list connector."""


class MarkdownTaskPathNotFoundError(MarkdownTaskSourceError):
    """Raised when the specified file or directory path does not exist."""


def parse_markdown_tasks(content: str, rel_path: str = "") -> list[dict[str, Any]]:
    """Parse markdown text and extract all checkbox task items with context."""
    # Remove BOM if present
    text = content.removeprefix("\ufeff")
    lines = text.splitlines()

    tasks = []
    current_section = "General"

    for idx, line in enumerate(lines, start=1):
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            current_section = heading_match.group(2).strip()
            continue

        task_match = _CHECKLIST_RE.match(line)
        if task_match:
            box = task_match.group(1)
            is_completed = box in ("x", "X")
            task_desc = task_match.group(2).strip()
            status_str = "Completed" if is_completed else "Pending"
            box_prefix = "[x]" if is_completed else "[ ]"

            task_id = f"mdtask:{rel_path}:{idx}"
            title = f"{box_prefix} {task_desc}"
            if rel_path:
                title = f"{title} ({os.path.basename(rel_path)})"

            doc_content = (
                f"Task: {task_desc}\n"
                f"Status: {status_str}\n"
                f"Section: {current_section}\n"
                f"File: {rel_path or 'document.md'}\n"
                f"Line: {idx}"
            )

            tasks.append(
                {
                    "id": task_id,
                    "title": title,
                    "content": doc_content,
                    "completed": is_completed,
                    "section": current_section,
                    "line": idx,
                    "raw_text": line.strip(),
                }
            )

    return tasks


def _iter_task_rows(
    root: Path,
    state: dict[str, Any],
    stats: dict[str, int],
    node_set: str,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Walk markdown files, parse tasks, and emit documents with deletion detection."""
    known_ids = dict(state.get("known_ids", {}))
    current_ids = {}

    files_to_scan = []
    if root.is_file():
        files_to_scan.append(root)
    else:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for filename in filenames:
                if filename.lower().endswith(".md"):
                    files_to_scan.append(Path(dirpath) / filename)

    files_to_scan.sort()

    for file_path in files_to_scan:
        if check_active is not None:
            check_active()

        try:
            rel_path = file_path.relative_to(root).as_posix() if root.is_dir() else file_path.name
        except ValueError:
            rel_path = file_path.name

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            logger.warning("Failed to read markdown file %s: %s", file_path, err)
            stats["failed_read"] = stats.get("failed_read", 0) + 1
            continue

        tasks = parse_markdown_tasks(content, rel_path=rel_path)

        for task in tasks:
            task_id = task["id"]
            content_hash = hashlib.sha256(task["content"].encode("utf-8")).hexdigest()
            current_ids[task_id] = content_hash

            # Yield updated or new task
            if known_ids.get(task_id) != content_hash:
                row = {
                    "id": task_id,
                    "title": task["title"],
                    "content": task["content"],
                    "_deleted": False,
                    NODE_SET_COLUMN: node_set,
                }
                stats["synced"] = stats.get("synced", 0) + 1
                yield row

    # Emit tombstones for removed tasks
    for old_id in known_ids:
        if old_id not in current_ids:
            stats["deleted"] = stats.get("deleted", 0) + 1
            yield {
                "id": old_id,
                "title": "",
                "content": "",
                "_deleted": True,
                NODE_SET_COLUMN: node_set,
            }

    state["known_ids"] = current_ids


def markdown_task_list_source(
    path: str | Path,
    resource_name: str | None = None,
    node_set: str = "markdown_tasks",
    check_active: Callable[[], None] | None = None,
) -> Any:
    """Return a dlt resource that yields one document per Markdown checklist task item.

    Args:
        path: A local file or directory containing .md files.
        resource_name: Unique resource identifier for dlt state. Defaults to a hash of path.
        node_set: Cognee node set name for graph scoping. Defaults to "markdown_tasks".
        check_active: Optional authorization checkpoint callback.
    """
    import dlt

    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        raise MarkdownTaskPathNotFoundError(f"Path does not exist: {path}")

    if resource_name is None:
        resource_name = (
            f"mdtasks_{hashlib.sha256(str(target_path).encode('utf-8')).hexdigest()[:12]}"
        )

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def markdown_tasks():
        stats.update(synced=0, deleted=0, failed=0, failed_read=0)
        state = dlt.current.resource_state()
        rows = _iter_task_rows(target_path, state, stats, node_set, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    res = markdown_tasks()
    setattr(res, DOCUMENT_SOURCE_ATTR, "markdown_task_list")
    setattr(res, PIPELINE_SCOPE_ATTR, resource_name)
    res.cognee_sync_stats = stats
    return res
