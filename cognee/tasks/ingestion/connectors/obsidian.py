"""Obsidian connector for cognee: a ``dlt`` source that reads a local vault, one note per document.

``cognee.remember(<folder>)`` stores a note once: an edited note leaves its old version in
memory and a deleted note is never forgotten. This source gives every note a stable id (its
path in the vault) and walks the whole selection on every run, so an edit replaces the old
document and a note that left the selection is forgotten. See ``obsidian.md`` for what is read,
deletion and its gaps, and privacy.
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

logger = get_logger("obsidian_connector")

# Obsidian's own frontmatter: a "---" line first in the file, the YAML, then the next "---" line.
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
# Frontmatter keys that hold a note's tags. Obsidian's current key is "tags", "tag" is its
# older spelling.
_TAG_KEYS = ("tags", "tag")


class ObsidianSourceError(RuntimeError):
    """Base error. Messages carry vault-relative paths only, never absolute paths or text."""


class ObsidianVaultUnavailableError(ObsidianSourceError):
    """The vault folder is gone at run time, or its root is empty though notes were synced."""


class ObsidianListingError(ObsidianSourceError):
    """A folder of the vault could not be listed, so the listing is not complete."""


def _note_id(rel_path: str) -> str:
    return f"obsidian:{rel_path}"


def _frontmatter(text: str) -> dict:
    """The note's frontmatter as a dict; empty when there is none or it is not a YAML mapping."""
    # A file saved with a byte order mark still has frontmatter; the BOM stays in the content.
    match = _FRONTMATTER.match(text.removeprefix("\ufeff"))
    if match is None:
        return {}
    import yaml

    try:
        parsed = yaml.safe_load(match.group(1))
    except (yaml.YAMLError, ValueError, RecursionError, KeyError, AttributeError, IndexError):
        # ValueError too: a mistyped date such as "2024-13-45" matches YAML's timestamp
        # pattern and raises ValueError, not YAMLError. RecursionError too: deeply nested flow
        # YAML (500 opening brackets) overflows the parser's stack. KeyError, AttributeError
        # and IndexError too: PyYAML's SafeConstructor bool, timestamp, int and float
        # constructors raise them on an explicit tag with a bad value (``!!bool maybe``,
        # ``!!timestamp abc``, an empty ``!!int``). Any of them would otherwise abort the whole
        # run. Malformed frontmatter only costs the note its title and tags; its text is
        # stored as read. The try body is the one parser call, so nothing of ours is hidden.
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _title(meta: dict, rel_path: str) -> str:
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return PurePosixPath(rel_path).stem


def _node_set(meta: dict) -> list[str]:
    """``obsidian:tag:<tag>`` for every frontmatter tag, sorted and without repeats.

    A tag is a list entry or a single string (one tag, not split on commas). A leading ``#`` is
    dropped. Entries that are not strings, like the number in ``tags: [2024]``, are not tags
    in Obsidian either.
    """
    names = set()
    for key in _TAG_KEYS:
        value = meta.get(key)
        for tag in value if isinstance(value, list) else [value]:
            name = tag.strip().removeprefix("#").strip() if isinstance(tag, str) else ""
            if name:
                names.add(f"obsidian:tag:{name}")
    return sorted(names)


def _ancestors(rel_path: str) -> list[dict]:
    """The note's folders, nearest parent first, as ``cognee_structure`` wants them."""
    folder = PurePosixPath(rel_path).parent
    return [
        {"kind": "folder", "id": parent.as_posix(), "name": parent.name}
        for parent in [folder, *folder.parents]
        if parent.parts
    ]


def _note_row(rel_path: str, text: str) -> dict[str, Any]:
    """One document row. The content is the file as read, frontmatter included."""
    meta = _frontmatter(text)
    row: dict[str, Any] = {
        "id": _note_id(rel_path),
        "title": _title(meta, rel_path),
        "content": text,
        "url": "",
        "_deleted": False,
    }
    node_set = _node_set(meta)
    if node_set:
        row[NODE_SET_COLUMN] = node_set
    ancestors = _ancestors(rel_path)
    if ancestors:
        row[STRUCTURE_COLUMN] = {"ancestors": ancestors}
    return row


def _list_notes(vault: Path, folders: list[str] | None) -> list[str]:
    """Vault-relative posix paths of the notes to ingest, sorted.

    Only regular ``*.md`` files (no symlinks, no FIFOs or devices, which would block a read),
    and nothing whose name starts with a dot: not the directories ``.obsidian``, ``.trash`` and
    ``.git``, and not hidden files such as ``._note.md``. Symlinked directories are not entered.
    A folder that cannot be listed raises, because ``os.walk`` would skip it silently and the
    notes inside would then look deleted. So does a selected folder that exists on disk but that
    the walk never visited under that exact name (another letter case or Unicode form on a
    case-insensitive file system, a symlink): it would select nothing and forget every note.
    The empty-root check comes after the listing, so an unreadable root is a listing error.
    A selected folder that does not exist is a removed folder and forgets its notes.
    """

    def refuse(error: OSError) -> None:
        rel = os.path.relpath(error.filename, vault)
        # from None: the OSError text carries the absolute path.
        raise ObsidianListingError(
            f"Could not list the vault folder {rel!r} ({type(error).__name__}); "
            "nothing was synced and nothing was forgotten."
        ) from None

    notes = []
    visited = set()
    for directory, subdirs, files in os.walk(vault, onerror=refuse):
        visited.add(Path(directory).relative_to(vault).as_posix())
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        for name in files:
            path = os.path.join(directory, name)
            if (
                name.endswith(".md")
                and not name.startswith(".")
                and not os.path.islink(path)
                and os.path.isfile(path)
            ):
                notes.append(Path(path).relative_to(vault).as_posix())
    if folders is not None:
        for folder in folders:
            if folder in visited:
                continue
            # is_file before exists: a file is not a removed folder, and treating it as one
            # would forget every member.
            if (vault / folder).is_file():
                raise ObsidianListingError(
                    f"The selected folder {folder!r} is a file, not a folder; nothing was "
                    "synced and nothing was forgotten."
                )
            if (vault / folder).exists():
                raise ObsidianListingError(
                    f"The selected folder {folder!r} exists on disk but the vault has no folder "
                    "with exactly that name (another letter case, another Unicode form or a "
                    "symbolic link); nothing was synced and nothing was forgotten."
                )
        notes = [n for n in notes if any(n.startswith(f"{folder}/") for folder in folders)]
    return sorted(notes)


def _iter_rows(
    vault: Path,
    folders: list[str] | None,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a row for every readable note, then a tombstone for every member that is gone.

    The listing is complete before the first row, so an error there yields nothing and leaves
    ``state`` alone. A previous member the listing no longer contains was deleted, moved, or
    fell outside a narrowed ``folders``; all three are forgotten the same way. A note that
    cannot be read keeps the row it already has: it stays a member and is counted in
    ``failed_read``. A vault root with no entries at all is refused when notes were synced
    before. ``state`` is written after the last row.
    """
    if not vault.is_dir():
        raise ObsidianVaultUnavailableError(
            "The Obsidian vault folder is not available; nothing was synced and nothing was "
            "forgotten. Check that the drive is mounted or point vault_path at the new location."
        )
    listed = _list_notes(vault, folders)
    previous = set(state.get("members", []))
    # Hidden entries count: a real vault has .obsidian, so only a root with nothing in it at
    # all is refused. A drive or bind mount that did not come up is an existing empty
    # directory on Linux, and would otherwise look like a vault whose notes were all deleted.
    if previous and not os.listdir(vault):
        raise ObsidianVaultUnavailableError(
            "The Obsidian vault folder is empty, but earlier runs synced notes from it; nothing "
            "was synced and nothing was forgotten. A drive that did not mount looks exactly like "
            "this. Check the mount, or leave one file or folder in the vault to forget everything "
            "on purpose."
        )
    for folder in folders or []:
        # A folder that is gone with members under it was removed on purpose and forgets them.
        # One with no members and not on disk is most likely a typo and selects nothing.
        if not (vault / folder).exists() and not any(
            note_id.startswith(_note_id(f"{folder}/")) for note_id in previous
        ):
            logger.warning(
                "The selected Obsidian folder %r does not exist in the vault and has no synced "
                "notes, so it selects nothing; check the spelling, or remove it from folders if "
                "it was deleted on purpose.",
                folder,
            )
    unreadable = set()
    for rel_path in listed:
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a failed note
        try:
            text = (vault / rel_path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            logger.warning(
                "Obsidian note %r could not be read (%s); it keeps what memory already holds.",
                rel_path,
                type(error).__name__,
            )
            unreadable.add(_note_id(rel_path))
            stats["failed"] += 1
            stats["failed_read"] += 1
            continue
        yield _note_row(rel_path, text)
        # After the yield: a row that check_active stopped before it was published is not counted.
        stats["synced"] += 1

    present = {_note_id(rel_path) for rel_path in listed}
    for note_id in sorted(previous - present):
        yield {"id": note_id, "_deleted": True}
        stats["deleted"] += 1
    state.update(v=1, members=sorted((present - unreadable) | (previous & unreadable)))
    logger.info(
        "Obsidian: %d note(s) synced, %d forgotten, %d unreadable.",
        stats["synced"],
        stats["deleted"],
        stats["failed_read"],
    )


def _folder_list(folders: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """The selected folders, sorted and without repeats; ``None`` is the whole vault.

    Anything but a non-empty list or tuple of strings is refused. A bare string would be read
    as one folder per letter, and an empty list is neither "everything" nor "nothing" as far
    as a caller can tell. Folders are vault-relative with ``/``. An entry that starts with a
    dot (hidden folders are never read), contains a backslash (one name on POSIX) or has
    surrounding spaces would select nothing, and the next run would forget every note.
    """
    if folders is None:
        return None
    if not isinstance(folders, (list, tuple)) or not folders:
        raise ValueError('folders must be a non-empty list or tuple, like ["Projects/Work"].')
    selected = set()
    for entry in folders:
        path = PurePosixPath(entry) if isinstance(entry, str) else None
        if (
            path is None
            or path.is_absolute()
            or "\\" in entry
            or entry != entry.strip()
            or not path.parts
            or any(part.startswith(".") for part in path.parts)
        ):
            raise ValueError(
                f"folders entry {entry!r} must be a path inside the vault written with /, "
                "without surrounding spaces, whose folder names do not start with a dot."
            )
        selected.add(path.as_posix())
    return sorted(selected)


def obsidian_source(
    *,
    vault_path: str | os.PathLike[str],
    resource_name: str | None = None,
    folders: list[str] | tuple[str, ...] | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per note of a local Obsidian vault.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. A run walks the whole selection; a note whose text did not change
    costs no extraction. A normal return can still have skipped notes: read
    ``cognee_sync_stats["failed"]`` (``failed_read``) afterwards.

    Args:
        vault_path: The vault folder, ``~`` expanded and resolved. A ``ValueError`` if it is not
            an existing directory. A run raises if it has disappeared by then, or if its root has
            no entries at all although notes were synced from it before (a drive that did not
            mount).
        resource_name: Names the staging table and the sync state. The default is derived from
            the resolved vault path, so moving the vault folder changes it and ingests every
            note again; pass a name to pin it. Two vaults in one dataset need different names.
        folders: Vault-relative folders to read, including their subfolders. ``None`` is the
            whole vault. Notes outside a narrowed selection are forgotten on the next run.
        check_active: Host authorization checkpoint, called before each note is read and before
            each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Obsidian sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("vault_path must be an existing directory.")
    if resource_name is None:
        resource_name = f"obsidian_{hashlib.sha256(str(vault).encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    selection = _folder_list(folders)
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def obsidian_notes():
        stats.update(synced=0, deleted=0, failed=0, failed_read=0)
        rows = _iter_rows(vault, selection, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = obsidian_notes()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "obsidian")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
