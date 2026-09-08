"""
Grouping for presort: propose which files belong together as datasets.

Deterministic layers, in priority order:
1. code projects — a subdirectory carrying a project marker (pyproject.toml,
   .git, ...) becomes one group, kind ``code_project``;
2. folder structure — remaining subdirectories become dot-namespaced groups
   (the layout mapping of ``discover_directory_datasets``), kind ``folder``;
3. loose files directly in the root — grouped by extension family, kind
   ``extension_family``.

Opt-in LLM/embedding layer (``use_llm=True``): loose text documents are
embedded (name + content class) and clustered with a union-find over cosine
similarity — the same shape as ``consolidate_entities`` — producing
``semantic`` groups that replace their extension-family grouping.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.code_repo import detect_code_project

from .models import FileRecord, ProposedGroup

logger = get_logger("presort")

_SEMANTIC_SIMILARITY_THRESHOLD = 0.6
_MIN_SEMANTIC_GROUP_SIZE = 2


def sanitize_dataset_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._]+", "_", name.strip()).strip("_.").lower()
    return sanitized or "unsorted"


def _folder_group_name(root: Path, directory: Path) -> str:
    relative = directory.relative_to(root)
    return ".".join(relative.parts)


async def group_files(
    root: Path,
    files: List[FileRecord],
    *,
    use_llm: bool = False,
    dataset_prefix: str = "",
) -> List[ProposedGroup]:
    remaining: Dict[str, FileRecord] = {record.path: record for record in files}
    groups: List[ProposedGroup] = []

    def make_group(name: str, kind, reason: str, paths: List[str]) -> None:
        member_paths = [path for path in paths if path in remaining]
        if not member_paths:
            return
        for path in member_paths:
            remaining.pop(path)
        groups.append(
            ProposedGroup(
                name=name,
                dataset_name=sanitize_dataset_name(f"{dataset_prefix}{name}"),
                kind=kind,
                reason=reason,
                file_paths=sorted(member_paths),
            )
        )

    # 1. Code projects: any directory under root carrying a project marker.
    project_dirs: List[Path] = []
    try:
        candidate_dirs = [root] + [path for path in sorted(root.rglob("*")) if path.is_dir()]
    except OSError:
        candidate_dirs = [root]
    for directory in candidate_dirs:
        if any(directory.is_relative_to(project) for project in project_dirs):
            continue  # nested inside an already-detected project
        try:
            if detect_code_project(directory):
                project_dirs.append(directory)
        except OSError:
            continue
    for directory in project_dirs:
        name = _folder_group_name(root, directory) if directory != root else root.name
        make_group(
            name or root.name,
            "code_project",
            f"code project detected at {directory}",
            [path for path in remaining if Path(path).is_relative_to(directory)],
        )

    # 2. Folder structure: first-level subdirectory of each remaining file.
    by_folder: Dict[str, List[str]] = defaultdict(list)
    loose: List[str] = []
    for path in list(remaining):
        relative = Path(path).relative_to(root)
        if len(relative.parts) > 1:
            by_folder[relative.parts[0]].append(path)
        else:
            loose.append(path)
    for folder_name in sorted(by_folder):
        make_group(
            folder_name,
            "folder",
            f"files under {root / folder_name}",
            by_folder[folder_name],
        )

    # 3. Optional semantic grouping of loose text documents.
    if use_llm:
        semantic_groups = await _semantic_groups(
            [remaining[path] for path in loose if path in remaining]
        )
        for index, member_records in enumerate(semantic_groups, start=1):
            label = _semantic_label(member_records) or f"topic_{index}"
            make_group(
                label,
                "semantic",
                "text documents with similar names/content classes",
                [record.path for record in member_records],
            )

    # 4. Remaining loose files: extension-family fallback.
    by_family: Dict[str, List[str]] = defaultdict(list)
    for path, record in remaining.items():
        by_family[record.family].append(path)
    for family in sorted(by_family):
        make_group(family, "extension_family", f"loose {family} in {root}", by_family[family])

    return groups


def _semantic_label(records: List[FileRecord]) -> str:
    classes = [record.content_class for record in records if record.content_class]
    if classes:
        most_common = max(set(classes), key=classes.count)
        return sanitize_dataset_name(most_common)[:40]
    return ""


async def _semantic_groups(records: List[FileRecord]) -> List[List[FileRecord]]:
    """Cluster text documents by embedding cosine similarity (union-find)."""
    candidates = [record for record in records if record.is_text and not record.is_code]
    if len(candidates) < _MIN_SEMANTIC_GROUP_SIZE:
        return []

    try:
        import numpy as np

        from cognee.infrastructure.databases.vector import get_vector_engine_async

        vector_engine = await get_vector_engine_async()
        texts = [
            f"{record.name} {record.content_class or ''} {record.family}" for record in candidates
        ]
        vectors = np.asarray(await vector_engine.embed_data(texts), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        parent = list(range(len(candidates)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        similarities = normalized @ normalized.T
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if similarities[i, j] >= _SEMANTIC_SIMILARITY_THRESHOLD:
                    parent[find(i)] = find(j)

        clusters: Dict[int, List[FileRecord]] = defaultdict(list)
        for index, record in enumerate(candidates):
            clusters[find(index)].append(record)
        return [
            members for members in clusters.values() if len(members) >= _MIN_SEMANTIC_GROUP_SIZE
        ]
    except Exception as error:  # embedding failures must not abort presort
        logger.warning(f"Presort semantic grouping skipped: {error}")
        return []
