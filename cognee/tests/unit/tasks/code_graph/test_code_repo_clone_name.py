"""A cloned repository keeps the code-graph name it had before clone directories were hashed.

enola names a repository after its directory and every code node id derives from that
name, so a clone directory that gained a hash suffix would give an already stored
repository a second subgraph that nothing sweeps. The name used is the readable slug the
directory had before; a local directory keeps its own name.
"""

import hashlib
import importlib
import json
import re
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.code_graph.code_repo import extract_code_repo_graph
from cognee.tasks.code_graph.extract_code_graph import (
    build_code_graph_edges,
    map_facts_to_data_points,
)
from cognee.tasks.code_graph.resolve_repo import clone_dir, clone_repo_name

_URL = "https://github.com/org/repo"


def _legacy_clone_slug(url, branch=None):
    """What the clone directory was called before it carried a hash."""
    tail = url.split("://")[-1].replace(":", "/").rstrip("/").removesuffix(".git")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.")
    if branch is None:
        return slug
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", branch)[:40]
    return f"{slug}@{readable}-{hashlib.sha256(branch.encode()).hexdigest()[:8]}"


@pytest.mark.parametrize("branch", [None, "dev", "feature/ü"])
def test_a_clone_is_named_as_its_directory_was_before_the_hash(tmp_path, branch):
    hashed = clone_dir(_URL, branch, tmp_path)
    assert hashed.name != _legacy_clone_slug(_URL, branch)

    assert clone_repo_name(_URL, hashed) == _legacy_clone_slug(_URL, branch)


def test_a_directory_that_is_not_the_urls_clone_keeps_its_name(tmp_path):
    assert clone_repo_name(_URL, tmp_path / "my-checkout") == "my-checkout"
    other = clone_dir("https://github.com/org/other", clones_dir=tmp_path)
    assert clone_repo_name(_URL, other) == other.name


def _facts(repo):
    """An enola snapshot of one repository, ``repo`` being the name enola gave it."""
    owner = {"repo": repo} if repo is not None else {}
    return [
        {"kind": "module", "name": "app", "file": "app/main.py", **owner, "relations": []},
        {
            "kind": "symbol",
            "name": "main",
            "file": "app/main.py",
            **owner,
            "props": {"symbol_kind": "function"},
            "relations": [{"type": "calls", "target": "helper"}],
        },
        {
            "kind": "symbol",
            "name": "helper",
            "file": "app/util.py",
            **owner,
            "props": {"symbol_kind": "function"},
            "relations": [],
        },
        {
            "kind": "route",
            "name": "GET /items",
            "file": "app/client.py",
            **owner,
            "props": {
                "role": "client",
                "caller": "main",
                "matched_routes": [{"repo": repo, "name": "GET /items"}],
            },
            "relations": [],
        },
    ]


@pytest.mark.parametrize("enola_names_the_repo", [True, False])
def test_node_and_edge_ids_of_a_hashed_clone_match_the_legacy_name(tmp_path, enola_names_the_repo):
    hashed = clone_dir(_URL, clones_dir=tmp_path)
    legacy = tmp_path / _legacy_clone_slug(_URL)
    repo_name = clone_repo_name(_URL, hashed)

    new_facts = _facts(hashed.name if enola_names_the_repo else None)
    old_facts = _facts(legacy.name if enola_names_the_repo else None)

    new_points = map_facts_to_data_points(new_facts, repo_path=hashed, repo_name=repo_name)
    old_points = map_facts_to_data_points(old_facts, repo_path=legacy)
    assert [point.id for point in new_points] == [point.id for point in old_points]
    assert {point.name for point in new_points if type(point).__name__ == "CodeRepository"} == {
        legacy.name
    }

    new_edges, _ = build_code_graph_edges(new_facts, repo_path=hashed, repo_name=repo_name)
    old_edges, _ = build_code_graph_edges(old_facts, repo_path=legacy)
    assert new_edges and [edge[:3] for edge in new_edges] == [edge[:3] for edge in old_edges]

    # Without a name, the directory names the repository, as for a local checkout.
    unnamed = map_facts_to_data_points(new_facts, repo_path=hashed)
    assert [point.id for point in unnamed] != [point.id for point in old_points]


@pytest.mark.asyncio
@pytest.mark.parametrize("repo_url", [_URL, None], ids=["clone", "local"])
@pytest.mark.parametrize("index_vectors", [False, True])
async def test_the_code_repo_route_passes_the_name_to_both_graph_tasks(
    monkeypatch, tmp_path, repo_url, index_vectors
):
    """The route names the repository for both graph tasks, and stores graph-only
    unless index_vectors asks it to embed."""
    open_data_file_module = importlib.import_module(
        "cognee.infrastructure.files.utils.open_data_file"
    )
    extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
    repo = clone_dir(repo_url, clones_dir=tmp_path) if repo_url else tmp_path / "local-checkout"
    repo.mkdir(parents=True)

    @asynccontextmanager
    async def fake_open(*_args, **_kwargs):
        yield StringIO(json.dumps({"repo_path": str(repo)}))

    monkeypatch.setattr(open_data_file_module, "open_data_file", fake_open)
    extract = AsyncMock(return_value=[])
    edges = AsyncMock()
    monkeypatch.setattr(extract_module, "extract_code_graph", extract)
    add_points = AsyncMock(return_value=[])
    monkeypatch.setattr(extract_module, "add_code_graph_data_points", add_points)
    monkeypatch.setattr(extract_module, "add_code_graph_edges", edges)

    metadata = {"source": "code_repo", **({"repo_url": repo_url} if repo_url else {})}
    data_item = SimpleNamespace(
        id="d1", raw_data_location="unused", system_metadata=metadata, external_metadata=None
    )
    await extract_code_repo_graph([data_item], index_vectors=index_vectors)

    expected = _legacy_clone_slug(repo_url) if repo_url else None
    assert extract.await_args.kwargs["repo_name"] == expected
    assert edges.await_args.kwargs["repo_name"] == expected
    assert Path(extract.await_args.kwargs["repo_path"]) == repo
    assert add_points.await_args.kwargs["graph_only"] is (not index_vectors)
