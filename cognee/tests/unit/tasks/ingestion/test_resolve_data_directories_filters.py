"""Directory-expansion filtering in resolve_data_directories.

Expanding a directory must skip version-control internals and binary files no
loader supports, optionally skip .gitignore matches (respect_gitignore=True)
and user-provided gitignore-style patterns (exclude_patterns) — while never
filtering explicitly passed files, and never touching the supported binary
formats (PDF, images, ...) that registered loaders claim.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from cognee.tasks.ingestion import directory_file_filters
from cognee.tasks.ingestion.directory_file_filters import (
    filter_s3_keys,
    loader_supported_extensions,
)
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

BINARY = b"\x7fELF\x00\x00\x01binary blob"


@pytest.fixture
def repo(tmp_path):
    """A repo-shaped tree: sources, junk binaries, a supported binary,
    VCS internals, a virtualenv, and a .gitignore covering some of it."""
    (tmp_path / "main.py").write_text("print('hello')\n")
    (tmp_path / "README.md").write_text("# readme\n")
    (tmp_path / "run.log").write_text("log line\n")
    (tmp_path / "module.so").write_bytes(BINARY)
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4\x00fake pdf")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("app = 1\n")
    (tmp_path / "src" / "app.pyc").write_bytes(BINARY)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "site.py").write_text("venv file\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "index.js").write_text("js\n")
    (tmp_path / ".gitignore").write_text(".venv/\n*.log\n")
    return tmp_path


@pytest.fixture(autouse=True)
def supported_extensions():
    """Pin the loader-supported set so tests don't depend on which optional
    loader extras are installed in the environment."""
    with patch.object(
        directory_file_filters,
        "loader_supported_extensions",
        return_value={"pdf", "txt", "md"},
    ):
        yield


def names(resolved, root):
    return sorted(str(Path(item).relative_to(root)) for item in resolved)


@pytest.mark.asyncio
async def test_default_expansion_skips_vcs_and_unsupported_binaries(repo):
    resolved = await resolve_data_directories([str(repo)])

    kept = names(resolved, repo)
    assert "main.py" in kept
    assert "src/app.py" in kept
    assert "report.pdf" in kept, "binary with a loader-supported extension must survive"
    assert ".venv/lib/site.py" in kept, "gitignore is opt-in — .venv stays without the flag"
    assert "run.log" in kept
    assert not any(name.startswith(".git/") for name in kept), "VCS internals always skipped"
    assert "module.so" not in kept, "unsupported binary always skipped"
    assert "src/app.pyc" not in kept


@pytest.mark.asyncio
async def test_respect_gitignore_skips_ignored_files(repo):
    resolved = await resolve_data_directories([str(repo)], respect_gitignore=True)

    kept = names(resolved, repo)
    assert "main.py" in kept
    assert not any(name.startswith(".venv/") for name in kept)
    assert "run.log" not in kept
    assert "node_modules/index.js" in kept, "not in .gitignore, stays without exclude_patterns"


@pytest.mark.asyncio
async def test_exclude_patterns_skip_matches(repo):
    resolved = await resolve_data_directories(
        [str(repo)], exclude_patterns=["node_modules/", "*.md"]
    )

    kept = names(resolved, repo)
    assert "main.py" in kept
    assert "README.md" not in kept
    assert not any(name.startswith("node_modules/") for name in kept)
    assert ".venv/lib/site.py" in kept, "exclude_patterns alone leave .gitignore alone"


@pytest.mark.asyncio
async def test_flags_combine(repo):
    resolved = await resolve_data_directories(
        [str(repo)], respect_gitignore=True, exclude_patterns=["node_modules/"]
    )

    kept = names(resolved, repo)
    # .gitignore itself is a tracked text file — git keeps it, so do we.
    assert kept == [".gitignore", "README.md", "main.py", "report.pdf", "src/app.py"]


@pytest.mark.asyncio
async def test_non_recursive_expansion_is_filtered_too(repo):
    resolved = await resolve_data_directories([str(repo)], include_subdirectories=False)

    kept = names(resolved, repo)
    assert "main.py" in kept
    assert "module.so" not in kept
    assert "src/app.py" not in kept, "non-recursive expansion keeps top-level files only"


@pytest.mark.asyncio
async def test_explicitly_passed_files_are_never_filtered(repo):
    binary = str(repo / "module.so")
    log = str(repo / "run.log")

    resolved = await resolve_data_directories(
        [binary, log], respect_gitignore=True, exclude_patterns=["*.log"]
    )

    assert resolved == [binary, log]


def test_filter_s3_keys_applies_exclude_patterns_only():
    keys = [
        "s3://bucket/project/main.py",
        "s3://bucket/project/.venv/lib/site.py",
        "s3://bucket/project/run.log",
    ]

    kept = filter_s3_keys("s3://bucket/project", keys, [".venv/", "*.log"])

    assert kept == ["s3://bucket/project/main.py"]
    assert filter_s3_keys("s3://bucket/project", keys, None) == keys


def test_loader_supported_extensions_reads_real_registry():
    # The autouse fixture patches the name inside directory_file_filters; this
    # module-level import still points at the real function.
    extensions = loader_supported_extensions()
    assert "txt" in extensions
