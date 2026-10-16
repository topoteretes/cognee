"""An ssh git remote is refused, never stored as a text document (SDK-793).

``code_repo_clone_url`` does not accept ``git@``/``ssh://`` specs, so before
this guard they fell through every repository branch and were saved as a
one-line text document naming the URL — a silent wrong result. ``is_remote_repo``
cannot be used for the check: it matches every http(s) URL, most of which really
are web pages.
"""

import pytest

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.code_graph.resolve_repo import is_ssh_repo_spec

SSH_SPECS = [
    "git@github.com:org/repo.git",
    "ssh://git@github.com/org/repo",
    "git@gitlab.company.internal:group/sub/repo",
    "  git@github.com:org/repo.git\n",  # surrounding whitespace is stripped
    "ssh://git@host.example:2222/srv/repo.git",
]
NOT_SSH = [
    "https://github.com/org/repo",
    "https://example.com/article",
    "http://news.site/post/1",
    "/srv/repos/foo",
    "just some text",
    # Prose that merely starts like a remote is a note, not a repository.
    "git@company.com is the bot account",
    "git@github.com:org/repo is where the code lives",
    "ssh:// is disabled on the build hosts",
    "git@company.com",
    "git@:repo",
]


@pytest.mark.parametrize("spec", SSH_SPECS)
def test_ssh_specs_are_detected(spec):
    assert is_ssh_repo_spec(spec)


@pytest.mark.parametrize("spec", NOT_SSH)
def test_web_pages_and_paths_are_not_ssh_specs(spec):
    assert not is_ssh_repo_spec(spec)


@pytest.mark.asyncio
async def test_resolve_data_directories_refuses_ssh_specs():
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    with pytest.raises(IngestionError, match="ssh git remotes"):
        await resolve_data_directories(["git@github.com:org/repo.git"])


@pytest.mark.asyncio
async def test_prose_starting_like_a_remote_is_ingested_as_text():
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    note = "git@company.com is the bot account"
    assert await resolve_data_directories([note]) == [note]
