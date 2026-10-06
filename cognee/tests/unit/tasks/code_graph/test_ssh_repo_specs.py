"""An ssh git remote is refused, never stored as a text document (SDK-793).

``code_repo_clone_url`` does not accept ``git@``/``ssh://`` specs, so before
this guard they fell through every repository branch and were saved as a
one-line text document naming the URL — a silent wrong result. ``is_remote_repo``
cannot be used for the check: it matches every http(s) URL, most of which really
are web pages.
"""

import pytest

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.code_graph.resolve_repo import (
    code_repo_clone_url,
    is_remote_repo,
    is_ssh_repo_spec,
)

SSH_SPECS = ["git@github.com:org/repo.git", "ssh://git@github.com/org/repo"]
NOT_SSH = [
    "https://github.com/org/repo",
    "https://example.com/article",
    "http://news.site/post/1",
    "/srv/repos/foo",
    "just some text",
]


@pytest.mark.parametrize("spec", SSH_SPECS)
def test_ssh_specs_are_detected(spec):
    assert is_ssh_repo_spec(spec)


@pytest.mark.parametrize("spec", NOT_SSH)
def test_web_pages_and_paths_are_not_ssh_specs(spec):
    assert not is_ssh_repo_spec(spec)


def test_the_guard_is_narrower_than_is_remote_repo():
    # The reason the check exists as its own predicate: is_remote_repo is true
    # for ordinary web pages, so using it here would refuse normal ingestion.
    assert is_remote_repo("https://example.com/article")
    assert not is_ssh_repo_spec("https://example.com/article")
    assert code_repo_clone_url("https://example.com/article") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", SSH_SPECS)
async def test_resolve_data_directories_refuses_ssh_specs(spec):
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    with pytest.raises(IngestionError, match="ssh git remotes"):
        await resolve_data_directories([spec])


@pytest.mark.asyncio
async def test_a_web_page_still_passes_through_untouched():
    from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories

    assert await resolve_data_directories(["https://example.com/article"]) == [
        "https://example.com/article"
    ]
