"""Preview uses the checkout selector and refuses incomplete provider trees."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.integrations.github import preview
from cognee.modules.integrations.github.sync_settings import GithubRepositorySettings
from cognee.tasks.code_graph.repo_selection import preview_tree


def tree(*paths):
    return {
        "sha": "b" * 40,
        "truncated": False,
        "tree": [
            {"mode": "100644", "type": "blob", "sha": "a" * 40, "path": path} for path in paths
        ],
    }


def test_preview_and_checkout_share_filters():
    result = preview_tree(
        preview.tree_listing(tree("src/main.py", "src/tests/test_main.py", "README.md")),
        include_documents=False,
        include_paths=["src/**"],
        exclude_paths=["**/tests/**"],
    )
    assert result["selected"] == ["src/main.py"]
    assert sorted(path for paths in result["skipped"].values() for path in paths) == [
        "README.md",
        "src/tests/test_main.py",
    ]


@pytest.mark.parametrize(
    "payload", [{"tree": []}, {"tree": [], "truncated": True}, {"tree": None, "truncated": False}]
)
def test_incomplete_tree_is_not_a_successful_preview(payload):
    with pytest.raises(ValueError, match="incomplete"):
        preview.tree_listing(payload)


def test_tree_rejects_delimiter_injection():
    with pytest.raises(ValueError, match="invalid"):
        preview.tree_listing(tree("ok.py\0evil.py"))


@pytest.mark.asyncio
async def test_repository_must_belong_to_installation(monkeypatch):
    credential = SimpleNamespace(provider="github", provider_account_id="42")
    monkeypatch.setattr(preview, "require_active_credential", AsyncMock(return_value=credential))
    monkeypatch.setattr(
        preview, "mint_installation_token", AsyncMock(return_value=("secret", None))
    )
    monkeypatch.setattr(
        preview,
        "list_installation_repository_records",
        AsyncMock(return_value=[{"id": 1, "full_name": "org/allowed", "default_branch": "main"}]),
    )
    with pytest.raises(ValueError, match="not available"):
        await preview.preview_repository(credential, "2", GithubRepositorySettings())


@pytest.mark.asyncio
async def test_preview_reads_branch_and_does_not_expose_token(monkeypatch):
    credential = SimpleNamespace(provider="github", provider_account_id="42")
    active = AsyncMock(return_value=credential)
    monkeypatch.setattr(preview, "require_active_credential", active)
    monkeypatch.setattr(
        preview, "mint_installation_token", AsyncMock(return_value=("secret", None))
    )
    monkeypatch.setattr(
        preview,
        "list_installation_repository_records",
        AsyncMock(return_value=[{"id": 1, "full_name": "org/repo", "default_branch": "main"}]),
    )

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return tree("src/main.py")

    class Session(Response):
        def get(self, url, **kwargs):
            assert url.endswith("/org/repo/git/trees/feature%2Fsync")
            assert kwargs["params"] == {"recursive": "1"}
            return Response()

    monkeypatch.setattr(preview.aiohttp, "ClientSession", lambda **kwargs: Session())
    result = await preview.preview_repository(
        credential, "1", GithubRepositorySettings(branch="feature/sync", include_paths=["src/**"])
    )
    assert result["selected"] == ["src/main.py"]
    assert result["tree_sha"] == "b" * 40
    assert "secret" not in str(result)
    assert active.await_count == 2
