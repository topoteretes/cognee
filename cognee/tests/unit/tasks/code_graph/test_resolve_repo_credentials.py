"""Unit tests for credentialed-URL handling in resolve_repo.

Connectors (the GitHub App integration) pass short-lived tokens in the URL
userinfo. The invariants that matter: the token is used for the clone/fetch
itself and nowhere else — not the clone slug (which must stay stable across
token rotations), not the stored git remote, not error messages.
"""

import importlib
from unittest.mock import AsyncMock

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

_TOKEN_URL = "https://x-access-token:tok-SECRET@github.com/org/repo.git"
_CLEAN_URL = "https://github.com/org/repo.git"


@pytest.fixture(autouse=True)
def _origin_matches(monkeypatch):
    """The clones here are fake ``.git`` directories; their origin is the clean URL."""

    async def fake_origin_url(clone):
        return _CLEAN_URL

    monkeypatch.setattr(resolve_module, "_origin_url", fake_origin_url)
    # git is mocked, so resolving the host would only tie these tests to the network.
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())


def test_redact_repo_spec_strips_userinfo():
    assert resolve_module.redact_repo_spec(_TOKEN_URL) == _CLEAN_URL
    assert resolve_module.redact_repo_spec("https://user@host/path") == "https://host/path"


def test_redact_repo_spec_leaves_credential_free_specs_alone():
    assert resolve_module.redact_repo_spec(_CLEAN_URL) == _CLEAN_URL
    assert resolve_module.redact_repo_spec("/local/path/repo") == "/local/path/repo"
    # scp-style git URLs have no "://", so the "@" is not URL userinfo.
    assert (
        resolve_module.redact_repo_spec("git@github.com:org/repo.git")
        == "git@github.com:org/repo.git"
    )


@pytest.mark.asyncio
async def test_clone_slug_and_remote_never_carry_the_token(monkeypatch, tmp_path):
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path)

    # Slug comes from the clean URL — same directory a credential-free call
    # (or a later sync with a rotated token) would use.
    assert resolved == resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)

    clone_args, _ = git_calls[0]
    # core.symlinks=false is a security requirement, not incidental: without it a
    # symlink committed in the remote is materialized and later written through.
    assert clone_args[:5] == ["clone", "-c", "core.symlinks=false", "--depth", "1"]
    assert _TOKEN_URL in clone_args

    # The persisted remote is rewritten to the credential-free URL.
    assert git_calls[1] == (["remote", "set-url", "origin", _CLEAN_URL], resolved)


@pytest.mark.asyncio
async def test_existing_clone_pulls_from_the_credentialed_url(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path)

    assert resolved == clone_dir
    # Explicit URL argument: the stored remote is credential-free on purpose.
    assert git_calls == [(["pull", "--ff-only", _TOKEN_URL], clone_dir)]


@pytest.mark.asyncio
async def test_credential_free_urls_keep_the_bare_pull(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    await resolve_module.resolve_repo_source(_CLEAN_URL, clones_dir=tmp_path)

    assert git_calls == [(["pull", "--ff-only"], clone_dir)]


@pytest.mark.asyncio
async def test_out_of_band_credentials_ride_the_environment(monkeypatch, tmp_path):
    """The preferred path (connectors): token via ``credentials=``, not the URL.

    The URL stays clean everywhere — argv, slug, remote — and the token
    reaches git only as environment-level config, so no URL-derived string
    can leak it.
    """
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd, env))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        _CLEAN_URL, clones_dir=tmp_path, credentials="tok-SECRET"
    )

    assert resolved == resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    ((clone_args, _cwd, clone_env),) = git_calls
    # argv carries only the clean URL; the token is nowhere in it.
    assert clone_args == [
        "clone",
        "-c",
        "core.symlinks=false",
        "--depth",
        "1",
        _CLEAN_URL,
        str(resolved),
    ]
    assert "tok-SECRET" not in " ".join(clone_args)
    # ...and rides GIT_CONFIG_* as a basic-auth header instead.
    assert clone_env["GIT_CONFIG_KEY_0"] == "http.extraHeader"
    assert clone_env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


@pytest.mark.asyncio
async def test_out_of_band_credentials_apply_to_pulls_too(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd, env))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    await resolve_module.resolve_repo_source(
        _CLEAN_URL, clones_dir=tmp_path, credentials="tok-SECRET"
    )

    ((pull_args, pull_cwd, pull_env),) = git_calls
    # Bare pull against the (clean) stored remote; auth via the environment.
    assert pull_args == ["pull", "--ff-only"]
    assert pull_cwd == clone_dir
    assert pull_env["GIT_CONFIG_KEY_0"] == "http.extraHeader"


@pytest.mark.asyncio
async def test_failed_clone_error_is_scrubbed(monkeypatch, tmp_path):
    async def fake_run_git(args, cwd=None, env=None):
        # git echoes the full URL, token included, in its error output.
        return 128, f"fatal: unable to access '{_TOKEN_URL}': 403"

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    with pytest.raises(resolve_module.CodeRepositoryError) as exc_info:
        await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path)

    message = str(exc_info.value)
    assert "tok-SECRET" not in message
    assert _CLEAN_URL in message


def _marker(clone_dir):
    return clone_dir / ".git" / resolve_module._CREDENTIALED_MARKER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec, credentials", [(_TOKEN_URL, None), (_CLEAN_URL, "tok-SECRET")], ids=["url", "argument"]
)
async def test_a_clone_made_with_credentials_is_marked(monkeypatch, tmp_path, spec, credentials):
    async def fake_run_git(args, cwd=None, env=None):
        if args[0] == "clone":
            (resolve_module.Path(args[-1]) / ".git").mkdir(parents=True)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        spec, clones_dir=tmp_path, credentials=credentials
    )

    assert _marker(resolved).is_file()


@pytest.mark.asyncio
async def test_a_public_clone_is_not_marked(monkeypatch, tmp_path):
    async def fake_run_git(args, cwd=None, env=None):
        if args[0] == "clone":
            (resolve_module.Path(args[-1]) / ".git").mkdir(parents=True)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(_CLEAN_URL, clones_dir=tmp_path)

    assert not _marker(resolved).exists()


async def _failing_refresh(args, cwd=None, env=None):
    return 128, "fatal: could not read Username for 'https://github.com': terminal prompts disabled"


@pytest.mark.asyncio
async def test_a_credentialed_clone_is_refused_when_a_caller_without_credentials_cannot_refresh(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(resolve_module, "_run_git", _failing_refresh)

    for branch in (None, "dev"):
        target = resolve_module.clone_dir(_CLEAN_URL, branch, tmp_path)
        (target / ".git").mkdir(parents=True, exist_ok=True)
        _marker(target).touch()
        with pytest.raises(resolve_module.CodeRepositoryError, match="made with credentials"):
            await resolve_module.resolve_repo_source(_CLEAN_URL, clones_dir=tmp_path, branch=branch)


@pytest.mark.asyncio
async def test_a_credentialed_caller_still_gets_the_stale_clone_and_keeps_the_marker(
    monkeypatch, tmp_path
):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _marker(clone_dir).touch()
    monkeypatch.setattr(resolve_module, "_run_git", _failing_refresh)

    resolved = await resolve_module.resolve_repo_source(
        _CLEAN_URL, clones_dir=tmp_path, credentials="tok-SECRET"
    )

    assert resolved == clone_dir
    assert _marker(clone_dir).is_file()


@pytest.mark.asyncio
async def test_a_public_clone_is_still_reused_stale(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    monkeypatch.setattr(resolve_module, "_run_git", _failing_refresh)

    resolved = await resolve_module.resolve_repo_source(_CLEAN_URL, clones_dir=tmp_path)

    assert resolved == clone_dir


@pytest.mark.asyncio
async def test_a_credentialed_refresh_marks_the_clone(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)

    async def fake_run_git(args, cwd=None, env=None):
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path)

    assert _marker(clone_dir).is_file()
