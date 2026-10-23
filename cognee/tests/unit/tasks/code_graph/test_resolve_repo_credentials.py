"""Unit tests for credentialed-URL handling in resolve_repo.

Connectors (the GitHub App integration) pass short-lived tokens in the URL
userinfo. The invariants that matter: the token is used for the clone/fetch
itself and nowhere else — not the clone slug (which must stay stable across
token rotations), not the stored git remote, not error messages.
"""

import importlib
import os
import shutil
import subprocess
from unittest.mock import AsyncMock

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


@pytest.fixture(autouse=True)
def _mock_partial_checkout(monkeypatch):
    """Replace the partial checkout for every test in this module.

    These tests exercise resolver and auth orchestration with fake .git directories.
    Partial checkout and actual object transfers are covered by the real-Git suite.
    """

    async def checkout(repo, ref, include_documents, env, scrub):
        if ref == "FETCH_HEAD":
            code, stderr = await resolve_module._run_git(
                ["reset", "--hard", ref], cwd=repo, env=env
            )
            if code:
                raise resolve_module.CodeRepositoryError(
                    message="Could not checkout repository: " + scrub(stderr)
                )

    monkeypatch.setattr(resolve_module, "_checkout_snapshot", checkout)


ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

_TOKEN_URL = "https://x-access-token:tok-SECRET@github.com/org/repo.git"
_CLEAN_URL = "https://github.com/org/repo.git"


@pytest.fixture(autouse=True)
def _origin_matches(monkeypatch):
    """The clones here are fake ``.git`` directories; their origin is the clean URL."""

    async def fake_origin_url(clone):
        return _CLEAN_URL

    monkeypatch.setattr(resolve_module, "_origin_url", fake_origin_url)
    monkeypatch.setattr(resolve_module, "_has_commit", AsyncMock(return_value=True))
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

    # The persisted remote is rewritten to the credential-free URL, in the clone's
    # temporary directory: the clone is renamed into place only once it holds no token.
    set_url_args, set_url_cwd = git_calls[1]
    assert set_url_args == ["remote", "set-url", "origin", _CLEAN_URL]
    assert set_url_cwd == resolve_module.Path(clone_args[-1])
    assert set_url_cwd.parent == tmp_path and set_url_cwd != resolved


@pytest.mark.asyncio
async def test_existing_clone_fetches_from_the_credentialed_url(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []
    environments = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        environments.append(env)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(_TOKEN_URL, clones_dir=tmp_path)

    assert resolved == clone_dir
    # Reset lazily fetches the selected blobs from origin, which is stored without
    # credentials. The token URL is supplied only through this process's config.
    assert ("remote.origin.url", _TOKEN_URL) in resolve_module._git_config_entries(environments[-1])
    # Explicit URL argument: the stored remote is credential-free on purpose.
    assert git_calls == [
        (
            [
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-recurse-submodules",
                _TOKEN_URL,
                "HEAD",
            ],
            clone_dir,
        ),
        (["reset", "--hard", "FETCH_HEAD"], clone_dir),
    ]


@pytest.mark.asyncio
async def test_credential_free_urls_fetch_from_origin(monkeypatch, tmp_path):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    git_calls = []

    async def fake_run_git(args, cwd=None, env=None):
        git_calls.append((args, cwd))
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    await resolve_module.resolve_repo_source(_CLEAN_URL, clones_dir=tmp_path)

    assert git_calls == [
        (
            [
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-recurse-submodules",
                "origin",
                "HEAD",
            ],
            clone_dir,
        ),
        (["reset", "--hard", "FETCH_HEAD"], clone_dir),
    ]


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
        "--filter=blob:none",
        "--no-checkout",
        "--no-recurse-submodules",
        _CLEAN_URL,
        clone_args[-1],
    ]
    # Cloned beside its target, then renamed into place.
    assert resolve_module.Path(clone_args[-1]).parent == tmp_path
    assert "tok-SECRET" not in " ".join(clone_args)
    # ...and rides GIT_CONFIG_* as a basic-auth header for github.com only instead.
    assert clone_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
    assert clone_env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


@pytest.mark.asyncio
async def test_out_of_band_credentials_apply_to_fetches_too(monkeypatch, tmp_path):
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

    (fetch_args, fetch_cwd, fetch_env), (reset_args, reset_cwd, reset_env) = git_calls
    # Fetch against the clean stored remote; auth via the environment.
    assert fetch_args == [
        "fetch",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-recurse-submodules",
        "origin",
        "HEAD",
    ]
    assert fetch_cwd == reset_cwd == clone_dir
    assert fetch_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
    assert resolve_module._git_config_entries(reset_env) == resolve_module._git_config_entries(
        fetch_env
    )
    assert reset_args == ["reset", "--hard", "FETCH_HEAD"]


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
    "spec, credentials, existing, marked",
    [
        (_TOKEN_URL, None, False, True),
        (_CLEAN_URL, "tok-SECRET", False, True),
        (_CLEAN_URL, None, False, False),
        (_TOKEN_URL, None, True, True),
    ],
    ids=["url", "argument", "public", "credentialed-refresh"],
)
async def test_a_clone_made_with_credentials_is_marked(
    monkeypatch, tmp_path, spec, credentials, existing, marked
):
    if existing:
        (resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path) / ".git").mkdir(parents=True)

    async def fake_run_git(args, cwd=None, env=None):
        if args[0] == "clone":
            (resolve_module.Path(args[-1]) / ".git").mkdir(parents=True)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        spec, clones_dir=tmp_path, credentials=credentials
    )

    assert _marker(resolved).is_file() is marked


_BOGUS_URL = "https://x-access-token:tok-BOGUS@github.com/org/repo.git"


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", [None, "dev"])
@pytest.mark.parametrize(
    "spec, credentials, secret",
    [
        (_CLEAN_URL, None, None),
        (_CLEAN_URL, "tok-WRONG", "tok-WRONG"),
        (_BOGUS_URL, None, "tok-BOGUS"),
    ],
    ids=["no-credentials", "wrong-credentials", "bogus-url-credentials"],
)
async def test_a_credentialed_clone_is_refused_to_a_caller_whose_credentials_fail_to_refresh(
    monkeypatch, tmp_path, branch, spec, credentials, secret
):
    """Passing credentials proves nothing until the remote accepts them: none, wrong
    or bogus ones must not unlock a private clone that someone else's credentials made."""
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, branch, tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _marker(clone_dir).touch()

    async def failing_refresh_echoing_the_url(args, cwd=None, env=None):
        # git echoes the URL it was given, credentials included, in its error output.
        return 128, f"fatal: Authentication failed for '{spec}'"

    monkeypatch.setattr(resolve_module, "_run_git", failing_refresh_echoing_the_url)

    with pytest.raises(resolve_module.CodeRepositoryError, match="made with credentials") as error:
        await resolve_module.resolve_repo_source(
            spec, clones_dir=tmp_path, credentials=credentials, branch=branch
        )

    if secret:
        assert secret not in str(error.value)
    assert _CLEAN_URL in str(error.value)
    # Refused, not discarded: a later caller with valid credentials refreshes it.
    assert _marker(clone_dir).is_file()


@pytest.mark.asyncio
async def test_a_credentialed_clone_is_reused_after_a_refresh_with_the_callers_credentials(
    monkeypatch, tmp_path
):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    _marker(clone_dir).touch()
    seen_env = []

    async def fake_run_git(args, cwd=None, env=None):
        seen_env.append(env)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        _CLEAN_URL, clones_dir=tmp_path, credentials="tok-SECRET"
    )

    assert resolved == clone_dir
    assert seen_env[0]["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


def test_the_credential_header_is_scoped_to_the_remote_host():
    env = resolve_module._credential_env("tok-SECRET", "https://GitHub.com:8443/org/repo.git")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com:8443/.extraHeader"
    # An ssh remote never sends http headers, so it gets none.
    assert resolve_module._credential_env("tok-SECRET", "git@github.com:org/repo.git") == {}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_sends_the_scoped_header_to_the_remote_host_only(tmp_path):
    empty_config = tmp_path / "gitconfig"
    empty_config.write_text("")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        **resolve_module._credential_env("tok-SECRET", _CLEAN_URL),
    }

    def header_for(url):
        return subprocess.run(
            ["git", "config", "--get-urlmatch", "http.extraHeader", url],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    assert header_for("https://github.com/org/repo.git").startswith("Authorization: Basic ")
    assert header_for("https://github.com/other/repo").startswith("Authorization: Basic ")
    assert header_for("https://evil.example/org/repo.git") == ""
    assert header_for("http://github.com/org/repo.git") == ""


def test_git_runs_without_stored_credentials_or_askpass(monkeypatch):
    """Every git call drops credential helpers and askpass programs, so a refresh
    authenticates only with what the caller passed. Config entries from the process,
    from the caller's env and the hardening are renumbered into one list."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "someone")
    monkeypatch.setenv("GIT_ASKPASS", "/usr/bin/ksshaskpass")
    monkeypatch.setenv("SSH_ASKPASS", "/usr/bin/ksshaskpass")

    env = resolve_module._git_env(resolve_module._credential_env("tok-SECRET", _CLEAN_URL))

    assert resolve_module._git_config_entries(env) == [
        ("user.name", "someone"),
        ("http.https://github.com/.extraHeader", env["GIT_CONFIG_VALUE_1"]),
        ("credential.helper", ""),
        ("core.askPass", ""),
        ("filter.lfs.process", ""),
        ("filter.lfs.smudge", ""),
        ("filter.lfs.required", "false"),
    ]
    assert env["GIT_CONFIG_VALUE_1"].startswith("Authorization: Basic ")
    assert (env["GIT_ASKPASS"], env["SSH_ASKPASS"], env["GIT_TERMINAL_PROMPT"]) == ("", "", "0")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_never_fills_a_credential_from_a_configured_helper(tmp_path):
    global_config = tmp_path / "gitconfig"
    global_config.write_text(
        '[credential]\n\thelper = "!f() { echo username=u; echo password=stored-SECRET; }; f"\n'
    )
    isolated = {"GIT_CONFIG_GLOBAL": str(global_config), "GIT_CONFIG_NOSYSTEM": "1"}

    def fill(env):
        return subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    # The helper works when git is left alone...
    assert "stored-SECRET" in fill({**os.environ, **isolated, "GIT_TERMINAL_PROMPT": "0"}).stdout
    # ...and is never asked under the environment cognee runs git with.
    hardened = fill(resolve_module._git_env(isolated))
    assert hardened.returncode != 0
    assert "stored-SECRET" not in hardened.stdout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec", ["git@github.com:org/repo.git", "ssh://git:pw-SECRET@github.com/org/repo.git"]
)
async def test_credentials_for_an_ssh_remote_are_refused(monkeypatch, tmp_path, spec):
    """They would only ride an http header an ssh remote never sends, while the clone
    would still be marked as made with credentials."""
    run_git = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(resolve_module, "_run_git", run_git)
    monkeypatch.setattr(resolve_module, "_exec_git", AsyncMock(return_value=(1, "", "")))

    with pytest.raises(resolve_module.CodeRepositoryError, match="only supported for http") as err:
        await resolve_module.resolve_repo_source(spec, clones_dir=tmp_path, credentials="tok")

    assert "pw-SECRET" not in str(err.value)
    run_git.assert_not_awaited()
    assert not any(tmp_path.iterdir())
