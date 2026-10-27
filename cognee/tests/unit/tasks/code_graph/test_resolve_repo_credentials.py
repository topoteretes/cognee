"""HTTP credentials reach Git's environment without entering paths or errors."""

import importlib
import os
import shutil
import subprocess
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
    monkeypatch.setattr(resolve_module, "_has_commit", AsyncMock(return_value=True))
    # git is mocked, so resolving the host would only tie these tests to the network.
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())


@pytest.fixture(autouse=True)
def _fake_tree_snapshot(monkeypatch):
    original = resolve_module._exec_git

    async def tree_command(args, cwd=None, env=None):
        if args[0] == "rev-parse":
            return 0, "a" * 40, ""
        if args[0] == "ls-tree":
            return 0, "", ""
        return await original(args, cwd=cwd, env=env)

    monkeypatch.setattr(resolve_module, "_exec_git", tree_command)
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.code_graph.repo_selection"),
        "read_repository_snapshot",
        lambda path: {"commit": "a" * 40},
    )


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

    # Slug comes from the clean URL , same directory a credential-free call
    # (or a later sync with a rotated token) would use.
    assert resolved == resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)

    clone_args, _ = git_calls[0]
    # core.symlinks=false is a security requirement, not incidental: without it a
    # symlink committed in the remote is materialized and later written through.
    assert clone_args[:5] == ["clone", "-c", "core.symlinks=false", "--depth", "1"]
    assert _CLEAN_URL in clone_args
    assert "tok-SECRET" not in str(git_calls)

    assert len(git_calls) == 6


@pytest.mark.asyncio
async def test_url_credentials_are_removed_before_fetch(monkeypatch, tmp_path):
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
    assert git_calls == [
        (
            [
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-recurse-submodules",
                _CLEAN_URL,
                "HEAD",
            ],
            clone_dir,
        ),
        (["config", "remote.origin.promisor", "true"], clone_dir),
        (["config", "remote.origin.partialclonefilter", "blob:none"], clone_dir),
        (["config", "core.sparseCheckout", "true"], clone_dir),
        (["config", "core.sparseCheckoutCone", "false"], clone_dir),
        (["reset", "--hard", "a" * 40], clone_dir),
    ]


@pytest.mark.asyncio
async def test_credential_free_urls_fetch_the_verified_remote(monkeypatch, tmp_path):
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
                _CLEAN_URL,
                "HEAD",
            ],
            clone_dir,
        ),
        (["config", "remote.origin.promisor", "true"], clone_dir),
        (["config", "remote.origin.partialclonefilter", "blob:none"], clone_dir),
        (["config", "core.sparseCheckout", "true"], clone_dir),
        (["config", "core.sparseCheckoutCone", "false"], clone_dir),
        (["reset", "--hard", "a" * 40], clone_dir),
    ]


@pytest.mark.asyncio
async def test_out_of_band_credentials_ride_the_environment(monkeypatch, tmp_path):
    """The preferred path (connectors): token via ``credentials=``, not the URL.

    The URL stays clean everywhere , argv, slug, remote , and the token
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
    clone_args, _cwd, clone_env = git_calls[0]
    # argv carries only the clean URL; the token is nowhere in it.
    assert clone_args == [
        "clone",
        "-c",
        "core.symlinks=false",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-checkout",
        "--no-local",
        "--no-recurse-submodules",
        _CLEAN_URL,
        clone_args[-1],
    ]
    # Cloned beside its target, then renamed into place.
    assert resolve_module.Path(clone_args[-1]).parent == tmp_path
    assert "tok-SECRET" not in " ".join(clone_args)
    # ...and rides GIT_CONFIG_* as a basic-auth header for github.com only instead.
    assert clone_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/org/repo.git/.extraHeader"
    assert clone_env["GIT_CONFIG_VALUE_1"].startswith("Authorization: Basic ")


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

    fetch_args, fetch_cwd, fetch_env = git_calls[0]
    reset_args, reset_cwd, reset_env = git_calls[-1]
    # Fetch against the clean stored remote; auth via the environment.
    assert fetch_args == [
        "fetch",
        "--depth",
        "1",
        "--filter=blob:none",
        "--no-recurse-submodules",
        _CLEAN_URL,
        "HEAD",
    ]
    assert fetch_cwd == reset_cwd == clone_dir
    assert fetch_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/org/repo.git/.extraHeader"
    assert reset_args == ["reset", "--hard", "a" * 40]
    assert reset_env == fetch_env


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

    async def failing_refresh_echoing_the_url(args, cwd=None, env=None):
        # git echoes the URL it was given, credentials included, in its error output.
        return 128, f"fatal: Authentication failed for '{spec}'"

    monkeypatch.setattr(resolve_module, "_run_git", failing_refresh_echoing_the_url)

    with pytest.raises(
        resolve_module.CodeRepositoryError, match="Could not (refresh|checkout)"
    ) as error:
        await resolve_module.resolve_repo_source(
            spec, clones_dir=tmp_path, credentials=credentials, branch=branch
        )

    if secret:
        assert secret not in str(error.value)
    assert _CLEAN_URL in str(error.value)
    # Refused, not discarded: a later caller with valid credentials refreshes it.
    assert (clone_dir / ".git").is_dir()


@pytest.mark.asyncio
async def test_a_credentialed_clone_is_reused_after_a_refresh_with_the_callers_credentials(
    monkeypatch, tmp_path
):
    clone_dir = resolve_module.clone_dir(_CLEAN_URL, clones_dir=tmp_path)
    (clone_dir / ".git").mkdir(parents=True)
    seen_env = []

    async def fake_run_git(args, cwd=None, env=None):
        seen_env.append(env)
        return 0, ""

    monkeypatch.setattr(resolve_module, "_run_git", fake_run_git)

    resolved = await resolve_module.resolve_repo_source(
        _CLEAN_URL, clones_dir=tmp_path, credentials="tok-SECRET"
    )

    assert resolved == clone_dir
    assert seen_env[0]["GIT_CONFIG_VALUE_1"].startswith("Authorization: Basic ")


def test_the_credential_header_is_scoped_to_the_remote_host():
    env = resolve_module._credential_env("tok-SECRET", "https://GitHub.com:8443/org/repo.git")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com:8443/org/repo.git/.extraHeader"
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
    assert header_for("https://github.com/other/repo") == ""
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
        ("http.https://github.com/org/repo.git/.extraHeader", ""),
        ("http.https://github.com/org/repo.git/.extraHeader", env["GIT_CONFIG_VALUE_2"]),
        ("credential.helper", ""),
        ("core.askPass", ""),
        ("filter.lfs.process", ""),
        ("filter.lfs.smudge", ""),
        ("filter.lfs.required", "false"),
        ("core.longpaths", "true"),
        ("core.ignorecase", "false"),
        ("maintenance.auto", "false"),
        ("gc.auto", "0"),
    ]
    assert env["GIT_CONFIG_VALUE_2"].startswith("Authorization: Basic ")
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
    """HTTP credentials cannot authenticate an SSH transport."""
    run_git = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(resolve_module, "_run_git", run_git)
    monkeypatch.setattr(resolve_module, "_exec_git", AsyncMock(return_value=(1, "", "")))

    with pytest.raises(resolve_module.CodeRepositoryError, match="require an http") as err:
        await resolve_module.resolve_repo_source(spec, clones_dir=tmp_path, credentials="tok")

    assert err.value.status_code == 400
    assert "pw-SECRET" not in str(err.value)
    run_git.assert_not_awaited()
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_an_ssh_url_with_a_password_is_refused_as_a_bad_request(monkeypatch, tmp_path):
    run_git = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(resolve_module, "_run_git", run_git)

    with pytest.raises(resolve_module.CodeRepositoryError, match="key authentication") as err:
        await resolve_module.resolve_repo_source(
            "ssh://git:pw-SECRET@github.com/org/repo.git", clones_dir=tmp_path
        )

    assert err.value.status_code == 400
    assert "pw-SECRET" not in str(err.value)
    run_git.assert_not_awaited()
