"""Recognising repositories add() could not see by URL shape (SDK-793).

ssh remotes are recognised by shape, other hosts' repository URLs by asking the
server (``probe_git_remote``), and git never waits for input. No network: the
probe's HTTP client runs on an httpx MockTransport.
"""

import importlib
from unittest.mock import AsyncMock

import httpx
import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
ssrf_module = importlib.import_module("cognee.tasks.web_scraper.ssrf_protection")

# Captured at import, before the unit conftest stubs the probe for every test.
_REAL_ANSWERS_AS_GIT_SERVER = resolve_module._answers_as_git_server

_ADVERTISEMENT = "application/x-git-upload-pack-advertisement"


@pytest.mark.parametrize(
    "spec",
    [
        "git@github.com:org/repo.git",
        "git@github.com:org/repo",
        "git@git.example.com:repo.git",
        "deploy@git.example.com:group/sub/repo.git",
        "ssh://git@github.com/org/repo.git",
        "ssh://git@git.example.com:2222/org/repo",
        "  git@github.com:org/repo.git  ",
    ],
)
def test_ssh_remotes_are_recognised(spec):
    assert resolve_module.code_repo_ssh_url(spec) == spec.strip()
    assert resolve_module.is_remote_repo(spec.strip())


@pytest.mark.parametrize(
    "spec",
    [
        "bob@example.com",
        "bob@example.com:42",
        "write to bob@example.com: about the repo",
        "git@github.com:org/repo.git is the repo",
        "ssh://",
        "ssh://host",
        "https://github.com/org/repo",
        "/local/path",
        "",
        None,
        42,
    ],
)
def test_text_and_other_specs_are_not_ssh_remotes(spec):
    assert resolve_module.code_repo_ssh_url(spec) is None


def _serve(monkeypatch, handler):
    """Run the probe's real HTTP check against ``handler``; return the requests it sent."""
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(record), **kwargs)

    monkeypatch.setattr(resolve_module, "_answers_as_git_server", _REAL_ANSWERS_AS_GIT_SERVER)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(ssrf_module, "validate_outbound_url", AsyncMock())
    return requests


@pytest.mark.asyncio
async def test_a_git_server_answer_makes_the_url_a_repository(monkeypatch):
    requests = _serve(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": _ADVERTISEMENT}),
    )

    clone_url = await resolve_module.probe_git_remote("https://git.example.com/team/repo/?x=1")

    assert clone_url == "https://git.example.com/team/repo"
    (request,) = requests
    assert str(request.url) == (
        "https://git.example.com/team/repo/info/refs?service=git-upload-pack"
    )
    assert "authorization" not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, headers={"content-type": "text/html"}, text="<html></html>"),
        httpx.Response(401, headers={"www-authenticate": "Basic"}),
        httpx.Response(404),
        httpx.Response(302, headers={"location": "http://169.254.169.254/"}),
    ],
)
async def test_any_other_answer_leaves_the_url_a_web_page(monkeypatch, response):
    requests = _serve(monkeypatch, lambda request: response)

    assert await resolve_module.probe_git_remote("https://example.com/blog/post") is None
    # Redirects are never followed.
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_a_network_error_leaves_the_url_a_web_page(monkeypatch):
    def fail(request):
        raise httpx.ConnectTimeout("timed out", request=request)

    _serve(monkeypatch, fail)

    assert await resolve_module.probe_git_remote("https://example.com/page") is None


@pytest.mark.asyncio
async def test_an_ssrf_refusal_sends_no_request(monkeypatch):
    requests = _serve(monkeypatch, lambda request: httpx.Response(200))
    monkeypatch.setattr(
        ssrf_module,
        "validate_outbound_url",
        AsyncMock(side_effect=ssrf_module.SSRFProtectionError("internal address")),
    )

    assert await resolve_module.probe_git_remote("http://10.0.0.5/team/repo") is None
    assert requests == []


@pytest.mark.asyncio
async def test_disabled_http_requests_send_no_probe(monkeypatch):
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", _REAL_ANSWERS_AS_GIT_SERVER)
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")
    sent = AsyncMock()
    monkeypatch.setattr(httpx.AsyncClient, "get", sent)

    assert await resolve_module.probe_git_remote("https://git.example.com/team/repo") is None
    sent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        # Shape already decided these: code_repo_clone_url said no.
        "https://github.com/org/repo/blob/main/README.md",
        "https://gitlab.com/group/repo/-/issues/1",
        "https://example.com",
        "ssh://git@example.com/org/repo",
        "https://example.com/team/repo is where the code lives",
        "some text",
        None,
    ],
)
async def test_the_probe_is_not_sent_for(monkeypatch, spec):
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", check)

    assert await resolve_module.probe_git_remote(spec) is None
    check.assert_not_awaited()


@pytest.mark.asyncio
async def test_answers_are_cached_per_credential_free_url(monkeypatch):
    check = AsyncMock(return_value=False)
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", check)

    await resolve_module.probe_git_remote("https://example.com/page")
    await resolve_module.probe_git_remote("https://example.com/page/")
    await resolve_module.probe_git_remote("https://token@example.com/page")

    check.assert_awaited_once_with("https://example.com/page")


@pytest.mark.asyncio
async def test_userinfo_is_kept_for_the_clone_but_never_sent_to_the_probe(monkeypatch):
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(resolve_module, "_answers_as_git_server", check)

    clone_url = await resolve_module.probe_git_remote("https://user:tok@git.example.com/o/r")

    assert clone_url == "https://user:tok@git.example.com/o/r"
    check.assert_awaited_once_with("https://git.example.com/o/r")


@pytest.mark.asyncio
async def test_credentials_are_refused_for_an_ssh_remote(tmp_path):
    with pytest.raises(resolve_module.CodeRepositoryError, match="only apply to https"):
        await resolve_module.resolve_repo_source(
            "git@github.com:org/repo.git", clones_dir=tmp_path, credentials="token"
        )


@pytest.mark.asyncio
async def test_git_runs_without_prompts(monkeypatch):
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git = resolve_module.shutil.which("git")

    env = await resolve_module._noninteractive_env(git)

    assert env == {"GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}


@pytest.mark.asyncio
async def test_the_users_ssh_command_is_kept(monkeypatch, tmp_path):
    git = resolve_module.shutil.which("git")
    monkeypatch.delenv("GIT_SSH", raising=False)

    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /keys/deploy")
    env = await resolve_module._noninteractive_env(git)
    assert env["GIT_SSH_COMMAND"] == "ssh -i /keys/deploy -o BatchMode=yes"

    monkeypatch.delenv("GIT_SSH_COMMAND")
    config = tmp_path / "gitconfig"
    config.write_text("[core]\n\tsshCommand = ssh -F /etc/cognee_ssh\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    env = await resolve_module._noninteractive_env(git)
    assert env["GIT_SSH_COMMAND"] == "ssh -F /etc/cognee_ssh -o BatchMode=yes"

    # A legacy GIT_SSH program would be overridden by GIT_SSH_COMMAND: leave it.
    monkeypatch.setenv("GIT_SSH", "/usr/local/bin/my-ssh")
    env = await resolve_module._noninteractive_env(git)
    assert env == {"GIT_TERMINAL_PROMPT": "0"}
