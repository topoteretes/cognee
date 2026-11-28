"""Real git/libcurl must use the pin and refuse redirects to a new destination."""

import importlib
import ipaddress
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.code_graph.resolve_repo import _credential_env, _pinned_http_env


def test_git_uses_pinned_address_and_refuses_redirects():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/private")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://pinning.invalid:{server.server_port}/repo.git"
        # Loopback is a test fixture only. Production supplies validate_outbound_url's
        # public addresses; pinning.invalid has no usable public DNS resolution.
        env = _pinned_http_env(url, [ipaddress.ip_address("127.0.0.1")], None)
        result = subprocess.run(
            ["git", "ls-remote", url],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode != 0
        assert len(requests) == 1
        assert "302" in result.stderr
        assert "port 1" not in result.stderr  # No connection to the redirect target.
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_pinning_preserves_scoped_credentials():
    url = "https://github.com/org/repo.git"
    auth = _credential_env("test-token", url)
    result = _pinned_http_env(url, [ipaddress.ip_address("140.82.112.3")], auth)
    assert result["GIT_CONFIG_KEY_0"] == auth["GIT_CONFIG_KEY_0"]
    assert result["GIT_CONFIG_VALUE_0"] == auth["GIT_CONFIG_VALUE_0"]
    values = {
        result[f"GIT_CONFIG_KEY_{i}"]: result[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(result["GIT_CONFIG_COUNT"]))
    }
    assert values["http.curloptResolve"] == "github.com:443:140.82.112.3"
    assert values["http.followRedirects"] == "false"
    assert values["http.sslVerify"] == "true"


resolver = importlib.import_module("cognee.tasks.code_graph.resolve_repo")


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["git version 2.34.1", "not git"])
async def test_old_or_unknown_git_fails_before_network(monkeypatch, version):
    monkeypatch.setattr(resolver, "_pin_capable_git", set())
    monkeypatch.setattr(resolver, "_execute_git", AsyncMock(return_value=(0, version, "")))
    with pytest.raises(resolver.CodeRepositoryError, match="Git 2.37"):
        await resolver._require_pinning_git()


@pytest.mark.asyncio
async def test_git_version_is_checked_once_per_binary(monkeypatch):
    monkeypatch.setattr(resolver, "_pin_capable_git", set())
    execute = AsyncMock(return_value=(0, "git version 2.37.0.windows.1", ""))
    monkeypatch.setattr(resolver, "_execute_git", execute)
    await resolver._require_pinning_git()
    await resolver._require_pinning_git()
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_global_credential_helper_survives_but_url_rewrites_do_not(monkeypatch, tmp_path):
    config = tmp_path / "gitconfig"
    config.write_text("""[credential "https://pinning.invalid"]
    helper = "!f() { echo username=test; echo password=test-password; }; f"
[http]
    sslCAInfo = /explicit/ca.pem
[url "http://127.0.0.1/"]
    insteadOf = https://pinning.invalid/
""")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    url = "https://pinning.invalid/repo.git"
    env = resolver._pinned_http_env(
        url, [ipaddress.ip_address("8.8.8.8")], await resolver._http_user_config()
    )
    result = subprocess.run(
        ["git", "credential", "fill"],
        input=f"url={url}\n\n",
        text=True,
        capture_output=True,
        env={**os.environ, **env},
        check=True,
    )
    assert "password=test-password" in result.stdout
    values = {
        env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(env["GIT_CONFIG_COUNT"]))
    }
    assert values["http.sslcainfo"] == "/explicit/ca.pem"
    assert not any(k.startswith("url.") for k in values)


def test_ipv6_literal_needs_no_curl_resolution_entry():
    env = resolver._pinned_http_env(
        "https://[2606:4700:4700::1111]/r.git", [ipaddress.ip_address("2606:4700:4700::1111")], None
    )
    entries = [
        env[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(env["GIT_CONFIG_COUNT"]))
        if env[f"GIT_CONFIG_KEY_{i}"] == "http.curloptResolve"
    ]
    assert entries == [""]


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", [None, "dev"])
@pytest.mark.parametrize("existing", [False, True])
async def test_every_http_git_operation_keeps_pin_and_refresh_uses_validated_url(
    monkeypatch, tmp_path, branch, existing
):
    from cognee.tasks.web_scraper import ssrf_protection

    url = "https://public.example/o/r.git"
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    monkeypatch.setattr(
        ssrf_protection,
        "validate_outbound_url",
        AsyncMock(return_value=[ipaddress.ip_address("8.8.8.8")]),
    )
    monkeypatch.setattr(resolver, "_http_user_config", AsyncMock(return_value={}))
    monkeypatch.setattr(resolver, "_require_pinning_git", AsyncMock())
    target = resolver.clone_dir(url, branch, tmp_path)
    if existing:
        (target / ".git").mkdir(parents=True)
        (target / ".git" / "config").write_text(f'[remote "origin"]\nurl = {url}\n')
    git = AsyncMock(return_value=(0, ""))
    monkeypatch.setattr(resolver, "_run_git", git)
    await resolver.resolve_repo_source(url, clones_dir=tmp_path, branch=branch)
    for call in git.await_args_list:
        env = call.kwargs["env"]
        assert "public.example:443:8.8.8.8" in env.values()
        if call.args[0][0] in {"fetch", "pull"}:
            assert url in call.args[0]
            assert "origin" not in call.args[0]


@pytest.mark.asyncio
async def test_mismatched_cached_origin_is_refused_before_fetch(monkeypatch, tmp_path):
    from cognee.tasks.web_scraper import ssrf_protection

    url = "https://public.example/6379/a/b.git"
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "true")
    monkeypatch.setattr(
        ssrf_protection,
        "validate_outbound_url",
        AsyncMock(return_value=[ipaddress.ip_address("8.8.8.8")]),
    )
    monkeypatch.setattr(resolver, "_http_user_config", AsyncMock(return_value={}))
    monkeypatch.setattr(resolver, "_require_pinning_git", AsyncMock())
    target = resolver.clone_dir(url, clones_dir=tmp_path)
    (target / ".git").mkdir(parents=True)
    (target / ".git" / "config").write_text(
        '[remote "origin"]\nurl = http://public.example:6379/a/b.git\n'
    )
    git = AsyncMock()
    monkeypatch.setattr(resolver, "_run_git", git)
    with pytest.raises(resolver.CodeRepositoryError, match="different origin"):
        await resolver.resolve_repo_source(url, clones_dir=tmp_path)
    git.assert_not_awaited()


@pytest.mark.parametrize(
    "error", ["fatal: The requested URL returned error: 301", "fatal: HTTP code: 302"]
)
def test_redirect_error_explains_canonical_url(error):
    with pytest.raises(resolver.CodeRepositoryError, match="canonical URL"):
        resolver._raise_on_redirect(error)
