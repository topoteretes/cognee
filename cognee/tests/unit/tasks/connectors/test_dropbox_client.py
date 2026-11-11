# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: the community used the dropbox SDK; this tests its httpx client.
"""The Dropbox client through a faked transport: header, typed errors, nothing secret in them."""

import httpx
import pytest

from cognee.tasks.ingestion.connectors.dropbox import (
    DropboxApiError,
    DropboxAuthError,
    DropboxClient,
    DropboxRateLimitedError,
    build_dropbox_service,
    dropbox_account_id,
)

TOKEN = "sl.secret-token-abc"
FILE_ID = "id:SecretFileId123"
SECRET_PATH = "/Private/salary.md"


def client_for(handler, token=TOKEN):
    return DropboxClient(
        token, http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None
    )


def assert_clean(error: BaseException) -> None:
    text = f"{error!s} {error!r} {error.args}"
    for secret in (TOKEN, FILE_ID, SECRET_PATH, "dropboxapi.com"):
        assert secret not in text
    assert error.__cause__ is None and error.__suppress_context__ or error.__context__ is None


def test_the_token_is_sent_as_a_bearer_header_and_never_follows_a_redirect():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    assert client_for(handler).rpc("files/get_metadata", {"path": FILE_ID}) == {"ok": True}
    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert str(seen[0].url) == "https://api.dropboxapi.com/2/files/get_metadata"

    redirect = client_for(lambda _r: httpx.Response(302, headers={"Location": "https://x.test"}))
    with pytest.raises(DropboxApiError) as error:
        redirect.rpc("files/get_metadata", {"path": FILE_ID})
    assert error.value.status == 302


def test_a_token_callable_is_asked_before_every_attempt():
    tokens = iter(["first-token", TOKEN])
    answers = iter([httpx.Response(503), httpx.Response(200, json={})])
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return next(answers)

    client_for(handler, token=lambda: next(tokens)).rpc("files/get_metadata", {})
    assert seen == ["Bearer first-token", f"Bearer {TOKEN}"]


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {".tag": "invalid_access_token"}}, DropboxAuthError),
        (403, {"error": {".tag": "missing_scope"}}, DropboxApiError),
        (404, b"not found " + SECRET_PATH.encode(), DropboxApiError),
        (409, {"error": {".tag": "path", "path": {".tag": "not_found"}}}, DropboxApiError),
        (429, {"error": {".tag": "too_many_requests"}}, DropboxRateLimitedError),
        (500, b"oops " + FILE_ID.encode(), DropboxApiError),
        (503, {}, DropboxApiError),
    ],
)
def test_every_refusal_is_a_typed_error_with_status_and_route_only(status, body, expected):
    def handler(_request):
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers={"Retry-After": "0"})
        return httpx.Response(status, json=body, headers={"Retry-After": "0"})

    with pytest.raises(expected) as error:
        client_for(handler).rpc("files/get_metadata", {"path": FILE_ID})

    assert error.value.status == status and error.value.endpoint == "files/get_metadata"
    assert_clean(error.value)


def test_a_409_carries_the_error_tags():
    def handler(_request):
        return httpx.Response(409, json={"error": {".tag": "path", "path": {".tag": "not_found"}}})

    with pytest.raises(DropboxApiError) as error:
        client_for(handler).rpc("files/get_metadata", {"path": FILE_ID})
    assert error.value.tags == ("path", "not_found")


def test_a_network_error_is_retried_then_typed_without_the_url():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(DropboxApiError) as error:
        client_for(handler).rpc("files/get_metadata", {"path": FILE_ID})
    assert len(calls) == 4 and error.value.status == 0
    assert_clean(error.value)
    assert error.value.__context__ is None or "dropboxapi" not in str(error.value.__context__)


@pytest.mark.parametrize("token", ["", "  ", "two words", "tökén", "line\nbreak"])
def test_a_token_that_is_not_a_header_value_is_refused_without_echoing_it(token):
    with pytest.raises(DropboxAuthError) as error:
        DropboxClient(token)
    if token.strip():
        assert token not in str(error.value)


def test_the_client_repr_hides_the_token():
    client = build_dropbox_service(TOKEN)
    try:
        assert TOKEN not in repr(client) and "redacted" in repr(client)
    finally:
        client.close()


def test_a_download_streams_into_a_temporary_file_and_sends_the_arg_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"hello")

    with client_for(handler).download("files/download", {"path": FILE_ID}) as stream:
        assert stream.read() == b"hello"
    assert seen[0].url.host == "content.dropboxapi.com"
    assert seen[0].headers["Dropbox-API-Arg"] == '{"path": "id:SecretFileId123"}'


def test_the_account_id_comes_from_get_current_account():
    def handler(request):
        assert request.url.path == "/2/users/get_current_account"
        return httpx.Response(200, json={"account_id": "dbid:AAA", "email": "x@example.com"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    assert dropbox_account_id(TOKEN, http=http) == "dbid:AAA"

    bad = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})))
    with pytest.raises(DropboxApiError):
        dropbox_account_id(TOKEN, http=bad)


def test_a_request_httpx_cannot_send_names_no_token():
    """httpx refuses an illegal header value with LocalProtocolError, whose text may hold it."""

    def handler(request):
        raise httpx.LocalProtocolError(f"Illegal header value {request.headers['Authorization']}")

    with pytest.raises(DropboxAuthError) as error:
        client_for(handler).rpc("files/get_metadata", {"path": FILE_ID})

    for text in (
        str(error.value),
        repr(error.value),
        repr(error.value.__cause__),
        repr(error.value.__context__),
    ):
        assert TOKEN not in text


# ---------------------------------------------------------------- 401s by AuthError tag
class _RenewingToken:
    """A token source like the integration module's cache: ``invalidate`` renews it."""

    def __init__(self, *tokens):
        self.tokens = list(tokens)
        self.renewals = 0

    def __call__(self):
        return self.tokens[0]

    def invalidate(self):
        self.renewals += 1
        self.tokens.pop(0)


def _answer_401(tag, *, accepted=None):
    """Dropbox answers 401 with ``tag`` (None: no JSON body) except for ``accepted``."""
    calls = []

    def handler(request):
        calls.append(request.headers["Authorization"])
        if accepted and request.headers["Authorization"] == f"Bearer {accepted}":
            return httpx.Response(200, json={"ok": True})
        if tag is None:
            return httpx.Response(401, content=b"<html>proxy says no</html>")
        return httpx.Response(401, json={"error_summary": f"{tag}/..", "error": {".tag": tag}})

    return handler, calls


@pytest.mark.parametrize("tag", ["expired_access_token", "invalid_access_token", "other", None])
def test_a_refused_token_is_renewed_once_and_the_request_retried(tag):
    handler, calls = _answer_401(tag, accepted="sl.renewed")
    token = _RenewingToken("sl.stale", "sl.renewed")
    checks = []

    result = client_for(handler, token=token).rpc(
        "files/get_metadata", {"path": FILE_ID}, check_active=lambda: checks.append(1)
    )

    assert result == {"ok": True} and token.renewals == 1
    assert calls == ["Bearer sl.stale", "Bearer sl.renewed"]
    assert len(checks) == 2, "check_active runs before the first attempt and before the retry"


@pytest.mark.parametrize("tag", ["expired_access_token", "invalid_access_token", "other", None])
def test_a_second_401_after_the_renewal_is_an_auth_failure(tag):
    handler, calls = _answer_401(tag)
    token = _RenewingToken("sl.stale", "sl.renewed")

    with pytest.raises(DropboxAuthError) as error:
        client_for(handler, token=token).rpc("files/get_metadata", {"path": FILE_ID})

    assert token.renewals == 1 and len(calls) == 2
    assert_clean(error.value)


@pytest.mark.parametrize("tag", ["missing_scope", "route_access_denied"])
def test_a_missing_scope_is_never_renewed_and_says_to_reconnect(tag):
    from cognee.tasks.ingestion.connectors.dropbox import DropboxScopeError

    handler, calls = _answer_401(tag, accepted="sl.renewed")
    token = _RenewingToken("sl.stale", "sl.renewed")

    with pytest.raises(DropboxScopeError) as error:
        client_for(handler, token=token).rpc("files/get_metadata", {"path": FILE_ID})

    assert token.renewals == 0 and len(calls) == 1
    assert "reconnect" in str(error.value) and "files.content.read" in str(error.value)
    assert isinstance(error.value, DropboxAuthError), "counted as failed_auth"
    assert_clean(error.value)


@pytest.mark.parametrize("tag", ["user_suspended", "invalid_select_user", "invalid_select_admin"])
def test_an_account_refusal_is_never_renewed_or_retried(tag):
    handler, calls = _answer_401(tag, accepted="sl.renewed")
    token = _RenewingToken("sl.stale", "sl.renewed")

    with pytest.raises(DropboxAuthError) as error:
        client_for(handler, token=token).rpc("files/get_metadata", {"path": FILE_ID})

    assert token.renewals == 0 and len(calls) == 1 and error.value.tags == (tag,)
