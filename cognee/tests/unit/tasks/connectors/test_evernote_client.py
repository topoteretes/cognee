# Adapted from topoteretes/cognee-community (Apache-2.0), PR #559, commit 25dc254 (blvckShawn).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""The real ``EvernoteApi`` with only the Thrift layer faked: error mapping, retries, waits, and
that no token, guid or URL reaches a message or a log line.

CI installs no provider SDK, so the EDAM error classes are stand-ins with the generated classes'
shape (``errorCode``, ``rateLimitDuration``, ``identifier``/``key``). The last test runs against
the real classes when the ``evernote`` extra is installed.
"""

import importlib.util
import logging
import sys
from enum import IntEnum
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.connectors import evernote as module
from cognee.tasks.ingestion.connectors.evernote import (
    EvernoteApi,
    EvernoteAuthError,
    EvernoteError,
    EvernotePermissionError,
    EvernoteRateLimitError,
    EvernoteUnavailableError,
    NoteNotFoundError,
    _iter_rows,
    new_stats,
)

TOKEN = "S=s1:U=3e9:E=19a:C=18f:P=1cd:A=app:V=2:H=0123456789abcdef"
GUID = "6f1c0d2e-aaaa-bbbb-cccc-1234567890ab"
REAL_EDAM = module._edam

pytestmark = pytest.mark.timeout(60)
BROKEN = "0a9b8c7d-retry-retry-retry-000000000003"


class Codes(IntEnum):
    PERMISSION_DENIED = 3
    INTERNAL_ERROR = 4
    INVALID_AUTH = 8
    AUTH_EXPIRED = 9
    RATE_LIMIT_REACHED = 19


class TException(Exception):
    pass


class TTransportException(TException):
    pass


class EDAMUserException(TException):
    def __init__(self, errorCode, parameter=None):
        super().__init__(errorCode, parameter)
        self.errorCode, self.parameter = errorCode, parameter


class EDAMSystemException(TException):
    def __init__(self, errorCode, message=None, rateLimitDuration=None):
        super().__init__(errorCode, message, rateLimitDuration)
        self.errorCode, self.message, self.rateLimitDuration = errorCode, message, rateLimitDuration


class EDAMNotFoundException(TException):
    def __init__(self, identifier=None, key=None):
        super().__init__(identifier, key)
        self.identifier, self.key = identifier, key


FAKE_EDAM = SimpleNamespace(
    chunk_filter=SimpleNamespace,
    codes=Codes,
    user=EDAMUserException,
    system=EDAMSystemException,
    not_found=EDAMNotFoundException,
    thrift=TException,
    transport=TTransportException,
)


@pytest.fixture(autouse=True)
def fake_thrift(monkeypatch):
    monkeypatch.setattr(module, "_edam", lambda: FAKE_EDAM)


class Store:
    """A UserStore and NoteStore in one: each method answers from ``script``, a list of results
    or exceptions per method, and records the arguments it was called with."""

    def __init__(self, **script):
        self.script = {name: list(answers) for name, answers in script.items()}
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.__dict__.get("script", {}):
            raise AttributeError(name)

        def call(*args):
            self.calls.append((name, *args))
            answers = self.script[name]
            answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(answer, BaseException):
                raise answer
            return answer

        return call


def _note(content: str, guid: str = GUID) -> SimpleNamespace:
    return SimpleNamespace(guid=guid, title="t", content=content, notebookGuid="nb1", tagGuids=None)


def _client(sleeps=None, **script):
    store = Store(**script)
    client = EvernoteApi(
        TOKEN,
        user_store=store,
        note_store=store,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )
    return client, store


def test_construction_makes_no_call_and_hides_the_token():
    client = EvernoteApi(TOKEN)
    assert TOKEN not in repr(client) and "redacted" in repr(client)
    with pytest.raises(EvernoteAuthError, match="token is required"):
        EvernoteApi("  ")


def test_every_call_carries_the_token_as_its_first_argument():
    client, store = _client(
        getUser=[SimpleNamespace(id=7)],
        getSyncState=["state"],
        getFilteredSyncChunk=["chunk"],
        getNote=[_note("<en-note/>")],
    )

    assert client.get_user_id() == 7 and client.get_user_id() == 7
    client.get_sync_state()
    client.get_filtered_sync_chunk(10, 250)
    client.get_note(GUID)

    assert [call[:2] for call in store.calls] == [
        ("getUser", TOKEN),
        ("getSyncState", TOKEN),
        ("getFilteredSyncChunk", TOKEN),
        ("getNote", TOKEN),
    ], "the user id is read once"
    after_usn, max_entries, chunk_filter = store.calls[2][2:]
    assert (after_usn, max_entries) == (10, 250)
    assert vars(chunk_filter) == {
        "includeNotes": True,
        "includeNotebooks": True,
        "includeTags": True,
        "includeExpunged": True,
    }, "notes, notebooks, tags and the expunged guids, nothing else"
    assert store.calls[3][2:] == (GUID, True, False, False, False), "content, no resource data"


def test_a_short_rate_limit_is_waited_out_and_the_call_answers():
    sleeps: list[float] = []
    client, _ = _client(
        sleeps,
        getSyncState=[EDAMSystemException(Codes.RATE_LIMIT_REACHED, rateLimitDuration=10), "ok"],
    )

    assert client.get_sync_state() == "ok"
    assert sleeps == [10.0]


def test_a_rate_limit_past_the_bound_raises_without_sleeping():
    sleeps: list[float] = []
    client, _ = _client(
        sleeps,
        getSyncState=[EDAMSystemException(Codes.RATE_LIMIT_REACHED, rateLimitDuration=900)],
    )

    with pytest.raises(EvernoteRateLimitError) as raised:
        client.get_sync_state()
    assert raised.value.retry_after == 900 and sleeps == []


def test_repeated_rate_limits_stop_once_their_waits_pass_the_bound():
    sleeps: list[float] = []
    client, _ = _client(
        sleeps,
        getSyncState=[EDAMSystemException(Codes.RATE_LIMIT_REACHED, rateLimitDuration=12)],
    )

    with pytest.raises(EvernoteRateLimitError):
        client.get_sync_state()
    assert sleeps == [12.0, 12.0], "a third wait would pass 30 seconds"


def test_a_transport_error_is_retried_with_backoff_then_reported_without_its_text():
    sleeps: list[float] = []
    client, store = _client(
        sleeps, getSyncState=[TTransportException(f"https://www.evernote.com/x?{TOKEN}")]
    )

    with pytest.raises(EvernoteUnavailableError) as raised:
        client.get_sync_state()
    assert sleeps == [1.0, 2.0, 4.0] and len(store.calls) == 4
    assert str(raised.value) == "Evernote did not answer after 4 attempts"


def test_a_5xx_like_internal_error_is_not_retried():
    """Evernote's 5xx is EDAMSystemException INTERNAL_ERROR in an HTTP 200; it fails the call."""
    client, store = _client(getSyncState=[EDAMSystemException(Codes.INTERNAL_ERROR, TOKEN)])

    with pytest.raises(EvernoteUnavailableError, match=r"system error INTERNAL_ERROR \(4\)"):
        client.get_sync_state()
    assert len(store.calls) == 1


@pytest.mark.parametrize("code", [Codes.INVALID_AUTH, Codes.AUTH_EXPIRED])
def test_a_rejected_token_is_an_auth_error_the_equivalent_of_a_401(code):
    client, _ = _client(getUser=[EDAMUserException(code, parameter=TOKEN)])

    with pytest.raises(EvernoteAuthError, match=code.name):
        client.get_user_id()


def test_permission_denied_is_a_permission_error_the_equivalent_of_a_403():
    client, _ = _client(getNote=[EDAMUserException(Codes.PERMISSION_DENIED, GUID)])

    with pytest.raises(EvernotePermissionError):
        client.get_note(GUID)


def test_not_found_is_a_note_not_found_error_without_the_guid_the_equivalent_of_a_404():
    """EV2: the community message named the guid."""
    client, _ = _client(getNote=[EDAMNotFoundException(identifier="Note.guid", key=GUID)])

    with pytest.raises(NoteNotFoundError) as raised:
        client.get_note(GUID)
    assert GUID not in str(raised.value) and GUID not in repr(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


def test_a_programming_error_propagates_unchanged():
    client, _ = _client(getSyncState=[TypeError("bad call")])

    with pytest.raises(TypeError, match="bad call"):
        client.get_sync_state()


def test_check_active_runs_before_every_attempt_and_every_wait():
    checks: list[int] = []
    client, store = _client(
        getSyncState=[EDAMSystemException(Codes.RATE_LIMIT_REACHED, rateLimitDuration=1), "ok"]
    )

    client.get_sync_state(check_active=lambda: checks.append(len(store.calls)))

    assert checks == [0, 1, 1], "before the call, before the wait, before the retry"


def test_a_note_store_on_another_host_is_refused():
    users = Store(getUserUrls=[SimpleNamespace(noteStoreUrl="https://evil.example/shard/s1")])
    client = EvernoteApi(TOKEN, user_store=users)

    with pytest.raises(EvernoteError, match="unexpected host"):
        client.get_sync_state()


def test_a_run_through_the_real_client_leaks_no_token_or_guid(caplog):
    """X3: every failure on the way carries the token and the guid; none of it comes out."""
    chunk = SimpleNamespace(
        chunkHighUSN=5,
        updateCount=5,
        notebooks=[SimpleNamespace(guid="nb1", name="Work")],
        tags=None,
        notes=[
            SimpleNamespace(
                guid=BROKEN,
                title="t3",
                notebookGuid="nb1",
                tagGuids=None,
                updateSequenceNum=3,
                active=True,
                deleted=None,
            ),
            SimpleNamespace(
                guid=GUID,
                title="t",
                notebookGuid="nb1",
                tagGuids=None,
                updateSequenceNum=5,
                active=True,
                deleted=None,
            ),
            SimpleNamespace(
                guid="g2",
                title="t2",
                notebookGuid="nb1",
                tagGuids=None,
                updateSequenceNum=4,
                active=True,
                deleted=None,
            ),
        ],
        expungedNotes=None,
        expungedNotebooks=None,
        expungedTags=None,
    )
    client, _ = _client(
        getUser=[SimpleNamespace(id=1)],
        getSyncState=[SimpleNamespace(currentTime=1, fullSyncBefore=0, updateCount=5)],
        getFilteredSyncChunk=[chunk],
        getNote=[
            EDAMUserException(Codes.PERMISSION_DENIED, parameter=f"{TOKEN} {BROKEN}"),
            _note("<en-note>ok</en-note>", "g2"),
            EDAMSystemException(
                Codes.RATE_LIMIT_REACHED, message=f"{TOKEN} {GUID}", rateLimitDuration=3600
            ),
        ],
    )
    state: dict = {}
    stats = new_stats()

    with caplog.at_level(logging.DEBUG):
        rows = list(_iter_rows(client, state, stats))

    assert stats["failed_rate_limit"] == 1 and [r["id"] for r in rows] == ["note:g2"]
    assert list(state["retry"]) == [BROKEN] and "could not be fetched" in caplog.text
    assert TOKEN not in caplog.text and GUID not in caplog.text and BROKEN not in caplog.text
    assert TOKEN not in repr(state) and TOKEN not in repr(stats)

    client, _ = _client(getUser=[EDAMUserException(Codes.AUTH_EXPIRED, parameter=TOKEN)])
    with pytest.raises(EvernoteAuthError) as raised:
        list(_iter_rows(client, {}, new_stats()))
    assert TOKEN not in str(raised.value) and TOKEN not in repr(raised.value)


def test_the_real_thrift_classes_map_the_same_way(monkeypatch):
    """Runs where the ``evernote`` extra is installed (the ubuntu unit legs): the mapping
    against the generated classes. An install that is there but broken fails here."""
    if importlib.util.find_spec("evernote") is None:
        pytest.skip("the evernote extra is not installed")
    monkeypatch.setattr(module, "_edam", REAL_EDAM)
    from evernote.edam.error.ttypes import EDAMErrorCode, EDAMNotFoundException, EDAMUserException

    REAL_EDAM()  # every class the client imports, the Thrift ones included

    client, _ = _client(
        getUser=[EDAMUserException(errorCode=EDAMErrorCode.AUTH_EXPIRED, parameter=TOKEN)],
        getNote=[EDAMNotFoundException(identifier="Note.guid", key=GUID)],
    )
    with pytest.raises(EvernoteAuthError) as raised:
        client.get_user_id()
    assert TOKEN not in str(raised.value)
    with pytest.raises(NoteNotFoundError) as missing:
        client.get_note(GUID)
    assert GUID not in str(missing.value)


@pytest.mark.parametrize("missing", ["thrift", "evernote"])
def test_a_missing_client_library_names_the_extra_to_install(monkeypatch, missing):
    """Without ``cognee[evernote]`` the first call says what to install, not just which module."""
    for name in list(sys.modules):
        if name == missing or name.startswith(missing + "."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(ImportError, match=r"cognee\[evernote\]"):
        REAL_EDAM()
    with pytest.raises(ImportError, match=r"cognee\[evernote\]"):
        module.EvernoteApi("token")._thrift_client("https://www.evernote.com/edam/user", "user")
