"""Wire-level tests for ``HttpBackend``: protocol args in, HTTP requests out.

The mirror of ``test_sdk_backend.py`` for the direct-HTTP transport. The provider
tests already pin Hermes behaviour against the protocol; this file pins that the
protocol becomes the request bodies cognee's routers actually accept (verified
against cognee 1.2.1's DTOs).

The headline assertion is :meth:`TestImproveWireFormat.test_session_ids_reach_the_wire`
— the whole reason this transport exists.

A fake opener stands in for ``urlopen``, so nothing here touches the network.
Runs standalone with ``python3 tests/test_http_backend.py``; needs no cognee.
"""

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _char_helpers import make_provider  # noqa: E402
from cognee_integration_hermes.http_backend import (  # noqa: E402
    CogneeHttpError,
    CogneeUnreachable,
    HttpBackend,
)

_TIMEOUT = 5.0
_URL = "http://127.0.0.1:8000"
_REMOTE_URL = "https://cloud.example"


class _Response(io.BytesIO):
    def __init__(self, payload, status=200):
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        super().__init__(body.encode("utf-8") if isinstance(body, str) else body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Records requests and replies from a per-path script."""

    def __init__(self, responses=None):
        self.requests = []
        self.responses = responses or {}
        self.default = {}

    def __call__(self, request, timeout=None):
        path = request.selector if hasattr(request, "selector") else request.full_url
        body = request.data
        self.requests.append(
            {
                "method": request.get_method(),
                "path": path,
                "url": request.full_url,
                "headers": {k.lower(): v for k, v in request.header_items()},
                "body": body,
                "timeout": timeout,
            }
        )
        outcome = self.responses.get(self._key(path), self.default)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    @staticmethod
    def _key(path):
        return path.split("?")[0]

    # -- inspection --------------------------------------------------------

    def paths(self):
        return [r["path"] for r in self.requests]

    def request_for(self, path):
        matches = [r for r in self.requests if r["path"] == path]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly 1 request to {path}, got {len(matches)}")
        return matches[0]

    def json_body(self, path):
        return json.loads(self.request_for(path)["body"].decode("utf-8"))

    def multipart_fields(self, path):
        """Parse a multipart body into ``{name: value}`` (text parts only)."""
        raw = self.request_for(path)["body"].decode("utf-8", errors="replace")
        fields = {}
        for chunk in raw.split("--")[1:]:
            if 'name="' not in chunk:
                continue
            name = chunk.split('name="', 1)[1].split('"', 1)[0]
            if "\r\n\r\n" not in chunk:
                continue
            value = chunk.split("\r\n\r\n", 1)[1]
            fields[name] = value.rsplit("\r\n", 1)[0]
        return fields


def _backend(opener, *, api_key="k", url=_URL):
    backend = HttpBackend(opener=opener)
    backend.url = url
    backend.api_key = api_key
    return backend


class TestRecallWireFormat(unittest.TestCase):
    def _recall(self, **overrides):
        params = {
            "query": "q",
            "session_id": "hermes_s1",
            "datasets": ["hermes"],
            "top_k": 5,
            "auto_route": True,
            "query_type": None,
            "timeout": _TIMEOUT,
        }
        params.update(overrides)
        opener = FakeOpener({"/api/v1/recall": [{"text": "hit"}]})
        results = _backend(opener).recall(**params)
        return opener, results

    def test_posts_to_the_recall_endpoint(self):
        opener, _ = self._recall()
        self.assertEqual(opener.request_for("/api/v1/recall")["method"], "POST")

    def test_query_is_named_query_not_query_text(self):
        # The endpoint's DTO field is ``query``; only the SDK calls it query_text.
        opener, _ = self._recall(query="hello")
        body = opener.json_body("/api/v1/recall")
        self.assertEqual(body["query"], "hello")
        self.assertNotIn("query_text", body)

    def test_session_and_datasets_are_sent_when_targeted(self):
        opener, _ = self._recall()
        body = opener.json_body("/api/v1/recall")
        self.assertEqual(body["session_id"], "hermes_s1")
        self.assertEqual(body["datasets"], ["hermes"])
        self.assertEqual(body["top_k"], 5)

    def test_none_targets_are_omitted(self):
        opener, _ = self._recall(datasets=None)
        self.assertNotIn("datasets", opener.json_body("/api/v1/recall"))

        opener, _ = self._recall(session_id=None)
        self.assertNotIn("session_id", opener.json_body("/api/v1/recall"))

    def test_query_type_is_sent_as_search_type_uppercased(self):
        opener, _ = self._recall(query_type="chunks")
        body = opener.json_body("/api/v1/recall")
        self.assertEqual(body["search_type"], "CHUNKS")
        self.assertNotIn("query_type", body)

    def test_absent_query_type_is_sent_as_an_explicit_null(self):
        # Not omitted: a missing search_type defaults to GRAPH_COMPLETION
        # server-side, which both disables auto-routing and drops the session
        # cache out of the server's source resolution.
        body = self._recall()[0].json_body("/api/v1/recall")
        self.assertIn("search_type", body)
        self.assertIsNone(body["search_type"])

    def test_api_key_header_is_sent(self):
        opener, _ = self._recall()
        self.assertEqual(opener.request_for("/api/v1/recall")["headers"]["x-api-key"], "k")

    def test_results_are_returned_as_a_list(self):
        _, results = self._recall()
        self.assertEqual(results, [{"text": "hit"}])

    def test_a_single_object_response_is_wrapped(self):
        opener = FakeOpener({"/api/v1/recall": {"text": "one"}})
        results = _backend(opener).recall(
            query="q",
            session_id=None,
            datasets=None,
            top_k=1,
            auto_route=True,
            query_type=None,
            timeout=_TIMEOUT,
        )
        self.assertEqual(results, [{"text": "one"}])

    def test_auto_route_false_becomes_an_explicit_graph_completion(self):
        # There is no auto_route field, but the setting is expressible: server-side
        # auto_route=False with no type means "skip the classifier, use
        # GRAPH_COMPLETION", and naming that type bypasses the classifier too.
        opener, results = self._recall(auto_route=False)
        body = opener.json_body("/api/v1/recall")
        self.assertEqual(body["search_type"], "GRAPH_COMPLETION")
        self.assertNotIn("auto_route", body)
        self.assertEqual(results, [{"text": "hit"}])

    def test_auto_route_false_does_not_override_an_explicit_search_type(self):
        body = self._recall(auto_route=False, query_type="CHUNKS")[0].json_body("/api/v1/recall")
        self.assertEqual(body["search_type"], "CHUNKS")

    def test_auto_route_true_leaves_the_classifier_to_the_server(self):
        # An explicit null is the *only* way to reach the classifier: the
        # endpoint's search_type defaults to GRAPH_COMPLETION for backward
        # compatibility, so omitting the key made auto_route=True and
        # auto_route=False issue byte-identical requests.
        body = self._recall(auto_route=True)[0].json_body("/api/v1/recall")
        self.assertIn("search_type", body)
        self.assertIsNone(body["search_type"])

    def test_auto_route_true_and_false_no_longer_produce_the_same_request(self):
        on = self._recall(auto_route=True)[0].json_body("/api/v1/recall")
        off = self._recall(auto_route=False)[0].json_body("/api/v1/recall")
        self.assertNotEqual(on["search_type"], off["search_type"])

    def test_scope_is_stated_outright(self):
        for scope in (["graph"], ["code"]):
            body = self._recall(scope=scope)[0].json_body("/api/v1/recall")
            self.assertEqual(body["scope"], scope)

    def test_scope_survives_an_explicit_search_type(self):
        # The server infers sources from session_id/datasets only while
        # search_type is null; the stated scope keeps the request meaning the
        # same whether or not a search strategy is pinned.
        body = self._recall(scope=["graph"], auto_route=False)[0].json_body("/api/v1/recall")
        self.assertEqual(body["scope"], ["graph"])
        self.assertEqual(body["search_type"], "GRAPH_COMPLETION")

    def test_absent_scope_is_stated_as_the_graph(self):
        # Left out, the server would resolve the scope to ``auto`` and fold the
        # session cache in alongside the session id; memory reads the graph only.
        body = self._recall(scope=None)[0].json_body("/api/v1/recall")
        self.assertEqual(body["scope"], ["graph"])


class TestRememberWireFormat(unittest.TestCase):
    def test_a_turn_is_a_typed_qa_entry_not_a_document_upload(self):
        # Live-diagnosed: /api/v1/remember takes its payload as a multipart file,
        # which the server coerces to "[UploadFile]" and then *skips* for the
        # session cache — reporting status "session_stored" while storing nothing.
        # Turns must go through the typed-entry endpoint.
        opener = FakeOpener({"/api/v1/remember/entry": {"status": "session_stored"}})
        result = _backend(opener).remember_session(
            text="User: hi\nAssistant: hello",
            session_id="hermes_s1",
            dataset="hermes",
            timeout=_TIMEOUT,
        )
        body = opener.json_body("/api/v1/remember/entry")
        self.assertEqual(body["dataset_name"], "hermes")
        self.assertEqual(body["session_id"], "hermes_s1")
        self.assertEqual(body["entry"]["type"], "qa")
        self.assertEqual(body["entry"]["answer"], "User: hi\nAssistant: hello")
        self.assertEqual(result.status, "session_stored")

    def test_a_turn_does_not_touch_the_document_endpoint(self):
        opener = FakeOpener({"/api/v1/remember/entry": {}})
        _backend(opener).remember_session(text="t", session_id="s", dataset="d", timeout=_TIMEOUT)
        self.assertEqual(opener.paths(), ["/api/v1/remember/entry"])

    def test_permanent_write_omits_the_session_id(self):
        # No session_id makes the server do a direct add + cognify.
        opener = FakeOpener({"/api/v1/remember": {"status": "completed"}})
        _backend(opener).remember_permanent(
            text="a fact", dataset="hermes", session_ids=[], timeout=_TIMEOUT
        )
        fields = opener.multipart_fields("/api/v1/remember")
        self.assertNotIn("session_id", fields)
        self.assertEqual(fields["data"], "a fact")

    def test_permanent_write_warns_that_session_ids_cannot_be_sent(self):
        opener = FakeOpener({"/api/v1/remember": {"status": "completed"}})
        with self.assertLogs("cognee_integration_hermes.http_backend", level="WARNING") as logs:
            _backend(opener).remember_permanent(
                text="a fact", dataset="hermes", session_ids=["hermes_s1"], timeout=_TIMEOUT
            )
        self.assertIn("session_ids", "\n".join(logs.output))
        self.assertNotIn("session_ids", opener.multipart_fields("/api/v1/remember"))

    def test_a_permanent_write_is_multipart(self):
        opener = FakeOpener({"/api/v1/remember": {}})
        _backend(opener).remember_permanent(text="t", dataset="d", session_ids=[], timeout=_TIMEOUT)
        content_type = opener.request_for("/api/v1/remember")["headers"]["content-type"]
        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))

    def test_permanent_write_runs_in_the_background(self):
        # A synchronous remember runs the full cognify (several LLM calls)
        # before answering; on a slow or rate-limited LLM endpoint that exceeds
        # the write timeout and the caller sees a failure for a write the server
        # actually completes. The request must ask the server to background it.
        opener = FakeOpener({"/api/v1/remember": {"status": "running"}})
        result = _backend(opener).remember_permanent(
            text="a fact", dataset="hermes", session_ids=[], timeout=_TIMEOUT
        )
        self.assertEqual(opener.multipart_fields("/api/v1/remember")["run_in_background"], "true")
        self.assertEqual(result.status, "running")

    def test_result_exposes_status_as_an_attribute(self):
        # The provider reads getattr(result, "status", "completed"); a bare dict
        # would silently report "completed" for every write.
        opener = FakeOpener({"/api/v1/remember": {"status": "running"}})
        result = _backend(opener).remember_permanent(
            text="t", dataset="d", session_ids=[], timeout=_TIMEOUT
        )
        self.assertEqual(result.status, "running")

    def test_missing_status_defaults_to_completed(self):
        opener = FakeOpener({"/api/v1/remember": {}})
        result = _backend(opener).remember_permanent(
            text="t", dataset="d", session_ids=[], timeout=_TIMEOUT
        )
        self.assertEqual(result.status, "completed")

    def _remember_filename(self, text):
        opener = FakeOpener({"/api/v1/remember": {"status": "completed"}})
        _backend(opener).remember_permanent(
            text=text, dataset="hermes", session_ids=[], timeout=_TIMEOUT
        )
        raw = opener.request_for("/api/v1/remember")["body"].decode("utf-8", "replace")
        return raw.split('filename="', 1)[1].split('"', 1)[0]

    def test_the_upload_name_follows_the_content(self):
        # The part's filename is the document's identity server-side, and cognee
        # >= 1.6.0 answers 409 to an add() whose name already exists in the
        # dataset with different content: one fixed name lost every memory after
        # the first. Different content must get a different name, and identical
        # content must keep the same one — re-adding it is a server-side no-op.
        first = self._remember_filename("a fact")
        self.assertEqual(self._remember_filename("a fact"), first)
        self.assertNotEqual(self._remember_filename("another fact"), first)
        # 32 hex chars (128 bits), the same length every other integration uses.
        self.assertRegex(first, r"^memory-[0-9a-f]{32}\.txt$")


class TestImproveWireFormat(unittest.TestCase):
    def test_session_ids_reach_the_wire(self):
        """The regression this transport exists to fix.

        ``cognee.improve(session_ids=[...])`` through the SDK's CloudClient never
        sends them (improve.py:128 forwards only node_name/**kwargs), so the
        session-to-graph bridge silently became a dataset-wide improve.
        """
        opener = FakeOpener({"/api/v1/improve": {"status": "started"}})
        _backend(opener).improve(
            dataset="hermes", session_ids=["hermes_s1"], background=True, timeout=_TIMEOUT
        )
        body = opener.json_body("/api/v1/improve")
        self.assertEqual(body["session_ids"], ["hermes_s1"])
        self.assertEqual(body["dataset_name"], "hermes")
        self.assertIs(body["run_in_background"], True)

    def test_background_false_is_sent_explicitly(self):
        opener = FakeOpener({"/api/v1/improve": {}})
        _backend(opener).improve(
            dataset="hermes", session_ids=["s"], background=False, timeout=_TIMEOUT
        )
        self.assertIs(opener.json_body("/api/v1/improve")["run_in_background"], False)

    def test_empty_session_ids_are_omitted(self):
        opener = FakeOpener({"/api/v1/improve": {}})
        _backend(opener).improve(
            dataset="hermes", session_ids=[], background=True, timeout=_TIMEOUT
        )
        self.assertNotIn("session_ids", opener.json_body("/api/v1/improve"))


class TestForgetWireFormat(unittest.TestCase):
    def _forget(self, **overrides):
        params = {
            "dataset": "hermes",
            "everything": False,
            "memory_only": False,
            "timeout": _TIMEOUT,
        }
        params.update(overrides)
        opener = FakeOpener({"/api/v1/forget": {"deleted": True}})
        _backend(opener).forget(**params)
        return opener.json_body("/api/v1/forget")

    def test_dataset_scoped_delete(self):
        body = self._forget()
        self.assertEqual(body["dataset"], "hermes")
        self.assertIs(body["everything"], False)
        self.assertIs(body["memory_only"], False)

    def test_everything_drops_the_dataset(self):
        body = self._forget(everything=True)
        self.assertIs(body["everything"], True)
        self.assertNotIn("dataset", body)

    def test_memory_only_passes_through(self):
        self.assertIs(self._forget(memory_only=True)["memory_only"], True)


class TestIndexRepositoryWireFormat(unittest.TestCase):
    """The code-graph submission, whose field name fails silently when wrong.

    cognee 1.5.4 renamed the repo-spec form field from ``repositories`` to
    ``raw_data``. An unrecognised multipart part is dropped by the server, not
    refused, so the old name reached a 1.5.4 server as a request carrying no
    repository at all and 400'd on every repo (issue #420).
    """

    def _index(self, **overrides):
        params = {
            "repo": "/srv/proj",
            "dataset": "codebase-proj",
            "timeout": _TIMEOUT,
        }
        params.update(overrides)
        opener = FakeOpener({"/api/v1/remember": {"status": "running"}})
        _backend(opener).index_repository(**params)
        return opener.multipart_fields("/api/v1/remember")

    def test_the_repo_spec_is_sent_as_raw_data(self):
        fields = self._index()
        self.assertEqual(fields["raw_data"], "/srv/proj")
        self.assertNotIn("repositories", fields)

    def test_content_type_and_dataset_accompany_it(self):
        fields = self._index()
        self.assertEqual(fields["content_type"], "code")
        self.assertEqual(fields["datasetName"], "codebase-proj")

    def test_index_vectors_is_off_unless_asked(self):
        self.assertEqual(self._index()["index_vectors"], "false")
        self.assertEqual(self._index(index_vectors=True)["index_vectors"], "true")

    def _reject(self, detail):
        opener = FakeOpener(
            {
                "/api/v1/remember": urllib.error.HTTPError(
                    _URL, 400, detail, {}, io.BytesIO(detail.encode("utf-8"))
                )
            }
        )
        with self.assertRaises(CogneeHttpError) as caught:
            _backend(opener).index_repository(repo="/srv/proj", dataset="ds", timeout=_TIMEOUT)
        return str(caught.exception)

    def test_an_unsupported_content_type_is_reported_as_an_old_server(self):
        self.assertIn("1.5.4", self._reject("Unsupported content_type 'code'."))

    def test_other_400s_keep_the_servers_own_message(self):
        """The second half of issue #420: every 400 the code branch raises names
        `content_type`, so classifying on that substring relabelled the server's
        actionable complaint as "your server is too old" and sent the reporter
        chasing a version that was already new enough."""
        detail = (
            "content_type='code' requires at least one repository path or git URL in 'raw_data'."
        )
        message = self._reject(detail)
        self.assertNotIn("requires cognee >=", message)
        self.assertIn(detail, message)


class TestConnect(unittest.TestCase):
    def _opener(self, **overrides):
        responses = {
            "/health": {"status": "ok"},
            "/api/v1/agents/register": {"id": "conn-1"},
        }
        responses.update(overrides)
        return FakeOpener(responses)

    def test_health_check_then_register(self):
        opener = self._opener()
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)
        self.assertEqual(opener.paths()[0], "/health")
        self.assertIn("/api/v1/agents/register", opener.paths())
        self.assertTrue(backend.registered)

    def test_unreachable_server_raises(self):
        # cognee.serve() only logged a warning here and handed back a client, so a
        # bad COGNEE_BASE_URL looked like success until the first real call.
        opener = self._opener(**{"/health": urllib.error.URLError("refused")})
        backend = HttpBackend(opener=opener)
        with self.assertRaises(CogneeUnreachable):
            backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)

    def test_unhealthy_server_raises(self):
        opener = self._opener(**{"/health": urllib.error.HTTPError(_URL, 503, "down", {}, None)})
        backend = HttpBackend(opener=opener)
        with self.assertRaises(CogneeHttpError):
            backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)

    def test_registration_failure_is_not_fatal(self):
        # Registration only drives the server's idle-shutdown watchdog.
        opener = self._opener(
            **{"/api/v1/agents/register": urllib.error.HTTPError(_URL, 404, "nope", {}, None)}
        )
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)
        self.assertFalse(backend.registered)

    def test_trailing_slash_is_normalized(self):
        opener = self._opener()
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL + "/", api_key="k", timeout=_TIMEOUT)
        self.assertEqual(backend.url, _URL)

    def test_close_unregisters_only_when_registered(self):
        opener = self._opener(**{"/api/v1/agents/unregister": {"active_agents": 0}})
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)
        backend.close(timeout=_TIMEOUT)
        self.assertIn("/api/v1/agents/unregister", opener.paths())
        self.assertFalse(backend.registered)

        fresh = FakeOpener()
        HttpBackend(opener=fresh).close(timeout=_TIMEOUT)
        self.assertEqual(fresh.paths(), [])

    def test_close_without_unregister_stays_registered_and_sends_nothing(self):
        # A detached worker owns the unregister and is still improving. Dropping
        # the agent count to zero here would let the server's idle watchdog
        # retire it mid-promotion.
        opener = self._opener(**{"/api/v1/agents/unregister": {"active_agents": 0}})
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL, api_key="k", timeout=_TIMEOUT)
        before = opener.paths()
        backend.close(timeout=_TIMEOUT, unregister=False)
        self.assertEqual(opener.paths(), before)
        self.assertTrue(backend.registered)


class TestApiKeyResolution(unittest.TestCase):
    def _mint_opener(self):
        return FakeOpener(
            {
                "/health": {"status": "ok"},
                "/api/v1/auth/login": {"access_token": "jwt-1"},
                "/api/v1/auth/api-keys": {"key": "minted-key"},
                "/api/v1/agents/register": {},
            }
        )

    def _connect(self, opener, *, api_key="", url=_URL):
        backend = HttpBackend(opener=opener)
        with mock.patch("pathlib.Path.read_text", side_effect=AssertionError("read a file")):
            backend.connect(url=url, api_key=api_key, timeout=_TIMEOUT)
        return backend

    def test_a_configured_key_is_used_as_is(self):
        opener = self._mint_opener()
        backend = self._connect(opener, api_key="given")
        self.assertEqual(backend.api_key, "given")
        self.assertNotIn("/api/v1/auth/login", opener.paths())

    def test_a_key_is_minted_when_none_is_configured(self):
        opener = self._mint_opener()
        backend = self._connect(opener)
        self.assertEqual(backend.api_key, "minted-key")
        login = opener.request_for("/api/v1/auth/login")
        self.assertEqual(login["headers"]["content-type"], "application/x-www-form-urlencoded")
        self.assertIn(b"username=", login["body"])

    def test_minting_reuses_an_existing_key_when_the_server_lists_one(self):
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/api-keys"] = [{"key": "existing-key"}]
        backend = self._connect(opener)
        self.assertEqual(backend.api_key, "existing-key")

    def test_every_session_looks_the_key_up_on_the_server(self):
        # No key file: a new process logs in again and reuses the server's key.
        import cognee_integration_hermes.http_backend as hb

        first = self._mint_opener()
        first.responses["/api/v1/auth/api-keys"] = [{"key": "claude-minted"}]
        self.assertEqual(self._connect(first).api_key, "claude-minted")
        hb._RESOLVED_KEYS.clear()  # a new Hermes process
        second = self._mint_opener()
        second.responses["/api/v1/auth/api-keys"] = [{"key": "claude-minted"}]
        self.assertEqual(self._connect(second).api_key, "claude-minted")
        self.assertIn("/api/v1/auth/login", second.paths())

    def test_a_second_connect_in_one_process_skips_the_login(self):
        self.assertEqual(self._connect(self._mint_opener()).api_key, "minted-key")
        again = self._mint_opener()
        self.assertEqual(self._connect(again).api_key, "minted-key")
        self.assertNotIn("/api/v1/auth/login", again.paths())

    def test_the_remembered_key_is_per_server(self):
        self._connect(self._mint_opener())
        other = self._mint_opener()
        self._connect(other, url="http://127.0.0.1:9999")
        self.assertIn("/api/v1/auth/login", other.paths())

    def _connect_as(self, scope, opener):
        import cognee_integration_hermes.config as config_mod

        with mock.patch.object(config_mod, "_hermes_get_secret", scope.get):
            return self._connect(opener)

    def test_two_profiles_with_different_users_get_their_own_keys(self):
        a = self._mint_opener()
        a.responses["/api/v1/auth/api-keys"] = [{"key": "key-of-a"}]
        b = self._mint_opener()
        b.responses["/api/v1/auth/api-keys"] = [{"key": "key-of-b"}]
        profile_a = {"COGNEE_USER_EMAIL": "a@x", "COGNEE_USER_PASSWORD": "pa"}
        profile_b = {"COGNEE_USER_EMAIL": "b@x", "COGNEE_USER_PASSWORD": "pb"}
        self.assertEqual(self._connect_as(profile_a, a).api_key, "key-of-a")
        self.assertEqual(self._connect_as(profile_b, b).api_key, "key-of-b")
        self.assertIn(b"username=b%40x", b.request_for("/api/v1/auth/login")["body"])

    def test_a_wrong_password_does_not_reuse_the_users_key(self):
        good = {"COGNEE_USER_EMAIL": "a@x", "COGNEE_USER_PASSWORD": "right"}
        bad = {"COGNEE_USER_EMAIL": "a@x", "COGNEE_USER_PASSWORD": "wrong"}
        self.assertEqual(self._connect_as(good, self._mint_opener()).api_key, "minted-key")
        rejected = self._mint_opener()
        rejected.responses["/api/v1/auth/login"] = urllib.error.HTTPError(
            _URL, 404, "bad login", {}, None
        )
        self.assertEqual(self._connect_as(bad, rejected).api_key, "")
        self.assertIn("/api/v1/auth/login", rejected.paths())

    def test_the_password_is_not_kept_in_the_cache(self):
        import cognee_integration_hermes.http_backend as hb

        self._connect_as({"COGNEE_USER_PASSWORD": "s3cret"}, self._mint_opener())
        self.assertTrue(hb._RESOLVED_KEYS)
        for cache_key in hb._RESOLVED_KEYS:
            self.assertNotIn("s3cret", cache_key)

    def test_a_failed_lookup_is_not_remembered(self):
        failing = self._mint_opener()
        failing.responses["/api/v1/auth/login"] = urllib.error.HTTPError(
            _URL, 404, "no auth", {}, None
        )
        self.assertEqual(self._connect(failing).api_key, "")
        retry = self._mint_opener()
        self.assertEqual(self._connect(retry).api_key, "minted-key")

    def test_the_login_and_key_lookup_have_a_short_timeout(self):
        opener = self._mint_opener()
        backend = HttpBackend(opener=opener)
        backend.connect(url=_URL, api_key="", timeout=30)
        auth = [r for r in opener.requests if r["path"].startswith("/api/v1/auth/")]
        self.assertEqual(len(auth), 3)  # login, list keys, create the key
        for request in auth:
            self.assertLessEqual(request["timeout"], 10.0, request["path"])

    def test_no_key_file_is_read_or_written(self):
        import cognee_integration_hermes.http_backend as hb

        source = Path(hb.__file__).read_text(encoding="utf-8")
        self.assertNotIn("api_key.json", source)
        with mock.patch("pathlib.Path.write_text", side_effect=AssertionError("wrote a file")):
            self.assertEqual(self._connect(self._mint_opener()).api_key, "minted-key")

    def test_failed_minting_proceeds_without_a_key(self):
        # A LOCAL server with authentication disabled needs no key at all.
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/login"] = urllib.error.HTTPError(
            _URL, 404, "no auth", {}, None
        )
        backend = self._connect(opener)
        self.assertEqual(backend.api_key, "")

    @staticmethod
    def _login_400(detail):
        # cognee 1.6.0's login answers HTTP 400 with a JSON ``detail``; the fake
        # needs a readable body for _request to pick that detail up.
        body = io.BytesIO(json.dumps({"detail": detail}).encode("utf-8"))
        return urllib.error.HTTPError(_URL, 400, "Bad Request", {}, body)

    def test_a_passwordless_default_user_names_the_fix(self):
        # cognee >= 1.6.0 creates the default user without a password unless the
        # server was started with DEFAULT_USER_PASSWORD. Connect still succeeds
        # (the server may not require a key) but the warning must be actionable.
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/login"] = self._login_400(
            "This user does not have a password. Use API key authentication."
        )
        with self.assertLogs("cognee_integration_hermes.http_backend", level="WARNING") as logs:
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "")
        text = "\n".join(logs.output)
        self.assertIn("DEFAULT_USER_PASSWORD", text)
        self.assertIn("COGNEE_USER_PASSWORD", text)
        self.assertIn("COGNEE_API_KEY", text)
        self.assertIn("1.6.0", text)

    def test_the_passwordless_hint_reaches_the_first_401(self):
        # The provider surfaces failures by stringifying the exception, so the
        # 401 a keyless call gets must carry the hint — that is the user-visible path.
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/login"] = self._login_400(
            "This user does not have a password. Use API key authentication."
        )
        with self.assertLogs("cognee_integration_hermes.http_backend", level="WARNING"):
            backend = self._connect(opener)
        opener.responses["/api/v1/recall"] = urllib.error.HTTPError(
            _URL, 401, "Unauthorized", {}, io.BytesIO(b'{"detail": "Unauthorized"}')
        )
        with self.assertRaises(CogneeHttpError) as ctx:
            backend.recall(
                query="q",
                session_id=None,
                datasets=None,
                top_k=5,
                auto_route=True,
                query_type=None,
                timeout=_TIMEOUT,
            )
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("DEFAULT_USER_PASSWORD", str(ctx.exception))
        self.assertIn("COGNEE_API_KEY", str(ctx.exception))

    def test_bad_credentials_name_the_env_vars(self):
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/login"] = self._login_400("LOGIN_BAD_CREDENTIALS")
        with self.assertLogs("cognee_integration_hermes.http_backend", level="WARNING") as logs:
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "")
        text = "\n".join(logs.output)
        self.assertIn("LOGIN_BAD_CREDENTIALS", text)
        self.assertIn("COGNEE_USER_EMAIL", text)
        self.assertIn("COGNEE_USER_PASSWORD", text)
        self.assertIn("COGNEE_API_KEY", text)

    def test_an_undiagnosed_login_failure_stays_quiet(self):
        # A 404 (auth disabled) or an unrecognised 400 is the pre-existing
        # "proceed without a key" case: debug-level only, no warning, no hint.
        opener = self._mint_opener()
        opener.responses["/api/v1/auth/login"] = self._login_400("something else")
        with self.assertNoLogs("cognee_integration_hermes.http_backend", level="WARNING"):
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "")
        self.assertEqual(backend._auth_hint, "")

    def test_a_remote_target_without_a_key_fails_at_connect(self):
        # Cognee Cloud exposes no login route to mint from; continuing without a
        # key would smear one clear startup error into a 401 on every call.
        opener = self._mint_opener()
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "COGNEE_API_KEY"):
                self._connect(opener, url=_REMOTE_URL)
        # And it never tried the default-user login against a remote host.
        self.assertNotIn("/api/v1/auth/login", opener.paths())

    def test_a_remote_target_with_an_env_key_connects(self):
        opener = self._mint_opener()
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": "cloud-key"}, clear=False):
            backend = self._connect(opener, url=_REMOTE_URL)
        self.assertEqual(backend.api_key, "cloud-key")
        self.assertNotIn("/api/v1/auth/login", opener.paths())

    def test_a_local_target_skips_an_env_key_it_rejects(self):
        # Switching from Cloud to local mode clears COGNEE_BASE_URL, not the
        # Cloud key; sent to the local server it turned every call into a 401.
        opener = self._mint_opener()
        opener.responses["/api/v1/datasets"] = urllib.error.HTTPError(
            _URL, 401, "Unauthorized", {}, None
        )
        import cognee_integration_hermes.http_backend as hb

        hb._RESOLVED_KEYS.clear()
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": "cloud-key"}, clear=False):
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "minted-key")
        probe = opener.request_for("/api/v1/datasets")
        self.assertEqual(probe["headers"]["x-api-key"], "cloud-key")
        self.assertIn("/api/v1/auth/login", opener.paths())

    def test_a_local_target_keeps_an_env_key_it_accepts(self):
        opener = self._mint_opener()
        opener.responses["/api/v1/datasets"] = []
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": "local-key"}, clear=False):
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "local-key")
        self.assertNotIn("/api/v1/auth/login", opener.paths())

    def test_a_local_target_keeps_an_env_key_when_the_check_cannot_tell(self):
        # A slow or erroring server is not a rejected key.
        opener = self._mint_opener()
        opener.responses["/api/v1/datasets"] = urllib.error.HTTPError(_URL, 500, "boom", {}, None)
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": "local-key"}, clear=False):
            backend = self._connect(opener)
        self.assertEqual(backend.api_key, "local-key")

    def test_a_remote_target_without_a_configured_key_still_fails(self):
        # A key left behind by another plugin no longer stands in for COGNEE_API_KEY.
        with mock.patch.dict("os.environ", {"COGNEE_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError):
                self._connect(self._mint_opener(), url=_REMOTE_URL)


class TestEnsureDatasetWireFormat(unittest.TestCase):
    def test_posts_the_dataset_name(self):
        opener = FakeOpener({"/api/v1/datasets": {"id": "d1"}})
        _backend(opener).ensure_dataset(dataset="agent_sessions", timeout=_TIMEOUT)
        request = opener.request_for("/api/v1/datasets")
        self.assertEqual(request["method"], "POST")
        self.assertEqual(opener.json_body("/api/v1/datasets"), {"name": "agent_sessions"})
        self.assertEqual(request["headers"]["x-api-key"], "k")

    def test_a_blank_dataset_sends_nothing(self):
        opener = FakeOpener()
        _backend(opener).ensure_dataset(dataset="", timeout=_TIMEOUT)
        self.assertEqual(opener.paths(), [])

    def test_a_redirecting_deployment_is_retried_at_the_slashed_path(self):
        # Some deployments route the collection at /datasets/ and answer the
        # non-slash POST with a 307, which urllib will not follow for a body.
        opener = FakeOpener(
            {
                "/api/v1/datasets": urllib.error.HTTPError(_URL, 307, "moved", {}, None),
                "/api/v1/datasets/": {"id": "d1"},
            }
        )
        _backend(opener).ensure_dataset(dataset="agent_sessions", timeout=_TIMEOUT)
        self.assertEqual(opener.json_body("/api/v1/datasets/"), {"name": "agent_sessions"})

    def test_a_genuine_error_raises(self):
        opener = FakeOpener(
            {"/api/v1/datasets": urllib.error.HTTPError(_URL, 500, "boom", {}, None)}
        )
        with self.assertRaises(CogneeHttpError):
            _backend(opener).ensure_dataset(dataset="agent_sessions", timeout=_TIMEOUT)


class TestConnectionInfo(unittest.TestCase):
    def test_none_before_connect(self):
        self.assertIsNone(HttpBackend().connection_info())

    def test_reports_the_server_details_after_connect(self):
        backend = _backend(FakeOpener(), api_key="k")
        self.assertEqual(
            backend.connection_info(),
            {"url": _URL, "api_key": "k", "agent_session_name": "hermes"},
        )


class TestErrorMapping(unittest.TestCase):
    def test_http_error_carries_the_status(self):
        opener = FakeOpener({"/api/v1/recall": urllib.error.HTTPError(_URL, 401, "nope", {}, None)})
        with self.assertRaises(CogneeHttpError) as caught:
            _backend(opener).recall(
                query="q",
                session_id=None,
                datasets=None,
                top_k=1,
                auto_route=True,
                query_type=None,
                timeout=_TIMEOUT,
            )
        self.assertEqual(caught.exception.status, 401)

    def test_transport_failure_is_unreachable(self):
        opener = FakeOpener({"/api/v1/recall": urllib.error.URLError("refused")})
        with self.assertRaises(CogneeUnreachable):
            _backend(opener).recall(
                query="q",
                session_id=None,
                datasets=None,
                top_k=1,
                auto_route=True,
                query_type=None,
                timeout=_TIMEOUT,
            )

    def test_an_error_shaped_2xx_body_is_an_error(self):
        opener = FakeOpener({"/api/v1/recall": {"error": "prerequisites not met"}})
        with self.assertRaises(CogneeHttpError):
            _backend(opener).recall(
                query="q",
                session_id=None,
                datasets=None,
                top_k=1,
                auto_route=True,
                query_type=None,
                timeout=_TIMEOUT,
            )

    def test_malformed_json_is_a_server_error_not_unreachability(self):
        opener = FakeOpener({"/api/v1/forget": "{not json"})
        with self.assertRaises(CogneeHttpError):
            _backend(opener).forget(
                dataset="d", everything=False, memory_only=False, timeout=_TIMEOUT
            )


class TestEmptyRecallHint(unittest.TestCase):
    def test_there_is_no_hint_over_http(self):
        # The server owns the vector store; the probe is in-process only.
        self.assertIsNone(HttpBackend(opener=FakeOpener()).empty_recall_hint())


class TestProviderOverHttp(unittest.TestCase):
    """The full stack: real provider, real HttpBackend, fake socket.

    The provider tests pin behaviour against the protocol and the tests above pin
    the protocol against the wire. This class is the join — the first place the two
    layers run together, and the cheapest way to catch a mismatch that both halves
    consider correct in isolation.
    """

    def _provider(self, opener, **kwargs):
        return make_provider(backend=_backend(opener), **kwargs)

    def test_recall_produces_the_model_visible_envelope(self):
        opener = FakeOpener({"/api/v1/recall": [{"text": "alpha"}, {"text": "beta"}]})
        provider = self._provider(opener)
        out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "q"}))
        self.assertEqual(out["count"], 2)
        self.assertEqual([item["text"] for item in out["results"]], ["alpha", "beta"])

    def test_recall_miss_produces_the_miss_envelope(self):
        opener = FakeOpener({"/api/v1/recall": []})
        out = json.loads(self._provider(opener).handle_tool_call("cognee_recall", {"query": "q"}))
        self.assertEqual(out, {"result": "No relevant Cognee memory found.", "count": 0})

    def test_graph_scope_survives_the_whole_stack(self):
        opener = FakeOpener({"/api/v1/recall": []})
        provider = self._provider(opener, session_cognee_id="hermes_sX")
        # A legacy ``scope`` argument is ignored on the way through.
        provider.handle_tool_call("cognee_recall", {"query": "q", "scope": "session"})
        body = opener.json_body("/api/v1/recall")
        self.assertEqual(body["scope"], ["graph"])
        # Auto recall (SDK-1393): unpinned, the body names no dataset.
        self.assertNotIn("datasets", body)
        self.assertNotIn("session_id", body)

    def test_remember_reports_the_server_status(self):
        # The trap: an HTTP dict has no .status attribute, so a bare dict here
        # would make every write report "completed".
        opener = FakeOpener({"/api/v1/remember": {"status": "running"}})
        out = json.loads(
            self._provider(opener).handle_tool_call("cognee_remember", {"content": "fact"})
        )
        self.assertEqual(out, {"result": "Content stored in Cognee.", "status": "running"})

    def test_a_turn_goes_to_the_session_cache(self):
        opener = FakeOpener({"/api/v1/remember/entry": {"status": "session_stored"}})
        provider = self._provider(opener, session_cognee_id="hermes_sY")
        provider.sync_turn("hello", "hi there")
        provider._sync_thread.join(timeout=2.0)
        body = opener.json_body("/api/v1/remember/entry")
        self.assertEqual(body["session_id"], "hermes_sY")
        # The prompt and the answer in their own fields, as the other plugins
        # send them: recaps list the prompts and the timeline searches them.
        self.assertEqual(body["entry"]["type"], "qa")
        self.assertEqual(body["entry"]["question"], "hello")
        self.assertEqual(body["entry"]["answer"], "hi there")

    def test_session_end_bridges_the_session_into_the_graph(self):
        """End-to-end proof that finding #2 is fixed."""
        opener = FakeOpener({"/api/v1/improve": {"status": "started"}})
        provider = self._provider(opener, session_cognee_id="hermes_sZ")
        provider.on_session_end([])
        body = opener.json_body("/api/v1/improve")
        self.assertEqual(body["session_ids"], ["hermes_sZ"])
        self.assertEqual(body["dataset_name"], "hermes")

    def test_a_backend_failure_becomes_the_error_envelope_and_trips_the_breaker(self):
        opener = FakeOpener({"/api/v1/recall": urllib.error.URLError("refused")})
        provider = self._provider(opener)
        for _ in range(5):
            out = json.loads(provider.handle_tool_call("cognee_recall", {"query": "q"}))
            self.assertTrue(out["error"].startswith("Cognee recall failed:"))
        self.assertTrue(provider._is_breaker_open())

    def test_forget_reaches_the_endpoint_and_reports_details(self):
        dataset_id = "11111111-1111-1111-1111-111111111111"
        data_id = "22222222-2222-2222-2222-222222222222"
        opener = FakeOpener(
            {
                "/api/v1/datasets": [{"id": dataset_id, "name": "hermes"}],
                "/api/v1/forget": {"deleted": 1},
            }
        )
        out = json.loads(
            self._provider(opener).handle_tool_call(
                "cognee_forget",
                {"action": "forget", "data_ids": [data_id], "confirm": True},
            )
        )
        self.assertEqual(out["deleted"], [data_id])
        self.assertEqual(out["count"], 1)
        # The wire body is the single-document shape — the only deletion this
        # tool path can express.
        self.assertEqual(
            opener.json_body("/api/v1/forget"),
            {"datasetId": dataset_id, "dataId": data_id},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
