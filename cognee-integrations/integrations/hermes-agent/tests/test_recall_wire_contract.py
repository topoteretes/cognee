"""The recall body, checked against cognee's own parser instead of our reading of it.

Every other test in this suite asserts the *shape* the plugin sends, using fakes,
so it never needs cognee installed. That is the right default, but it cannot
catch the failure this file exists for: a body that is well-formed, accepted with
a 2xx, and still means something other than what the plugin intended — because
the server supplies a default for a field the plugin left out.

Two such defaults matter here. ``/api/v1/recall`` once defaulted a *missing*
``search_type`` to ``GRAPH_COMPLETION`` (1.6.0 moved that default to null), and
an *unstated* scope resolves to ``auto``, which folds the session cache into the
sources whenever a session id travels with a null search type. Memory must read
the knowledge graph only — the session-cache scopes are noise — so the plugin
states ``scope=["graph"]`` on every request while still sending the session id
(on 1.6.0 that is what puts the conversation history into the graph item's
prompt). Left unstated, the very same body would read the session cache too.

So these tests parse the real ``RecallPayloadDTO`` and call the real
``normalize_scope``: the server's own code decides what the request means. They
are skipped when cognee is not installed.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognee_integration_hermes.http_backend import HttpBackend  # noqa: E402
from test_http_backend import FakeOpener  # noqa: E402

_HAS_COGNEE = importlib.util.find_spec("cognee") is not None


def _server_defaults_search_type_to_null() -> bool:
    """cognee >= 1.6.0 defaults an omitted ``search_type`` to null (auto-routing);
    1.4 pinned GRAPH_COMPLETION and 1.5.x HYBRID_COMPLETION."""
    if not _HAS_COGNEE:
        return False
    from importlib.metadata import version

    major, minor = (int(part) for part in version("cognee").split(".")[:2])
    return (major, minor) >= (1, 6)


_REASON = "install cognee to check the wire contract against its own parser"


def _sent_body(**overrides):
    """The JSON the transport actually puts on the wire for a recall."""
    params = {
        "query": "q",
        "session_id": "hermes_s1",
        "datasets": ["agent_sessions"],
        "top_k": 5,
        "auto_route": True,
        "query_type": None,
        "scope": ["graph"],
        "timeout": 5.0,
    }
    params.update(overrides)
    opener = FakeOpener({"/api/v1/recall": []})
    backend = HttpBackend(opener=opener)
    backend.url = "http://127.0.0.1:8011"
    backend.api_key = "k"
    backend.recall(**params)
    return opener.json_body("/api/v1/recall")


@unittest.skipUnless(_HAS_COGNEE, _REASON)
class TestRecallBodyMeansWhatWeIntend(unittest.TestCase):
    def _parse(self, body):
        from cognee.api.v1.recall.routers.get_recall_router import RecallPayloadDTO

        return RecallPayloadDTO(**body)

    def _sources(self, dto):
        """cognee's source resolution for an ``auto`` scope (recall.py).

        Mirrors the server's branch rather than importing it — it is inline in
        ``recall()`` — so the assertions below lean on the two *parsed* inputs it
        reads, which is where the bug actually lived.
        """
        from cognee.memory.entries import normalize_scope

        resolved = normalize_scope(dto.scope)
        if resolved != ["auto"]:
            return resolved
        if dto.session_id and dto.search_type is None:
            return ["session", "graph"]
        return ["graph"]

    def test_the_body_is_accepted_by_the_servers_own_dto(self):
        self._parse(_sent_body())

    def test_the_default_config_reaches_the_query_classifier(self):
        # search_type must survive parsing as None. The regression was that the
        # key never arrived, so the DTO default took over.
        self.assertIsNone(self._parse(_sent_body()).search_type)

    def test_omitting_search_type_is_what_broke_auto_routing(self):
        # Characterizes the old wire format: same request minus the key, parsed
        # by the same DTO. Before 1.6.0 that silently meant a *pinned* search
        # type instead of the query classifier (GRAPH_COMPLETION on 1.4,
        # HYBRID_COMPLETION on 1.5.x) — auto-routing and the session fold were
        # conditional on an explicit null. cognee 1.6.0 moved the DTO default to
        # null itself, so an omitted key now auto-routes; the plugin still sends
        # the explicit null (test above) so the meaning does not depend on the
        # server's version.
        body = _sent_body()
        body.pop("search_type")
        parsed = self._parse(body)
        if _server_defaults_search_type_to_null():
            self.assertIsNone(parsed.search_type)
        else:
            self.assertIsNotNone(parsed.search_type)

    def test_auto_route_false_still_pins_graph_completion(self):
        dto = self._parse(_sent_body(auto_route=False))
        self.assertEqual(str(dto.search_type), "SearchType.GRAPH_COMPLETION")

    def test_the_graph_scope_resolves_to_the_graph_alone(self):
        # The session id travels (it is what gives the 1.6.0 graph item its
        # history), yet the server never folds the session cache in: an explicit
        # graph scope is resolved before the session_id/search_type inference.
        dto = self._parse(_sent_body())
        self.assertEqual(dto.session_id, "hermes_s1")
        self.assertEqual(self._sources(dto), ["graph"])

    def test_the_default_transport_scope_is_the_graph_too(self):
        # A caller that names no scope gets the same request, not the server's
        # auto default.
        self.assertEqual(self._sources(self._parse(_sent_body(scope=None))), ["graph"])

    def test_the_code_scope_resolves_to_the_code_graph_alone(self):
        dto = self._parse(_sent_body(scope=["code"], code_query={"operation": "query_facts"}))
        self.assertEqual(self._sources(dto), ["code"])

    def test_the_graph_scope_survives_a_pinned_search_type(self):
        # COGNEE_AUTO_ROUTE=false pins GRAPH_COMPLETION; the sources still come
        # from the stated scope, not from the search type.
        dto = self._parse(_sent_body(auto_route=False))
        self.assertEqual(self._sources(dto), ["graph"])

    def test_an_unstated_scope_would_have_read_the_session_cache(self):
        # Why the scope is always stated: the same body minus the key — a
        # session id and a null search type — is exactly the shape the server
        # folds the session cache into on 1.6.0 (and, before 1.6.0, only while
        # the search type arrived as an explicit null).
        body = _sent_body()
        body.pop("scope")
        self.assertEqual(self._sources(self._parse(body)), ["session", "graph"])
        body.pop("search_type")
        expected = ["session", "graph"] if _server_defaults_search_type_to_null() else ["graph"]
        self.assertEqual(self._sources(self._parse(body)), expected)


if __name__ == "__main__":
    unittest.main()
