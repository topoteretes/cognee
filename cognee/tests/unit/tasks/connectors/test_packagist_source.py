"""The Packagist source fetches selected packages, then only those the change feed names."""

import pytest

from cognee.tasks.ingestion.connectors import packagist_source
from cognee.tasks.ingestion.connectors.packagist import _iter_rows, expand_versions
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR

NAME = "acme/lib"


class FakeClient:
    def __init__(self):
        self.feed = {"actions": [], "timestamp": 100}
        self.package_body = {
            "description": "A lib",
            "repository": "https://github.com/acme/lib",
            "maintainers": [{"name": "ada"}],
            "abandoned": "acme/new-lib",
        }
        self.advisories_body = {
            NAME: [
                {
                    "advisoryId": "PKSA-1",
                    "title": "XSS",
                    "cve": "CVE-1",
                    "affectedVersions": "<1.1",
                    "link": "https://x/1",
                }
            ]
        }
        self.gone = False

    def changes(self, since):
        return {"timestamp": 100} if since is None else self.feed

    def metadata(self, name):
        if self.gone:
            return None
        return {
            "packages": {
                name: [
                    {"version": "1.1", "time": "t2", "require": {"php": ">=8.1", "a/b": "^1"}},
                    {
                        "version": "1.0",
                        "time": "t1",
                        "require": "__unset",
                        "conflict": {"c/d": "*"},
                    },
                ]
            }
        }

    def package(self, name):
        return self.package_body

    def advisories(self, names):
        return self.advisories_body


def run(client, state):
    stats = {"scanned": 0, "skipped": 0, "synced": 0, "deleted": 0, "failed": 0}
    return list(_iter_rows(client, [NAME], state, stats)), stats


def test_expand_versions_unsets_keys():
    versions = expand_versions(FakeClient().metadata(NAME), NAME)
    assert versions[1]["version"] == "1.0"
    assert "require" not in versions[1]
    assert versions[1]["conflict"] == {"c/d": "*"}


def test_first_run_then_only_feed_updates():
    client, state = FakeClient(), {}
    rows, _ = run(client, state)
    ids = [r["id"] for r in rows]
    assert ids == ["package:acme/lib", "advisory:PKSA-1"]
    assert "replaced by acme/new-lib" in rows[0]["content"]
    assert "acme/lib" in rows[1]["content"]
    assert state["since"] == 100

    rows, stats = run(client, state)
    assert rows == [] and stats["scanned"] == 0

    client.feed = {"actions": [{"type": "update", "package": NAME + "~dev"}], "timestamp": 200}
    client.advisories_body = {NAME: []}
    rows, _ = run(client, state)
    assert [(r["id"], r["_deleted"]) for r in rows] == [("advisory:PKSA-1", True)]
    assert state["since"] == 200


def test_delete_action_tombstones_package_and_advisories():
    client, state = FakeClient(), {}
    run(client, state)
    client.feed = {"actions": [{"type": "delete", "package": NAME}], "timestamp": 300}
    rows, _ = run(client, state)
    assert sorted(r["id"] for r in rows if r["_deleted"]) == ["advisory:PKSA-1", "package:acme/lib"]


def test_source_requires_one_selection_and_user_agent():
    with pytest.raises(ValueError):
        packagist_source(user_agent="t", service=FakeClient())
    with pytest.raises(ValueError):
        packagist_source(user_agent="", packages=[NAME])
    resource = packagist_source(user_agent="t", packages=[NAME], service=FakeClient())
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "packagist"
