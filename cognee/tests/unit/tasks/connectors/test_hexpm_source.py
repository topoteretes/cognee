"""Hex.pm source: rendering, incremental skips, retirements and tombstones."""

import pytest

from cognee.tasks.ingestion.connectors import hexpm_source, parse_mix_lock
from cognee.tasks.ingestion.connectors.hexpm import _iter_rows
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR


class FakeHex:
    def __init__(self):
        self.packages = {}
        self.release_calls = []

    def package(self, name, etag=None):
        if name not in self.packages:
            return 404, None, None
        return 200, self.packages[name], None

    def release(self, name, version):
        self.release_calls.append((name, version))
        return (
            200,
            {
                "version": version,
                "requirements": {
                    "plug": {"app": "plug", "requirement": "~> 1.0", "optional": True}
                },
                "meta": {"elixir": "~> 1.14"},
                "retirement": self.packages[name]["retirements"].get(version),
            },
            None,
        )


def _pkg(versions, retirements=None, updated="t1"):
    return {
        "name": "phoenix",
        "updated_at": updated,
        "meta": {"description": "Web framework", "licenses": ["MIT"], "links": {}},
        "downloads": {"all": 5},
        "releases": [{"version": v} for v in versions],
        "retirements": retirements or {},
    }


def test_mix_lock_names():
    lock = '%{\n  "ecto": {:hex, :ecto, "3.0.0", "x", [:mix], [], "hexpm", "y"},\n  "foo": {:git, "u", "r", []},\n}'
    assert parse_mix_lock(lock) == ["ecto"]


def test_incremental_retirement_and_forget():
    fake, state = FakeHex(), {}
    fake.packages["phoenix"] = _pkg(["1.0.0", "1.1.0"])
    rows = list(_iter_rows(fake, state, ["phoenix"], True))
    assert {r["id"] for r in rows} == {
        "package:phoenix",
        "release:phoenix:1.0.0",
        "release:phoenix:1.1.0",
    }

    fake.release_calls.clear()
    assert list(_iter_rows(fake, state, ["phoenix"], True)) == []
    assert fake.release_calls == []

    fake.packages["phoenix"] = _pkg(
        ["1.0.0", "1.1.0"], {"1.0.0": {"reason": "security", "message": "CVE"}}, "t2"
    )
    rows = list(_iter_rows(fake, state, ["phoenix"], True))
    assert fake.release_calls == [("phoenix", "1.0.0")]
    assert any("RETIRED (security)" in r.get("content", "") for r in rows)

    fake.packages["phoenix"] = _pkg(["1.1.0"], {}, "t3")
    rows = list(_iter_rows(fake, state, ["phoenix"], True))
    assert {"id": "release:phoenix:1.0.0", "_deleted": True} in rows

    del fake.packages["phoenix"]
    rows = list(_iter_rows(fake, state, ["phoenix"], True))
    assert {r["id"] for r in rows} == {"package:phoenix", "release:phoenix:1.1.0"}
    assert all(r["_deleted"] for r in rows)


def test_resource_is_document_source_and_validates():
    resource = hexpm_source("hexpm_x", package_names=["phoenix"], service=FakeHex())
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "hexpm"
    with pytest.raises(ValueError):
        hexpm_source("hexpm_x")
    with pytest.raises(ValueError):
        hexpm_source("hexpm_x", package_names=["Bad Name"])
