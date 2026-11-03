"""The NuGet source discovers services, renders registrations and forgets what disappeared."""

import gzip
import json

import httpx
import pytest

from cognee.tasks.ingestion.connectors import nuget_source
from cognee.tasks.ingestion.connectors.nuget import _iter_rows
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

BASE = "https://reg.example/v3/"
INDEX = "https://feed.example/v3/index.json"


def _entry(version, **extra):
    return {"version": version, "description": "Json lib", "authors": "James", **extra}


def _registration(commit, entries):
    return {
        "commitId": commit,
        "items": [
            {
                "@id": f"{BASE}newtonsoft.json/page/1.json",
                "commitId": commit,
                "items": [{"catalogEntry": e} for e in entries],
            }
        ],
    }


def _client(registrations):
    def handler(request):
        url = str(request.url)
        if url == INDEX:
            body = {
                "resources": [
                    {"@id": BASE, "@type": "RegistrationsBaseUrl/Versioned"},
                    {"@id": "https://s.example/q", "@type": "SearchQueryService"},
                ]
            }
        elif url.startswith(BASE):
            package = url.removeprefix(BASE).split("/")[0]
            if package not in registrations:
                return httpx.Response(404)
            # Registration blobs are gzipped on the wire.
            return httpx.Response(
                200,
                content=gzip.compress(json.dumps(registrations[package]).encode()),
                headers={"content-encoding": "gzip"},
            )
        else:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(client, state, ids=("Newtonsoft.Json",)):
    stats = {"synced": 0, "deleted": 0, "unchanged": 0}
    rows = list(_iter_rows(client, INDEX, list(ids), None, None, state, stats))
    return rows, stats


def test_renders_deprecation_dependencies_and_vulnerabilities():
    entry = _entry(
        "13.0.1",
        licenseExpression="MIT",
        dependencyGroups=[
            {"targetFramework": "net6.0", "dependencies": [{"id": "A", "range": "1"}]}
        ],
        deprecation={
            "reasons": ["Legacy"],
            "message": "Use the new one",
            "alternatePackage": {"id": "System.Text.Json", "range": "[6.0.0, )"},
        },
        vulnerabilities=[{"advisoryUrl": "https://github.com/advisories/x", "severity": "2"}],
    )
    rows, _ = _run(_client({"newtonsoft.json": _registration("c1", [entry])}), {})
    by_id = {r["id"]: r for r in rows}
    text = by_id["nuget:newtonsoft.json@13.0.1"]["content"]
    for needle in ("MIT", "net6.0", "System.Text.Json", "advisories/x", "Legacy"):
        assert needle in text
    assert by_id["nuget:newtonsoft.json"][NODE_SET_COLUMN] == ["nuget:newtonsoft.json"]


def test_unchanged_commit_is_skipped_and_removed_version_is_forgotten():
    state = {}
    reg = {"newtonsoft.json": _registration("c1", [_entry("1.0.0"), _entry("2.0.0")])}
    _run(_client(reg), state)
    rows, stats = _run(_client(reg), state)
    assert rows == [] and stats["unchanged"] == 1

    reg = {"newtonsoft.json": _registration("c2", [_entry("2.0.0")])}
    rows, _ = _run(_client(reg), state)
    assert {"id": "nuget:newtonsoft.json@1.0.0", "_deleted": True} in rows


def test_missing_package_and_dropped_selection_are_tombstoned():
    state = {}
    reg = {"newtonsoft.json": _registration("c1", [_entry("1.0.0")])}
    _run(_client(reg), state)
    rows, _ = _run(_client({}), state)
    assert {r["id"] for r in rows} == {"nuget:newtonsoft.json", "nuget:newtonsoft.json@1.0.0"}
    assert all(r["_deleted"] for r in rows)
    assert state["packages"] == {}


def test_server_error_raises_and_forgets_nothing():
    state = {}
    reg = {"newtonsoft.json": _registration("c1", [_entry("1.0.0")])}
    _run(_client(reg), state)
    failing = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (
                httpx.Response(
                    200,
                    json={"resources": [{"@id": BASE, "@type": "RegistrationsBaseUrl/Versioned"}]},
                )
                if str(r.url) == INDEX
                else httpx.Response(500)
            )
        )
    )
    with pytest.raises(httpx.HTTPStatusError):
        _run(failing, state)
    assert "newtonsoft.json" in state["packages"]


def test_selection_is_validated_and_resource_is_a_document_source():
    with pytest.raises(ValueError):
        nuget_source(resource_name="n")
    with pytest.raises(ValueError):
        nuget_source(resource_name="n", package_ids=["A"], query="x")
    resource = nuget_source(resource_name="n", package_ids=["A"], client=_client({}))
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "nuget"


def test_project_file_package_ids(tmp_path):
    from cognee.tasks.ingestion.connectors.nuget import _ids_from_project_file

    csproj = tmp_path / "a.csproj"
    csproj.write_text(
        '<Project><ItemGroup><PackageReference Include="Serilog" Version="3"/></ItemGroup></Project>'
    )
    assert _ids_from_project_file(csproj) == ["Serilog"]
    lock = tmp_path / "packages.lock.json"
    lock.write_text('{"dependencies": {"net8.0": {"Dapper": {"resolved": "2.0.0"}}}}')
    assert _ids_from_project_file(lock) == ["Dapper"]
