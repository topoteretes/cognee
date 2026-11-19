"""Unit tests for the NuGet v3 connector and DLT source."""

import pytest
import httpx
from cognee_connectors_core.registry import use_connector, describe, get_connector
from cognee_connector_nuget import CONNECTOR, nuget_source
from cognee_connector_nuget.source import _fetch_package_data


def test_nuget_connector_registration():
    connector = use_connector(CONNECTOR)
    assert get_connector("nuget") is connector

    desc = describe(connector)
    assert desc["provider"] == "nuget"
    assert desc["display_name"] == "NuGet"
    assert desc["auth"] == "none"


def test_fetch_package_data_success(monkeypatch):
    sample_response = {
        "count": 1,
        "items": [
            {
                "items": [
                    {
                        "catalogEntry": {
                            "id": "Newtonsoft.Json",
                            "version": "13.0.3",
                            "authors": "James Newton-King",
                            "description": "Json.NET is a popular high-performance JSON framework for .NET",
                            "published": "2023-03-08T00:00:00Z",
                            "dependencyGroups": [
                                {
                                    "targetFramework": "net6.0",
                                    "dependencies": [],
                                }
                            ],
                            "deprecation": {
                                "reasons": ["Legacy"],
                                "message": "Consider System.Text.Json",
                                "alternatePackage": {
                                    "id": "System.Text.Json",
                                    "range": "[7.0.0, )"
                                }
                            },
                            "vulnerabilities": [
                                {
                                    "advisoryUrl": "https://github.com/advisories/GHSA-5crp-9r3c-p9vr",
                                    "severity": "2"
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    def mock_get(self, url, **kwargs):
        if "newtonsoft.json" in url:
            return httpx.Response(200, json=sample_response)
        return httpx.Response(404)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    with httpx.Client() as client:
        data = _fetch_package_data(client, "Newtonsoft.Json")

    assert data is not None
    assert data["package_id"] == "Newtonsoft.Json"
    assert data["id"] == "newtonsoft.json"
    assert data["latest_version"] == "13.0.3"
    assert data["authors"] == "James Newton-King"
    assert data["deprecation"]["alternatePackage"]["id"] == "System.Text.Json"
    assert len(data["vulnerabilities"]) == 1
    assert len(data["versions"]) == 1


def test_fetch_package_data_not_found(monkeypatch):
    def mock_get(self, url, **kwargs):
        return httpx.Response(404)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    with httpx.Client() as client:
        data = _fetch_package_data(client, "NonExistentPackage123")

    assert data is None
